import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  BidirectionalNLayerRetriever,
  ContentFetcherRegistry,
  DataSource,
  FileOntologyStore,
  InMemoryKnowledgeGraphRepository,
  InMemoryMetaIndexStore,
  InMemoryOntologyStore,
  InMemoryResourceContentStore,
  InMemoryVectorStore,
  IngestPipeline,
  KeywordMappingStrategy,
  type LLMAdapter,
  OntologyGraph,
  RouterLLMAdapter,
  ScoreBasedSelector,
  SyncResourceUseCase,
  TraversalStrategy,
  createNode,
} from "@kontext-brain/core";
import type { LLMProviderRegistry } from "@kontext-brain/llm";
import {
  KontextAgent,
  KontextConfigSchema,
  KontextLoader,
  computeOntologyContentHash,
} from "@kontext-brain/loader";
import {
  GenericMCPResourceSnapshotAdapter,
  type MCPConnector,
  MCPContentFetcherBridge,
  type MCPData,
  MCPKnowledgeSynchronizer,
  MCPLayerAdapterFactory,
  type MCPResource,
} from "@kontext-brain/mcp";
import { afterEach, describe, expect, it } from "vitest";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  for (const directory of temporaryDirectories.splice(0)) {
    await rm(directory, { recursive: true, force: true });
  }
});

class MutableConnector implements MCPConnector {
  readonly name = "notion";
  resources: MCPResource[] = [];
  shouldFail = false;

  async listResources(): Promise<MCPResource[]> {
    if (this.shouldFail) throw new Error("connector unavailable");
    return [...this.resources];
  }

  async fetchResource(resourceId: string): Promise<MCPData> {
    return {
      resourceId,
      content: `Body for ${resourceId}`,
      metadata: {},
      fetchedAt: new Date(),
    };
  }

  async search(): Promise<MCPData[]> {
    return [];
  }
}

class StateTestLLM implements LLMAdapter {
  classificationCalls = 0;

  async complete(systemPrompt: string, _context: string, _query: string): Promise<string> {
    if (systemPrompt.includes("Classify each document")) {
      this.classificationCalls++;
      return JSON.stringify({
        mappings: { backend: [0] },
        unmapped: [],
      });
    }
    if (systemPrompt.includes("Extract entities and relationships")) {
      return JSON.stringify({
        nodes: [
          {
            id: "manual-topic",
            description: "Created by manual ingest",
            weight: 0.8,
          },
        ],
        edges: [],
      });
    }
    return "answer";
  }
}

function buildGraph(): OntologyGraph {
  return new OntologyGraph(
    new Map([
      [
        "backend",
        createNode({
          id: "backend",
          description: "backend API authentication",
        }),
      ],
      [
        "frontend",
        createNode({
          id: "frontend",
          description: "frontend React UI",
        }),
      ],
    ]),
    [],
    {
      maxDepth: 2,
      maxTokens: 2000,
      strategy: TraversalStrategy.WEIGHTED_DFS,
    },
  );
}

describe("KontextAgent unified state", () => {
  it("classifies incremental MCP resources once and removes deleted resources", async () => {
    const connector = new MutableConnector();
    connector.resources = [
      {
        id: "notion://api",
        name: "Backend API Guide",
        description: "Authentication and REST",
      },
    ];
    const adapter = MCPLayerAdapterFactory.notion(connector);
    const fetchers = new ContentFetcherRegistry();
    fetchers.register(new MCPContentFetcherBridge(adapter));
    const llm = new StateTestLLM();
    const store = new InMemoryOntologyStore();
    const metaIndex = new InMemoryMetaIndexStore();
    const vectorStore = new InMemoryVectorStore(async () => new Float32Array([1, 0]));
    const knowledgeRepository = new InMemoryKnowledgeGraphRepository();
    const knowledgeContent = new InMemoryResourceContentStore();
    const knowledgeSync = new MCPKnowledgeSynchronizer(
      new SyncResourceUseCase(knowledgeRepository, knowledgeContent),
      [new GenericMCPResourceSnapshotAdapter("notion", "notion", { organizationWide: true })],
    );
    const agent = new KontextAgent({
      graph: buildGraph(),
      router: new RouterLLMAdapter(llm, llm),
      mcpConnectors: [connector],
      mcpLayerAdapters: [adapter],
      metaIndexStore: metaIndex,
      fetcherRegistry: fetchers,
      vectorStore,
      mappingStrategy: new KeywordMappingStrategy(),
      metaSelector: new ScoreBasedSelector(),
      ingestPipeline: new IngestPipeline(llm, store, vectorStore),
      ontologyStore: store,
      stateId: "test-agent",
      organizationId: "acme",
      mcpKnowledgeSynchronizer: knowledgeSync,
    });

    const first = await agent.syncMCP();
    expect(first.resourcesAdded).toBe(1);
    expect(first.resourcesClassified).toBe(1);
    expect(llm.classificationCalls).toBe(1);
    expect((await metaIndex.list("backend")).map((doc) => doc.id)).toEqual(["notion://api"]);
    expect(await metaIndex.list("frontend")).toEqual([]);
    expect(
      (
        await knowledgeRepository.getResourceBySource("acme", {
          connectorId: "notion",
          externalId: "notion://api",
          type: "notion",
        })
      )?.status,
    ).toBe("active");

    const unchanged = await agent.syncMCP();
    expect(unchanged.resourcesAdded).toBe(0);
    expect(unchanged.resourcesUpdated).toBe(0);
    expect(llm.classificationCalls).toBe(1);

    const persisted = await store.load("test-agent");
    expect(persisted.resources).toHaveLength(1);
    expect(persisted.metaDocuments?.backend).toHaveLength(1);

    connector.shouldFail = true;
    const unavailable = await agent.syncMCP();
    expect(unavailable.connectorsSynced).toBe(0);
    expect(unavailable.resourcesRemoved).toBe(0);
    expect((await metaIndex.list("backend")).map((doc) => doc.id)).toEqual(["notion://api"]);

    connector.shouldFail = false;
    connector.resources = [
      {
        id: "notion://api",
        name: "Backend API Guide",
        description: "Updated authentication and REST",
      },
    ];
    const updated = await agent.syncMCP();
    expect(updated.resourcesUpdated).toBe(1);
    expect(updated.resourcesClassified).toBe(1);
    expect(llm.classificationCalls).toBe(2);

    connector.resources = [];
    const removed = await agent.syncMCP();
    expect(removed.resourcesRemoved).toBe(1);
    expect(await metaIndex.list("backend")).toEqual([]);
    expect((await store.load("test-agent")).resources).toEqual([]);
    expect(
      (
        await knowledgeRepository.getResourceBySource("acme", {
          connectorId: "notion",
          externalId: "notion://api",
          type: "notion",
        })
      )?.status,
    ).toBe("stale");
  });

  it("applies manual ingest to both the runtime graph and persisted snapshot", async () => {
    const llm = new StateTestLLM();
    const store = new InMemoryOntologyStore();
    const vectorStore = new InMemoryVectorStore(async () => new Float32Array([1, 0]));
    const agent = new KontextAgent({
      graph: buildGraph(),
      router: new RouterLLMAdapter(llm, llm),
      mcpConnectors: [],
      mcpLayerAdapters: [],
      metaIndexStore: new InMemoryMetaIndexStore(),
      fetcherRegistry: new ContentFetcherRegistry(),
      vectorStore,
      mappingStrategy: new KeywordMappingStrategy(),
      metaSelector: new ScoreBasedSelector(),
      ingestPipeline: new IngestPipeline(llm, store, vectorStore),
      ontologyStore: store,
      stateId: "test-agent",
    });

    await agent.ingest("A new topic");

    expect(agent.ontologyGraph.nodes.has("manual-topic")).toBe(true);
    expect((await store.load("test-agent")).nodes["manual-topic"]).toBeDefined();
  });

  it("uses bounded bidirectional retrieval when a Knowledge search graph is configured", async () => {
    const llm = new StateTestLLM();
    const store = new InMemoryOntologyStore();
    const vectorStore = new InMemoryVectorStore(async () => new Float32Array([1, 0]));
    const knowledgeRetriever = new BidirectionalNLayerRetriever({
      async seed() {
        return [{ node: { kind: "chunk" as const, id: "slack:message-1" }, score: 1 }];
      },
      async neighbors() {
        return [];
      },
      async evidence() {
        return [
          {
            evidenceId: "evidence-1",
            resourceId: "slack:thread-1",
            chunkId: "slack:message-1",
            text: "Order 42 was paid",
            score: 1,
          },
        ];
      },
    });
    const citedLlm: LLMAdapter = {
      async complete() {
        return "Order 42 was paid [Evidence evidence-1]";
      },
    };
    const agent = new KontextAgent({
      graph: buildGraph(),
      router: new RouterLLMAdapter(llm, citedLlm),
      mcpConnectors: [],
      mcpLayerAdapters: [],
      metaIndexStore: new InMemoryMetaIndexStore(),
      fetcherRegistry: new ContentFetcherRegistry(),
      vectorStore,
      mappingStrategy: new KeywordMappingStrategy(),
      metaSelector: new ScoreBasedSelector(),
      ingestPipeline: new IngestPipeline(llm, store, vectorStore),
      ontologyStore: store,
      stateId: "test-agent",
      organizationId: "acme",
      knowledgeRetriever,
    });
    const principal = { organizationId: "acme", subjectId: "u1", groupIds: ["finance"] };

    const retrieval = await agent.retrieve("Was order 42 paid?", principal);
    const answered = await agent.answer("Was order 42 paid?", principal);

    expect(retrieval.retrievalMode).toBe("bidirectional");
    expect(retrieval.evidence?.[0]?.evidenceId).toBe("evidence-1");
    expect(answered.answer).toContain("evidence-1");
    await expect(
      agent.retrieve("Was order 42 paid?", { ...principal, organizationId: "other" }),
    ).rejects.toThrow("Organization mismatch");
  });

  it("round-trips the unified snapshot through FileOntologyStore", async () => {
    const directory = await mkdtemp(join(tmpdir(), "kontext-state-"));
    temporaryDirectories.push(directory);
    const store = new FileOntologyStore(directory);
    await store.save("agent", {
      userId: "agent",
      nodes: {
        backend: {
          id: "backend",
          description: "Backend",
          weight: 1,
          webSearch: false,
        },
      },
      edges: [],
      metaDocuments: {
        backend: [
          {
            id: "custom://api",
            title: "API",
            source: DataSource.CUSTOM,
            ontologyNodeId: "backend",
            score: 1,
            metadata: { connector: "custom" },
            fetchedAt: "2026-01-01T00:00:00.000Z",
          },
        ],
      },
      resources: [
        {
          connectorName: "custom",
          resourceId: "custom://api",
          title: "API",
          description: "API docs",
          source: DataSource.CUSTOM,
          nodeIds: ["backend"],
          signature: "signature",
          lastSeenAt: "2026-01-01T00:00:00.000Z",
        },
      ],
    });

    const restored = await store.load("agent");
    expect(restored.nodes.backend?.description).toBe("Backend");
    expect(restored.resources?.[0]?.nodeIds).toEqual(["backend"]);
  });

  it("does not treat a corrupt persisted snapshot as an empty graph", async () => {
    const directory = await mkdtemp(join(tmpdir(), "kontext-corrupt-"));
    temporaryDirectories.push(directory);
    await writeFile(join(directory, "agent.json"), "{ invalid json", "utf-8");

    await expect(new FileOntologyStore(directory).load("agent")).rejects.toThrow();
  });

  it("hydrates the graph and meta index when KontextLoader restarts", async () => {
    const directory = await mkdtemp(join(tmpdir(), "kontext-reload-"));
    temporaryDirectories.push(directory);
    const store = new FileOntologyStore(directory);
    await store.save("default", {
      userId: "default",
      nodes: {
        backend: {
          id: "backend",
          description: "Backend API",
          weight: 1,
          webSearch: false,
        },
      },
      edges: [],
      metaDocuments: {
        backend: [
          {
            id: "custom://api",
            title: "Backend API Guide",
            source: DataSource.CUSTOM,
            ontologyNodeId: "backend",
            score: 1,
            metadata: { connector: "custom" },
            fetchedAt: "2026-01-01T00:00:00.000Z",
          },
        ],
      },
      resources: [
        {
          connectorName: "custom",
          resourceId: "custom://api",
          title: "Backend API Guide",
          description: "API docs",
          source: DataSource.CUSTOM,
          nodeIds: ["backend"],
          signature: "signature",
          lastSeenAt: "2026-01-01T00:00:00.000Z",
        },
      ],
    });

    const fakeRegistry = {
      createChat() {
        return {
          async invoke() {
            return { content: "[]" };
          },
        };
      },
      createEmbedding() {
        throw new Error("No embeddings in this test");
      },
    } as unknown as LLMProviderRegistry;
    const config = KontextConfigSchema.parse({
      llm: {
        traversal: { provider: "test", model: "test" },
        reasoning: { provider: "test", model: "test" },
      },
      storage: { type: "file", path: directory },
      graph: {
        maxDepth: 2,
        maxTokens: 2000,
        strategy: "WEIGHTED_DFS",
      },
    });

    const agent = await new KontextLoader({ llmRegistry: fakeRegistry }).from(config);
    const retrieval = await agent.retrieve("backend API");

    expect(agent.ontologyGraph.nodes.has("backend")).toBe(true);
    expect(retrieval.selectedMetaDocs.map((document) => document.id)).toEqual(["custom://api"]);
  });

  it("activates changed YAML instead of silently keeping the persisted graph", async () => {
    const directory = await mkdtemp(join(tmpdir(), "kontext-yaml-change-"));
    temporaryDirectories.push(directory);
    const store = new FileOntologyStore(directory);
    await store.save("default", {
      userId: "default",
      nodes: {
        legacy: { id: "legacy", description: "Old", weight: 1, webSearch: false },
      },
      edges: [],
      ontologyContentHash: computeOntologyContentHash([{ id: "legacy", description: "Old" }]),
    });
    const fakeRegistry = {
      createChat() {
        return {
          async invoke() {
            return { content: "[]" };
          },
        };
      },
      createEmbedding() {
        throw new Error("No embeddings in this test");
      },
    } as unknown as LLMProviderRegistry;
    const config = KontextConfigSchema.parse({
      llm: {
        traversal: { provider: "test", model: "test" },
        reasoning: { provider: "test", model: "test" },
      },
      storage: { type: "file", path: directory },
      ontology: [{ id: "order", description: "Customer orders" }],
    });

    const agent = await new KontextLoader({ llmRegistry: fakeRegistry }).from(config);

    expect([...agent.ontologyGraph.nodes.keys()]).toEqual(["order"]);
    expect((await store.load("default")).ontologyContentHash).toBe(
      computeOntologyContentHash(config.ontology),
    );
  });

  it("preserves the active graph when changed YAML has an invalid relation", async () => {
    const directory = await mkdtemp(join(tmpdir(), "kontext-yaml-invalid-"));
    temporaryDirectories.push(directory);
    const store = new FileOntologyStore(directory);
    await store.save("default", {
      userId: "default",
      nodes: {
        order: { id: "order", description: "Orders", weight: 1, webSearch: false },
      },
      edges: [],
      ontologyContentHash: "active-hash",
    });
    const fakeRegistry = {
      createChat() {
        return {
          async invoke() {
            return { content: "[]" };
          },
        };
      },
      createEmbedding() {
        throw new Error("No embeddings in this test");
      },
    } as unknown as LLMProviderRegistry;
    const config = KontextConfigSchema.parse({
      llm: {
        traversal: { provider: "test", model: "test" },
        reasoning: { provider: "test", model: "test" },
      },
      storage: { type: "file", path: directory },
      ontology: [{ id: "order", description: "Orders", relates: [{ to: "missing" }] }],
    });

    await expect(new KontextLoader({ llmRegistry: fakeRegistry }).from(config)).rejects.toThrow(
      "unknown ontology node",
    );
    expect((await store.load("default")).ontologyContentHash).toBe("active-hash");
  });
});
