import {
  type ContentFetcherRegistry,
  DEFAULT_PIPELINE,
  DataSource,
  DepthType,
  ContentFetcherRegistry as FetcherRegistry,
  GraphTraverser,
  InMemoryMetaIndexStore,
  InMemoryVectorStore,
  KeywordMappingStrategy,
  type LLMAdapter,
  LayeredContextCollector,
  LayeredQueryPipeline,
  OntologyGraph,
  RouterLLMAdapter,
  ScoreBasedSelector,
  TraversalStrategy,
  VectorStepExecutor,
  createMetaDocument,
  createNode,
  resourceDocumentIdentity,
  step,
} from "@kontext-brain/core";
import { describe, expect, it } from "vitest";

class MockLLMAdapter implements LLMAdapter {
  async complete(_system: string, _context: string, _query: string): Promise<string> {
    return "Mock answer based on the retrieved context.";
  }
}

describe("End-to-end LayeredQueryPipeline", () => {
  it("executes default 3-layer pipeline and returns an answer", async () => {
    const nodes = new Map([
      [
        "engineering",
        createNode({
          id: "engineering",
          description: "software engineering development api",
          weight: 1.0,
        }),
      ],
      [
        "operations",
        createNode({ id: "operations", description: "deploy infra monitoring", weight: 0.9 }),
      ],
    ]);
    const graph = new OntologyGraph(nodes, [], {
      maxDepth: 2,
      maxTokens: 2000,
      strategy: TraversalStrategy.WEIGHTED_DFS,
    });

    const metaIndex = new InMemoryMetaIndexStore();
    await metaIndex.index("engineering", [
      createMetaDocument({
        id: "doc-1",
        title: "API Design Guide",
        source: DataSource.NOTION,
        ontologyNodeId: "engineering",
      }),
    ]);

    const fetcherRegistry: ContentFetcherRegistry = new FetcherRegistry();
    fetcherRegistry.register({
      source: DataSource.NOTION,
      async fetch(doc) {
        return {
          metaDocumentId: doc.id,
          title: doc.title,
          body: "Rest API guidelines: use nouns for resources, verbs via HTTP methods.",
          source: doc.source,
          sectionContent: null,
          fetchedAt: new Date(),
        };
      },
    });

    const mockAdapter = new MockLLMAdapter();
    const router = new RouterLLMAdapter(mockAdapter, mockAdapter);
    const pipeline = new LayeredQueryPipeline(graph, router, metaIndex, fetcherRegistry, {
      mappingStrategy: new KeywordMappingStrategy(),
      metaSelector: new ScoreBasedSelector(),
    });

    const result = await pipeline.execute("engineering api design");
    expect(result.answer.length).toBeGreaterThan(0);
    expect(result.usedOntologyNodes.map((n) => n.id)).toContain("engineering");
  });

  it("retrieves documents for a leaf node without calling the reasoning LLM", async () => {
    const graph = new OntologyGraph(
      new Map([
        [
          "authentication",
          createNode({
            id: "authentication",
            description: "JWT authentication",
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
    const metaIndex = new InMemoryMetaIndexStore();
    await metaIndex.index("authentication", [
      createMetaDocument({
        id: "notion://jwt",
        title: "JWT key rotation",
        source: DataSource.NOTION,
        ontologyNodeId: "authentication",
      }),
    ]);
    const fetcherRegistry: ContentFetcherRegistry = new FetcherRegistry();
    fetcherRegistry.register({
      source: DataSource.NOTION,
      async fetch(document) {
        return {
          metaDocumentId: document.id,
          title: document.title,
          body: "Rotate JWT signing keys regularly.",
          source: document.source,
          fetchedAt: new Date(),
        };
      },
    });

    let reasoningCalls = 0;
    const traversal = new MockLLMAdapter();
    const reasoning: LLMAdapter = {
      async complete() {
        reasoningCalls++;
        return "Reasoned answer";
      },
    };
    const pipeline = new LayeredQueryPipeline(
      graph,
      new RouterLLMAdapter(traversal, reasoning),
      metaIndex,
      fetcherRegistry,
      {
        mappingStrategy: new KeywordMappingStrategy(),
        metaSelector: new ScoreBasedSelector(),
      },
    );

    const retrieval = await pipeline.retrieve("JWT authentication");
    expect(reasoningCalls).toBe(0);
    expect(retrieval.selectedMetaDocs.map((document) => document.id)).toEqual(["notion://jwt"]);
    expect(retrieval.fetchedContents).toHaveLength(1);
    expect(retrieval.context).toContain("Rotate JWT signing keys");

    await pipeline.answer("JWT authentication", retrieval);
    expect(reasoningCalls).toBe(1);
  });

  it("InMemoryVectorStore basic cosine similarity", async () => {
    // Deterministic tiny embedder for testing
    const embedder = async (text: string): Promise<Float32Array> => {
      const v = new Float32Array(4);
      for (const ch of text.toLowerCase()) {
        const idx = ch.charCodeAt(0) % 4;
        v[idx] = (v[idx] ?? 0) + 1;
      }
      return v;
    };
    const store = new InMemoryVectorStore(embedder);
    await store.upsert("a:apple", await store.embed("apple fruit red"));
    await store.upsert("a:orange", await store.embed("orange fruit citrus"));
    await store.upsert("a:car", await store.embed("automobile vehicle"));
    const results = await store.similaritySearchWithPrefix("apple", "a:", 2);
    expect(results.length).toBeGreaterThan(0);
  });

  it("routes content fetches to the connector recorded on the document", async () => {
    const fetchers = new FetcherRegistry();
    for (const connectorName of ["team-a", "team-b"]) {
      fetchers.register({
        source: DataSource.NOTION,
        connectorName,
        async fetch(document) {
          return {
            metaDocumentId: document.id,
            title: document.title,
            body: `fetched-from:${connectorName}`,
            source: document.source,
            fetchedAt: new Date(),
          };
        },
      });
    }

    const content = await fetchers.fetch(
      createMetaDocument({
        id: "notion://shared-id",
        title: "Shared document",
        source: DataSource.NOTION,
        ontologyNodeId: "engineering",
        metadata: { connector: "team-b" },
      }),
    );
    expect(content.body).toBe("fetched-from:team-b");
  });

  it("keeps connector identity and URI resource IDs intact in vector retrieval", async () => {
    const vectorStore = new InMemoryVectorStore(async () => new Float32Array([1, 0]));
    const metaIndex = new InMemoryMetaIndexStore();
    const sharedDocuments = ["team-a", "team-b"].map((connector) =>
      createMetaDocument({
        id: "notion://shared/page",
        title: `Shared document from ${connector}`,
        source: DataSource.NOTION,
        ontologyNodeId: "engineering",
        metadata: { connector },
      }),
    );
    await metaIndex.index("engineering", sharedDocuments);
    expect(await metaIndex.list("engineering")).toHaveLength(2);

    await vectorStore.upsert(
      `content:engineering:${resourceDocumentIdentity("team-b", "notion://shared/page")}`,
      new Float32Array([1, 0]),
    );
    const result = await new VectorStepExecutor().execute(
      {
        node: createNode({ id: "engineering", description: "Engineering" }),
        query: "shared document",
        accumulatedDocs: [],
        metaIndexStore: metaIndex,
        metaSelector: new ScoreBasedSelector(),
        fetcherRegistry: new FetcherRegistry(),
        vectorStore,
      },
      step({ depth: 1, type: DepthType.VECTOR, maxSelect: 5 }),
    );

    expect(result.selectedDocs).toHaveLength(1);
    expect(result.selectedDocs[0]?.metadata.connector).toBe("team-b");
    expect(result.selectedDocs[0]?.id).toBe("notion://shared/page");
  });

  it("GraphTraverser hierarchical expansion", () => {
    const nodes = new Map([
      ["root", createNode({ id: "root", description: "", weight: 1.0 })],
      [
        "child1",
        createNode({ id: "child1", description: "", weight: 0.9, parentId: "root", level: 1 }),
      ],
      [
        "child2",
        createNode({ id: "child2", description: "", weight: 0.8, parentId: "root", level: 1 }),
      ],
    ]);
    const graph = new OntologyGraph(nodes, [], {
      maxDepth: 3,
      maxTokens: 1000,
      strategy: TraversalStrategy.WEIGHTED_DFS,
    });
    const result = new GraphTraverser(graph).traverse(["root"]);
    expect(result.nodes.map((n) => n.node.id)).toEqual(["root", "child1", "child2"]);
  });
});
