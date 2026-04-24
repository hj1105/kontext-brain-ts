/**
 * autoSetup E2E example with in-process mock MCP connectors.
 *
 * Shows the full flow:
 *   1. Connect multiple MCP sources (Notion-like + GitHub-like, both mocked in-process)
 *   2. Call autoSetup() -> LLM builds the ontology + classifies documents
 *   3. Query the agent
 *
 * Run: pnpm --filter @kontext-brain/example-auto-setup start
 */

import {
  type ContentFetcherRegistry,
  ContentFetcherRegistry as FetcherRegistry,
  DataSource,
  DEFAULT_PIPELINE,
  InMemoryMetaIndexStore,
  InMemoryOntologyStore,
  InMemoryVectorStore,
  IngestPipeline,
  KeywordMappingStrategy,
  type LLMAdapter,
  OntologyGraph,
  RouterLLMAdapter,
  ScoreBasedSelector,
  TraversalStrategy,
} from "@kontext-brain/core";
import { KontextAgent } from "@kontext-brain/loader";
import {
  type MCPConnector,
  type MCPData,
  type MCPResource,
  MCPContentFetcherBridge,
  MCPLayerAdapterFactory,
} from "@kontext-brain/mcp";

// ── Scripted LLM: returns JSON matching autoSetup prompts ─────

class ScriptedLLM implements LLMAdapter {
  async complete(_system: string, _context: string, query: string): Promise<string> {
    if (query.toLowerCase().includes("extract topic categories")) {
      return '["Backend", "Frontend"]';
    }
    if (query.toLowerCase().includes("design ontology nodes")) {
      return JSON.stringify({
        nodes: [
          { id: "Engineering", description: "software dev api", weight: 1.0, level: 0, parentId: null },
          { id: "Backend", description: "api server db", weight: 0.9, level: 1, parentId: "Engineering" },
          { id: "Frontend", description: "ui react components", weight: 0.8, level: 1, parentId: "Engineering" },
        ],
      });
    }
    if (query.toLowerCase().includes("infer relationships")) {
      return JSON.stringify({ edges: [{ from: "Backend", to: "Frontend", weight: 0.6 }] });
    }
    if (query.toLowerCase().includes("classify each document")) {
      return JSON.stringify({
        mappings: { Backend: [0, 1, 3], Frontend: [2, 4] },
        unmapped: [],
      });
    }
    return "Mock reasoned answer based on retrieved context.";
  }
}

// ── Mock in-process MCP connectors ───────────────────────────

class MockNotionConnector implements MCPConnector {
  readonly name = "notion";
  async listResources(): Promise<MCPResource[]> {
    return [
      { id: "n-1", name: "API Design Guide", description: "REST best practices" },
      { id: "n-2", name: "Database Schema", description: "Postgres schema" },
      { id: "n-3", name: "Frontend Components", description: "React component library" },
    ];
  }
  async fetchResource(resourceId: string): Promise<MCPData> {
    return {
      resourceId,
      content: `Notion content for ${resourceId}`,
      metadata: {},
      fetchedAt: new Date(),
    };
  }
  async search(): Promise<MCPData[]> {
    return [];
  }
}

class MockSlackConnector implements MCPConnector {
  readonly name = "slack";
  async listResources(): Promise<MCPResource[]> {
    return [
      { id: "s-1", name: "backend-help", description: "Backend engineering channel" },
      { id: "s-2", name: "frontend-questions", description: "Frontend and UI channel" },
    ];
  }
  async fetchResource(resourceId: string): Promise<MCPData> {
    return {
      resourceId,
      content: `Slack content for ${resourceId}`,
      metadata: {},
      fetchedAt: new Date(),
    };
  }
  async search(): Promise<MCPData[]> {
    return [];
  }
}

async function main(): Promise<void> {
  const notionConnector = new MockNotionConnector();
  const slackConnector = new MockSlackConnector();
  const connectors: MCPConnector[] = [notionConnector, slackConnector];

  // Layer adapters
  const notionAdapter = MCPLayerAdapterFactory.notion(notionConnector);
  const slackAdapter = MCPLayerAdapterFactory.slack(slackConnector);
  const adapters = [notionAdapter, slackAdapter];

  // Content fetcher registry
  const fetcherRegistry: ContentFetcherRegistry = new FetcherRegistry();
  for (const a of adapters) {
    fetcherRegistry.register(new MCPContentFetcherBridge(a));
  }

  // Empty graph — autoSetup will populate it
  const graph = new OntologyGraph(new Map(), [], {
    maxDepth: 2,
    maxTokens: 2000,
    strategy: TraversalStrategy.WEIGHTED_DFS,
  });

  const llm = new ScriptedLLM();
  const router = new RouterLLMAdapter(llm, llm);
  // Tiny deterministic embedder to avoid needing a real embeddings API
  const vectorStore = new InMemoryVectorStore(async (text) => {
    const v = new Float32Array(8);
    for (const ch of text.toLowerCase()) {
      const idx = ch.charCodeAt(0) % 8;
      v[idx] = (v[idx] ?? 0) + 1;
    }
    return v;
  });
  const metaIndex = new InMemoryMetaIndexStore();
  const ingestPipeline = new IngestPipeline(llm, new InMemoryOntologyStore(), vectorStore);

  const agent = new KontextAgent({
    graph,
    router,
    mcpConnectors: connectors,
    mcpLayerAdapters: adapters,
    metaIndexStore: metaIndex,
    fetcherRegistry,
    vectorStore,
    mappingStrategy: new KeywordMappingStrategy(),
    metaSelector: new ScoreBasedSelector(),
    ingestPipeline,
    pipeline: DEFAULT_PIPELINE,
  });

  console.log("Before autoSetup:", agent.ontologyGraph.nodes.size, "nodes");

  const setup = await agent.autoSetup(3);
  console.log("\nautoSetup result:");
  console.log("  Nodes created:   ", setup.nodesCreated);
  console.log("  Docs classified: ", setup.documentsClassified);
  console.log("  Docs unmapped:   ", setup.documentsUnmapped);
  console.log("\nGraph now has:", agent.ontologyGraph.nodes.size, "nodes");
  console.log("\nGenerated YAML:\n" + setup.ontologyYaml.slice(0, 400) + "...");

  console.log("\n--- Query ---");
  const answer = await agent.query("How should I design backend APIs?");
  console.log("Answer:", answer.answer);
  console.log(
    "Sources:",
    answer.selectedMetaDocs.map((d) => `${d.title} (${d.source})`),
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
