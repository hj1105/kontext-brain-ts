import { describe, expect, it } from "vitest";
import {
  DEFAULT_PIPELINE,
  type ContentFetcherRegistry,
  ContentFetcherRegistry as FetcherRegistry,
  DataSource,
  InMemoryMetaIndexStore,
  KeywordMappingStrategy,
  type LLMAdapter,
  OntologyGraph,
  RouterLLMAdapter,
  ScoreBasedSelector,
  TraversalStrategy,
  createNode,
  IngestPipeline,
  InMemoryOntologyStore,
  InMemoryVectorStore,
} from "@kontext-brain/core";
import { KontextAgent } from "@kontext-brain/loader";
import { KontextToolServer } from "@kontext-brain/tool-server";

class StubLLM implements LLMAdapter {
  async complete(): Promise<string> {
    return "stub answer";
  }
}

function buildAgent(): KontextAgent {
  const nodes = new Map([
    ["docs", createNode({ id: "docs", description: "documentation guides", weight: 1.0 })],
  ]);
  const graph = new OntologyGraph(nodes, [], {
    maxDepth: 2,
    maxTokens: 1000,
    strategy: TraversalStrategy.WEIGHTED_DFS,
  });
  const stub = new StubLLM();
  const router = new RouterLLMAdapter(stub, stub);
  const fetcherRegistry: ContentFetcherRegistry = new FetcherRegistry();
  const vectorStore = new InMemoryVectorStore(async () => new Float32Array(4));
  return new KontextAgent({
    graph,
    router,
    mcpConnectors: [],
    mcpLayerAdapters: [],
    metaIndexStore: new InMemoryMetaIndexStore(),
    fetcherRegistry,
    vectorStore,
    mappingStrategy: new KeywordMappingStrategy(),
    metaSelector: new ScoreBasedSelector(),
    ingestPipeline: new IngestPipeline(stub, new InMemoryOntologyStore(), vectorStore),
    pipeline: DEFAULT_PIPELINE,
  });
}

describe("KontextToolServer construction", () => {
  it("constructs with an agent without throwing", () => {
    const agent = buildAgent();
    const server = new KontextToolServer(agent);
    expect(server).toBeDefined();
  });

  it("agent.describeGraph() includes ontology section", () => {
    const agent = buildAgent();
    const desc = agent.describeGraph();
    expect(desc).toContain("Ontology Graph");
    expect(desc).toContain("docs");
  });
});
