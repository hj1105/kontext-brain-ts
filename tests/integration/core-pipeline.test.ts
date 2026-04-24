import { describe, expect, it } from "vitest";
import {
  type ContentFetcherRegistry,
  ContentFetcherRegistry as FetcherRegistry,
  DEFAULT_PIPELINE,
  GraphTraverser,
  InMemoryMetaIndexStore,
  InMemoryVectorStore,
  KeywordMappingStrategy,
  LayeredContextCollector,
  LayeredQueryPipeline,
  type LLMAdapter,
  OntologyGraph,
  RouterLLMAdapter,
  ScoreBasedSelector,
  TraversalStrategy,
  createNode,
  createMetaDocument,
  DataSource,
} from "@kontext-brain/core";

class MockLLMAdapter implements LLMAdapter {
  async complete(_system: string, _context: string, _query: string): Promise<string> {
    return "Mock answer based on the retrieved context.";
  }
}

describe("End-to-end LayeredQueryPipeline", () => {
  it("executes default 3-layer pipeline and returns an answer", async () => {
    const nodes = new Map([
      ["engineering", createNode({ id: "engineering", description: "software engineering development api", weight: 1.0 })],
      ["operations", createNode({ id: "operations", description: "deploy infra monitoring", weight: 0.9 })],
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

  it("GraphTraverser hierarchical expansion", () => {
    const nodes = new Map([
      ["root", createNode({ id: "root", description: "", weight: 1.0 })],
      ["child1", createNode({ id: "child1", description: "", weight: 0.9, parentId: "root", level: 1 })],
      ["child2", createNode({ id: "child2", description: "", weight: 0.8, parentId: "root", level: 1 })],
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
