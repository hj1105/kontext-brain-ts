import { describe, expect, it } from "vitest";
import { BidirectionalBenchmarkSearchGraph } from "../bidirectional-benchmark-search-graph.js";

describe("BidirectionalBenchmarkSearchGraph adaptive observations", () => {
  it("derives chunk seed consensus from the providers available for that query", async () => {
    const graph = new BidirectionalBenchmarkSearchGraph(
      { entities: new Map(), edges: [], chunkToEntities: new Map() },
      [
        { id: "both", title: "alpha", body: "alpha evidence" },
        { id: "vector-only", title: "beta", body: "beta evidence" },
        { id: "lexical-only", title: "alpha extra", body: "alpha secondary" },
      ],
      [{ question: "alpha", chunkIds: ["both", "vector-only", "lexical-only"] }],
      { seedChunks: 3, lexicalSeedChunks: 2, providerConsensus: true },
    );

    const seeds = await graph.seed("alpha", {
      organizationId: "test",
      subjectId: "test",
      groupIds: [],
    });
    const byId = new Map(seeds.map((seed) => [seed.node.id, seed]));

    expect(byId.get("both")?.observations).toMatchObject({
      providers: ["benchmark-lexical", "benchmark-vector"],
      query: { rerankerScore: 1 },
    });
    expect(byId.get("vector-only")?.observations?.query?.rerankerScore).toBeLessThan(0.5);
  });

  it("reports full resource fanout and query-local rank on a bounded chunk list", async () => {
    const docs = Array.from({ length: 100 }, (_, index) => ({
      id: `guide-${index}`,
      title: `guide chunk ${index}`,
      body: index === 0 ? "specific treatment answer" : `unrelated material ${index}`,
    }));
    const graph = new BidirectionalBenchmarkSearchGraph(
      {
        entities: new Map(),
        edges: [],
        chunkToEntities: new Map(),
      },
      docs,
      [],
      { queryAware: true, resourceChunks: 10 },
    );

    const edges = await graph.neighbors({ kind: "resource", id: "guide" }, "specific treatment", {
      organizationId: "test",
      subjectId: "test",
      groupIds: [],
    });

    expect(edges).toHaveLength(10);
    expect(edges[0]).toMatchObject({
      to: { kind: "chunk", id: "guide-0" },
      observations: {
        query: { lexical: { rank: 1, candidateCount: 100 } },
        fanout: { returnedCount: 10, candidateCount: 100 },
        supportApplicability: "not-applicable",
      },
    });
  });
});
