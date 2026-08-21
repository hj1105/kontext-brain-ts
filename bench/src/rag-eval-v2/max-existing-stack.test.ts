import { describe, expect, it } from "vitest";
import { CorpusBm25Ranker, fuseRankings } from "./max-existing-stack.js";

describe("CorpusBm25Ranker", () => {
  it("promotes a document containing the rare query term", () => {
    const ranker = new CorpusBm25Ranker([
      { id: "generic", text: "common treatment and common symptoms" },
      { id: "answer", text: "xanthopsia is a rare visual symptom" },
      { id: "other", text: "common anatomy reference" },
    ]);

    expect(ranker.rank("What causes xanthopsia?", 3)[0]).toBe("answer");
  });
});

describe("fuseRankings", () => {
  it("combines vector, graph, BM25, and context-rerank evidence deterministically", () => {
    const result = fuseRankings(
      [
        { name: "vector", ids: ["vector-only", "shared"], weight: 1 },
        { name: "graph", ids: ["graph-only", "shared"], weight: 1 },
        { name: "bm25", ids: ["shared", "lexical-only"], weight: 1 },
        { name: "context-rerank", ids: ["shared", "graph-only"], weight: 1 },
      ],
      3,
    );

    expect(result.map((item) => item.id)).toEqual(["shared", "graph-only", "vector-only"]);
    expect(result[0]?.sourceRanks).toEqual({
      vector: 2,
      graph: 2,
      bm25: 1,
      "context-rerank": 1,
    });
  });
});
