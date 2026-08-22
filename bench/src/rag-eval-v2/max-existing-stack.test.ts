import { describe, expect, it } from "vitest";
import { CorpusBm25Ranker, fuseQueryPerspectives, fuseRankings } from "./max-existing-stack.js";

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

  it("anchors expanded-query fusion on the original query without suppressing agreement", () => {
    const result = fuseQueryPerspectives(
      ["original-first", "shared"],
      [["expanded-only", "shared"]],
      {
        limit: 3,
        originalQueryWeight: 2,
        expandedQueryWeight: 1,
        reciprocalRankConstant: 10,
      },
    );

    expect(result.map((item) => item.id)).toEqual(["shared", "original-first", "expanded-only"]);
  });

  it("preserves equal-weight perspective fusion when both weights remain one", () => {
    const original = ["original-first", "shared"];
    const expansions = [["expanded-first", "shared"]];

    expect(
      fuseQueryPerspectives(original, expansions, {
        limit: 3,
        originalQueryWeight: 1,
        expandedQueryWeight: 1,
        reciprocalRankConstant: 10,
      }),
    ).toEqual(
      fuseRankings(
        [
          { name: "vector", ids: original, weight: 1 },
          { name: "vector", ids: expansions[0] ?? [], weight: 1 },
        ],
        3,
        10,
      ),
    );
  });
});
