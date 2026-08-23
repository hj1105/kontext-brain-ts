import { describe, expect, it } from "vitest";
import {
  CorpusBm25Ranker,
  applyOriginalAndExpansionQuota,
  fuseQueryPerspectives,
  fuseRankings,
} from "./max-existing-stack.js";

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

  it("reserves deterministic top-window coverage for the original and each expansion", () => {
    const ids = [
      "base-only",
      "e1-first",
      "original-5",
      "e2-first",
      "original-1",
      "original-2",
      "e3-first",
      "original-3",
      "original-4",
      "tail",
      "later",
    ];
    const base = ids.map((id, index) => ({
      id,
      score: 1 - index / 100,
      sourceRanks: {},
    }));

    const result = applyOriginalAndExpansionQuota(
      base,
      ["original-1", "original-2", "original-2", "original-3", "original-4", "original-5"],
      [
        ["e1-first", "original-1"],
        ["e2-first", "e1-first"],
        ["e3-first", "e2-first"],
      ],
      { topWindow: 10, originalQuota: 5, perExpansionQuota: 1 },
    );

    expect(result.slice(0, 10).map((item) => item.id)).toEqual([
      "original-1",
      "original-2",
      "original-3",
      "original-4",
      "original-5",
      "e1-first",
      "e2-first",
      "e3-first",
      "base-only",
      "tail",
    ]);
    expect(result.map((item) => item.id)).toHaveLength(new Set(ids).size);
    expect(new Set(result.map((item) => item.id))).toEqual(new Set(ids));
    expect(result.at(-1)?.id).toBe("later");
  });

  it("breaks quota ties from the supplied rankings and deduplicates repeatably", () => {
    const base = ["a", "b", "c", "d"].map((id) => ({ id, score: 1, sourceRanks: {} }));
    const select = () =>
      applyOriginalAndExpansionQuota(base, ["b", "b", "a"], [["d", "d", "c"]], {
        topWindow: 3,
        originalQuota: 1,
        perExpansionQuota: 1,
      });

    const expected = ["b", "d", "a", "c"];
    for (let index = 0; index < 10; index += 1) {
      expect(select().map((item) => item.id)).toEqual(expected);
    }
  });
});
