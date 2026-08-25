import { describe, expect, it } from "vitest";
import { type SearchSeed, fuseSearchSeeds } from "../src/index.js";

describe("fuseSearchSeeds", () => {
  it("keeps lexical and vector ranks as separate observations", () => {
    const seeds: SearchSeed[] = [
      {
        node: { kind: "chunk", id: "c1" },
        observations: {
          providers: ["lexical"],
          query: { lexical: { rank: 2, candidateCount: 10 } },
        },
      },
      {
        node: { kind: "chunk", id: "c1" },
        observations: {
          providers: ["vector"],
          query: { vector: { rank: 1, candidateCount: 20 } },
        },
      },
    ];

    expect(fuseSearchSeeds(seeds)).toEqual([
      {
        node: { kind: "chunk", id: "c1" },
        observations: {
          fallback: false,
          providers: ["lexical", "vector"],
          query: {
            exactMatch: false,
            aliasMatch: false,
            lexical: { rank: 2, candidateCount: 10 },
            vector: { rank: 1, candidateCount: 20 },
          },
        },
      },
    ]);
  });

  it("keeps the best rank per provider and uses normalized score only to break rank ties", () => {
    const node = { kind: "chunk" as const, id: "c1" };
    const seeds: SearchSeed[] = [
      {
        node,
        observations: {
          providers: ["lexical"],
          query: { lexical: { rank: 3, candidateCount: 20, normalizedScore: 0.95 } },
        },
      },
      {
        node,
        observations: {
          providers: ["lexical"],
          query: { lexical: { rank: 2, candidateCount: 20, normalizedScore: 0.4 } },
        },
      },
      {
        node,
        observations: {
          providers: ["lexical"],
          query: { lexical: { rank: 2, candidateCount: 20, normalizedScore: 0.8 } },
        },
      },
    ];

    expect(fuseSearchSeeds(seeds)[0]?.observations?.query?.lexical).toEqual({
      rank: 2,
      candidateCount: 20,
      normalizedScore: 0.8,
    });
  });

  it("combines match signals while keeping fallback exclusive to fallback-only seeds", () => {
    const node = { kind: "resource" as const, id: "orders" };
    const fused = fuseSearchSeeds([
      {
        node,
        score: 0.3,
        observations: {
          fallback: true,
          providers: ["fallback", "lexical"],
          query: { aliasMatch: true, rerankerScore: 0.4 },
        },
      },
      {
        node,
        score: 0.9,
        observations: {
          providers: ["vector", "lexical"],
          query: { exactMatch: true, rerankerScore: 0.8 },
        },
      },
    ]);

    expect(fused).toEqual([
      {
        node,
        score: 0.9,
        observations: {
          fallback: false,
          providers: ["fallback", "lexical", "vector"],
          query: {
            exactMatch: true,
            aliasMatch: true,
            rerankerScore: 0.8,
          },
        },
      },
    ]);
  });
});
