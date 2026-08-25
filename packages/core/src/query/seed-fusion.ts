import type { SearchSeed } from "./bidirectional-retriever.js";
import type { RankedRetrievalObservation } from "./traversal-scoring.js";

/** Merges provider observations without adding incomparable native scores. */
export function fuseSearchSeeds(seeds: readonly SearchSeed[]): SearchSeed[] {
  const fused = new Map<string, SearchSeed>();
  for (const seed of seeds) {
    const key = `${seed.node.kind}:${seed.node.id}`;
    const previous = fused.get(key);
    if (!previous) {
      fused.set(key, seed);
      continue;
    }
    const previousQuery = previous.observations?.query;
    const nextQuery = seed.observations?.query;
    fused.set(key, {
      node: seed.node,
      ...maxLegacyScore(previous, seed),
      observations: {
        fallback: Boolean(previous.observations?.fallback && seed.observations?.fallback),
        providers: Array.from(
          new Set([
            ...(previous.observations?.providers ?? []),
            ...(seed.observations?.providers ?? []),
          ]),
        ).sort(),
        query: {
          exactMatch: Boolean(previousQuery?.exactMatch || nextQuery?.exactMatch),
          aliasMatch: Boolean(previousQuery?.aliasMatch || nextQuery?.aliasMatch),
          ...optionalRank("lexical", betterRank(previousQuery?.lexical, nextQuery?.lexical)),
          ...optionalRank("vector", betterRank(previousQuery?.vector, nextQuery?.vector)),
          ...maxReranker(previousQuery?.rerankerScore, nextQuery?.rerankerScore),
        },
      },
    });
  }
  return Array.from(fused.values());
}

function betterRank(
  left: RankedRetrievalObservation | undefined,
  right: RankedRetrievalObservation | undefined,
): RankedRetrievalObservation | undefined {
  if (!left) return right;
  if (!right) return left;
  if (left.rank !== right.rank) return left.rank < right.rank ? left : right;
  const leftScore = left.normalizedScore ?? Number.NEGATIVE_INFINITY;
  const rightScore = right.normalizedScore ?? Number.NEGATIVE_INFINITY;
  return leftScore >= rightScore ? left : right;
}

function optionalRank(
  key: "lexical" | "vector",
  value: RankedRetrievalObservation | undefined,
): Partial<Record<"lexical" | "vector", RankedRetrievalObservation>> {
  return value === undefined ? {} : { [key]: value };
}

function maxReranker(
  left: number | undefined,
  right: number | undefined,
): { readonly rerankerScore?: number } {
  return left === undefined && right === undefined
    ? {}
    : { rerankerScore: Math.max(left ?? 0, right ?? 0) };
}

function maxLegacyScore(left: SearchSeed, right: SearchSeed): { readonly score?: number } {
  return left.score === undefined && right.score === undefined
    ? {}
    : { score: Math.max(left.score ?? 0, right.score ?? 0) };
}
