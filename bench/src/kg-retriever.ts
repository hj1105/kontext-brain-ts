/**
 * KG-based retriever for GraphRAG-Bench. HippoRAG2-style:
 *
 *   1. Extract query entities (LLM or simple regex/BM25)
 *   2. Match query entities → KG entity nodes (substring + token overlap)
 *   3. Personalized PageRank (PPR) from matched seed nodes — propagate
 *      score along graph edges
 *   4. Score each chunk = sum of activation across the entities it
 *      mentions + boost from chunks that contain a seed entity directly
 *   5. Return top-K chunks
 */
import type { KGEdge, KGStore } from "./kg-builder.js";
import { normalize } from "./kg-builder.js";

export interface KGRetrieved {
  chunkId: string;
  score: number;
  matchedEntities: string[];
  reason: string;
}

interface BuiltAdjacency {
  /** entity id → list of neighbors (any direction) with weight */
  adj: Map<string, Array<{ to: string; weight: number }>>;
}

/** Build adjacency for PPR. Edge is bidirectional with weight 1. */
function buildAdjacency(kg: KGStore): BuiltAdjacency {
  const adj = new Map<string, Array<{ to: string; weight: number }>>();
  const ensure = (id: string) => {
    const neighbors = adj.get(id);
    if (neighbors) return neighbors;
    const created: Array<{ to: string; weight: number }> = [];
    adj.set(id, created);
    return created;
  };
  for (const ent of kg.entities.values()) ensure(ent.id);
  for (const e of kg.edges) {
    ensure(e.src).push({ to: e.dst, weight: 1 });
    ensure(e.dst).push({ to: e.src, weight: 1 });
  }
  return { adj };
}

/** Personalized PageRank — power iteration, restart prob alpha to seeds. */
function personalizedPageRank(
  kg: KGStore,
  adjacency: BuiltAdjacency,
  seeds: Map<string, number>, // node id → initial weight (sum to 1 ideally)
  alpha = 0.15,
  iters = 20,
): Map<string, number> {
  // Normalize seeds
  let totalSeed = 0;
  for (const v of seeds.values()) totalSeed += v;
  if (totalSeed === 0) return new Map();
  const seedNorm = new Map<string, number>();
  for (const [k, v] of seeds) seedNorm.set(k, v / totalSeed);

  // Initialize all nodes
  const nodes = Array.from(kg.entities.keys());
  let r = new Map<string, number>(nodes.map((n) => [n, seedNorm.get(n) ?? 0]));

  for (let iter = 0; iter < iters; iter++) {
    const r2 = new Map<string, number>(nodes.map((n) => [n, 0]));
    for (const n of nodes) {
      const score = r.get(n) ?? 0;
      if (score === 0) continue;
      const neighbors = adjacency.adj.get(n) ?? [];
      if (neighbors.length === 0) {
        // sink: redistribute to seeds
        for (const [s, w] of seedNorm) r2.set(s, (r2.get(s) ?? 0) + (1 - alpha) * score * w);
        continue;
      }
      const sumW = neighbors.reduce((a, b) => a + b.weight, 0);
      for (const ne of neighbors) {
        r2.set(ne.to, (r2.get(ne.to) ?? 0) + (1 - alpha) * score * (ne.weight / sumW));
      }
    }
    // Add restart contribution
    for (const [s, w] of seedNorm) r2.set(s, (r2.get(s) ?? 0) + alpha * w);
    r = r2;
  }
  return r;
}

/** Match query content tokens to KG entity nodes by substring + overlap. */
export function findSeedEntities(kg: KGStore, query: string): Map<string, number> {
  const seeds = new Map<string, number>();
  const qNorm = normalize(query);
  const qTokens = new Set(qNorm.split(/\s+/).filter((t) => t.length >= 3));

  for (const ent of kg.entities.values()) {
    // Exact substring of entity surface in query
    if (qNorm.includes(ent.id)) {
      seeds.set(ent.id, (seeds.get(ent.id) ?? 0) + 2.0);
      continue;
    }
    // Token overlap (e.g. "cancer" matches "skin cancer")
    const eTokens = ent.id.split(/\s+/).filter((t) => t.length >= 3);
    if (eTokens.length === 0) continue;
    const overlap = eTokens.filter((t) => qTokens.has(t)).length;
    if (overlap > 0) {
      const ratio = overlap / eTokens.length;
      if (ratio >= 0.5) seeds.set(ent.id, (seeds.get(ent.id) ?? 0) + ratio);
    }
  }
  return seeds;
}

export interface KGRetrieverConfig {
  /** PPR damping factor (alpha = restart prob to seeds). 0.15 typical. */
  alpha?: number;
  /** PPR iterations. */
  iters?: number;
  /** Top-K chunks to return. */
  topK?: number;
  /** Minimum entity-PPR score for a chunk's entities to count. */
  minActivation?: number;
}

export class KGRetriever {
  private adjacency: BuiltAdjacency;

  constructor(private readonly kg: KGStore) {
    this.adjacency = buildAdjacency(kg);
  }

  retrieve(query: string, cfg: KGRetrieverConfig = {}): KGRetrieved[] {
    const { alpha = 0.15, iters = 20, topK = 5, minActivation = 0 } = cfg;
    const seeds = findSeedEntities(this.kg, query);
    if (seeds.size === 0) return [];

    const ppr = personalizedPageRank(this.kg, this.adjacency, seeds, alpha, iters);

    // Score each chunk: sum of PPR activation across its entities
    const chunkScore = new Map<string, { score: number; matched: Set<string> }>();
    for (const [chunkId, entIds] of this.kg.chunkToEntities) {
      let s = 0;
      const matched = new Set<string>();
      for (const eid of entIds) {
        const a = ppr.get(eid) ?? 0;
        if (a < minActivation) continue;
        s += a;
        if (a > 0.005) matched.add(eid);
      }
      if (s > 0) chunkScore.set(chunkId, { score: s, matched });
    }

    const ranked = Array.from(chunkScore.entries())
      .sort((a, b) => b[1].score - a[1].score || a[0].localeCompare(b[0]))
      .slice(0, topK)
      .map(([chunkId, v]) => ({
        chunkId,
        score: v.score,
        matchedEntities: Array.from(v.matched),
        reason: `PPR seeds=${seeds.size}, matched=${v.matched.size}`,
      }));

    return ranked;
  }
}
