import type { MetaDocument } from "../graph/layered-models.js";
import type { MetaDocumentSelector } from "./content-fetcher.js";

/**
 * Maximal Marginal Relevance selector.
 *
 *   MMR(d) = lambda * relevance(d, query) - (1 - lambda) * max(similarity(d, d') for d' in selected)
 *
 * Picks documents iteratively: each pick maximises relevance to the query
 * while penalising similarity to already-selected docs. Reduces redundancy
 * when retrieved candidates are near-duplicates.
 *
 * Useful when:
 *  - corpus has multiple docs covering the same concept
 *  - L1/L2 retrieval pulled in 5 docs with overlapping content
 *  - LLM context window is tight, so each picked doc should add new info
 *
 * Document similarity here uses bag-of-words Jaccard on titles. For richer
 * similarity, inject an embedding-based variant.
 */
export class MmrSelector implements MetaDocumentSelector {
  constructor(
    /** 0.0 = pure diversity, 1.0 = pure relevance. 0.7 is a good default. */
    private readonly lambda = 0.7,
  ) {}

  async select(
    query: string,
    candidates: readonly MetaDocument[],
    maxSelect: number,
  ): Promise<MetaDocument[]> {
    if (candidates.length === 0) return [];
    if (candidates.length <= maxSelect) return [...candidates];

    const q = tokens(query);
    const cands = candidates.map((doc) => ({
      doc,
      titleTokens: tokens(doc.title),
      relevance: jaccard(q, tokens(doc.title)) + doc.score * 0.3,
    }));

    const selected: typeof cands = [];
    const remaining = [...cands];

    // Pick the most relevant first
    remaining.sort((a, b) => b.relevance - a.relevance);
    const first = remaining.shift();
    if (first) selected.push(first);

    while (selected.length < maxSelect && remaining.length > 0) {
      let best: { idx: number; mmr: number } | null = null;
      for (let i = 0; i < remaining.length; i++) {
        const cand = remaining[i]!;
        const maxSim = Math.max(
          ...selected.map((s) => jaccard(cand.titleTokens, s.titleTokens)),
        );
        const mmr = this.lambda * cand.relevance - (1 - this.lambda) * maxSim;
        if (!best || mmr > best.mmr) best = { idx: i, mmr };
      }
      if (!best) break;
      selected.push(remaining.splice(best.idx, 1)[0]!);
    }

    return selected.map((s) => ({ ...s.doc, score: s.relevance }));
  }
}

function tokens(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .split(/[\s\p{P}]+/u)
      .filter((w) => w.length > 1),
  );
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 0;
  let inter = 0;
  for (const t of a) if (b.has(t)) inter++;
  const union = a.size + b.size - inter;
  return union === 0 ? 0 : inter / union;
}
