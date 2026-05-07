/**
 * Multi-hop retriever v2 — addresses HotpotQA bridge/comparison weakness.
 *
 * Observation from probe-retrieval.ts: nomic-embed-text produces near-
 * identical embeddings for short entity queries ("Tom Petty", "Traveling
 * Wilburys"). Vector similarity is useless for entity retrieval.
 *
 * Strategy:
 *   1. Extract named entities from the question.
 *   2. For each entity: BM25 (exact-token) search over corpus titles+bodies
 *      to find the doc that LITERALLY mentions that entity name. This is
 *      robust for proper nouns.
 *   3. Also run full-question vector search for the composition context
 *      (long queries work fine with nomic-embed).
 *   4. Union and rank by:
 *        - multi-hit bonus (entities + vector all surface the same doc)
 *        - entity-match priority: guarantee one doc per entity
 */

import type { VectorStore } from "@kontext-brain/core";

export interface MultiHopDoc {
  docId: string;
  title: string;
  body: string;
  score: number;
  foundBy: string[]; // which subquery surfaced it
}

export interface MultiHopDocIndex {
  id: string;
  title: string;
  body: string;
}

// Stop-cap words that frequently appear as the first word of a question
// or sentence but carry no semantic anchor for retrieval. Failure analysis
// of GraphRAG-Bench Novel showed "According", "Within", "During" leaking
// through.
const STOP_CAP = new Set([
  "what","which","who","whom","when","where","how","why",
  "the","this","that","these","those","they","them","their","there",
  "are","is","was","were","be","been","being",
  "does","do","did","has","have","had","can","could","would","should",
  "may","might","will","shall","must",
  "in","on","at","of","for","and","or","but","not","from",
  "according","within","during","before","after","while",
  "based","given","such","each","every",
  "list","name","describe","explain","mention","identify","tell",
]);

const STOP_CONTENT = new Set([
  "the","a","an","and","or","but","of","to","in","on","at","for","by",
  "with","as","that","this","it","its","is","are","was","were","be","been",
  "have","has","had","do","does","did","will","would","could","should",
  "may","might","can","not","also","than","then","so","such","these","those",
  "there","they","them","their","when","where","what","which","who","whom",
  "how","why","into","onto","over","under","about","through","between",
  "among","because","although","though","while","during","before","after",
  "according","within","based","given","including","such","each","every",
  "narrative","novel","story","text","question","answer","reference",
]);

export interface ExtractedEntities {
  /** Capitalized proper nouns / quoted strings — high-confidence anchors. */
  proper: string[];
  /** Content tokens (4+ char, non-stopword) — IDF-weighted lookup later. */
  content: string[];
}

export function extractEntitiesV2(question: string): ExtractedEntities {
  const proper = new Set<string>();
  for (const m of question.matchAll(/"([^"]{2,60})"/g)) proper.add(m[1]!.trim());
  for (const m of question.matchAll(/'([^']{2,60})'/g)) proper.add(m[1]!.trim());
  for (const m of question.matchAll(/\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z0-9&][a-zA-Z0-9]*){0,3})\b/g)) {
    const phrase = m[1]!.trim();
    if (STOP_CAP.has(phrase.toLowerCase())) continue;
    if (phrase.length < 3) continue;
    proper.add(phrase);
  }
  // Content tokens — capture lowercased domain-specific words like "shehna",
  // "soaking", "ingots", "syrinx", "carillon" that the cap regex misses.
  const content = new Set<string>();
  for (const tok of question.toLowerCase().split(/[^a-z0-9]+/)) {
    if (tok.length < 4) continue;
    if (STOP_CONTENT.has(tok)) continue;
    if (STOP_CAP.has(tok)) continue;
    content.add(tok);
  }
  return { proper: Array.from(proper), content: Array.from(content) };
}

/** Backwards-compatible wrapper — returns proper nouns only. */
export function extractEntities(question: string): string[] {
  return extractEntitiesV2(question).proper;
}

/**
 * BM25 + title-exact-match scoring. Entity-name queries should strongly
 * prefer a doc whose TITLE is exactly the entity (or contains it verbatim),
 * over a doc that merely has the entity mentioned in the body.
 */
function bm25Score(query: string, doc: MultiHopDocIndex, dfMap: Map<string, number>, totalDocs: number): number {
  const qNorm = query.toLowerCase().trim();
  const titleNorm = doc.title.toLowerCase();

  // Title match bonuses (very strong — proper-noun queries should find
  // the canonical doc, not a related one with more token overlap)
  let titleBonus = 0;
  if (titleNorm === qNorm) {
    titleBonus = 100;
  } else if (titleNorm.startsWith(qNorm + " ") || titleNorm === qNorm + ".") {
    titleBonus = 30; // "Lulu" → "Lulu (opera)" etc.
  } else if (titleNorm.includes(qNorm)) {
    titleBonus = 10;
  }

  const qTokens = tokenize(query);
  if (qTokens.length === 0) return titleBonus;
  const titleTokens = new Set(tokenize(doc.title));
  const bodyTokens = new Set(tokenize(doc.body));
  let score = titleBonus;
  for (const t of qTokens) {
    const df = dfMap.get(t) ?? 1;
    const idf = Math.log((totalDocs - df + 0.5) / (df + 0.5) + 1);
    if (titleTokens.has(t)) score += idf * 3; // title hit counts 3x
    else if (bodyTokens.has(t)) score += idf;
  }
  return score;
}

function tokenize(s: string): string[] {
  return s
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 1);
}

export class MultiHopRetriever {
  private dfMap: Map<string, number> = new Map();
  private totalDocs = 0;

  constructor(private readonly docs: MultiHopDocIndex[], private readonly vectorStore: VectorStore) {
    // Build doc frequency map
    this.totalDocs = docs.length;
    for (const d of docs) {
      const tokens = new Set(tokenize(`${d.title} ${d.body}`));
      for (const t of tokens) {
        this.dfMap.set(t, (this.dfMap.get(t) ?? 0) + 1);
      }
    }
  }

  /** BM25 top-k for a single query (typically an entity name). */
  bm25Top(query: string, k: number): MultiHopDocIndex[] {
    const scored = this.docs.map((d) => ({
      doc: d,
      score: bm25Score(query, d, this.dfMap, this.totalDocs),
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored
      .filter((s) => s.score > 0)
      .slice(0, k)
      .map((s) => s.doc);
  }

  /** Hybrid multi-hop retrieval with iterative entity expansion (V3).
   *
   * V3 changes vs V2:
   *   - Adds content-token (lowercase, 4+ char, non-stopword) BM25 lookup
   *     to handle queries like "shehna / soaking pit / ingots / Cilicia"
   *     where the cap-regex missed the gold entity.
   *   - Anti-bias: optionally accepts a `noiseDocPenalty` set of doc IDs
   *     that are over-retrieved across the workload — these get a score
   *     penalty to prevent the corpus's "default-noise" doc from
   *     crowding out actual gold (failure mode found in HotpotQA's
   *     viva-la-vida and GB-Novel's Elsie-Inglis-biography 51410).
   */
  async retrieve(
    question: string,
    topK: number = 5,
    opts: { noiseDocs?: Set<string>; useContentBM25?: boolean } = {},
  ): Promise<MultiHopDoc[]> {
    const ext = extractEntitiesV2(question);
    const entities = ext.proper;
    const useContent = opts.useContentBM25 ?? true;

    const foundBy = new Map<string, { doc: MultiHopDocIndex; score: number; foundBy: Set<string> }>();

    // 1) Full-question vector search (works well for long questions)
    const vecHits = await this.vectorStore.similaritySearchWithPrefix(question, "doc:", 5);
    for (let i = 0; i < vecHits.length; i++) {
      const docId = vecHits[i]!;
      const doc = this.docs.find((d) => d.id === docId);
      if (!doc) continue;
      const rankScore = 1 - i / 5;
      const existing = foundBy.get(docId) ?? { doc, score: 0, foundBy: new Set<string>() };
      existing.score += rankScore * 0.6;
      existing.foundBy.add("full-question-vector");
      foundBy.set(docId, existing);
    }

    // 2) Per-entity BM25 top-2
    for (const ent of entities) {
      const bm25Hits = this.bm25Top(ent, 2);
      for (let i = 0; i < bm25Hits.length; i++) {
        const doc = bm25Hits[i]!;
        const rankScore = 1 - i / 2;
        const existing = foundBy.get(doc.id) ?? { doc, score: 0, foundBy: new Set<string>() };
        existing.score += rankScore * 1.0;
        existing.foundBy.add(`entity:${ent}`);
        foundBy.set(doc.id, existing);
      }
    }

    // 3) Full-question BM25 top-3
    const qBm25 = this.bm25Top(question, 3);
    for (let i = 0; i < qBm25.length; i++) {
      const doc = qBm25[i]!;
      const rankScore = 1 - i / 3;
      const existing = foundBy.get(doc.id) ?? { doc, score: 0, foundBy: new Set<string>() };
      existing.score += rankScore * 0.4;
      existing.foundBy.add("full-question-bm25");
      foundBy.set(doc.id, existing);
    }

    // 3b) Content-token BM25 — handles lowercase domain words (shehna,
    // soaking, ingots, syrinx) the cap regex misses. Each high-IDF token
    // contributes its top-1 doc.
    if (useContent) {
      // Rank content tokens by their IDF (high IDF = rare = informative)
      const ranked = ext.content
        .map((t) => ({ t, idf: Math.log((this.totalDocs - (this.dfMap.get(t) ?? 0) + 0.5) / ((this.dfMap.get(t) ?? 0) + 0.5) + 1) }))
        .filter((x) => x.idf > 0)
        .sort((a, b) => b.idf - a.idf)
        .slice(0, 10);
      // Top-2 per content token (was top-1) — gives chunk-level granularity
      // for cases where the right novel is found but the specific scene chunk
      // is the #2 not #1 BM25 hit.
      for (const { t, idf } of ranked) {
        const hits = this.bm25Top(t, 2);
        for (let i = 0; i < hits.length; i++) {
          const doc = hits[i]!;
          const rankScore = 1 - i / 2;
          const existing = foundBy.get(doc.id) ?? { doc, score: 0, foundBy: new Set<string>() };
          existing.score += Math.min(idf / 5, 1) * 0.7 * rankScore;
          existing.foundBy.add(`content:${t}`);
          foundBy.set(doc.id, existing);
        }
      }
      // Multi-token-hit bonus: chunks that match multiple high-IDF tokens
      // are likely the correct scene, not just topically related.
      for (const v of foundBy.values()) {
        const contentHits = Array.from(v.foundBy).filter((s) => s.startsWith("content:")).length;
        if (contentHits >= 2) v.score += 0.3 * (contentHits - 1);
      }
    }

    // 4) Iterative expansion (second hop): from top-3 current results,
    //    extract named entities in their BODIES (not just titles) and
    //    retrieve the top doc for each. Handles bridge questions where
    //    hop-2 entity isn't in the original question (e.g. "Massimo
    //    Giordano was born in Pompei" → need Pompei page).
    const topSoFar = Array.from(foundBy.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);
    const seenEntities = new Set(entities.map((e) => e.toLowerCase()));
    const hopEntities: string[] = [];
    for (const { doc } of topSoFar) {
      // Extract capitalized phrases from full body (was 500-char limit, too short)
      const snippet = doc.body.slice(0, 2500);
      const matches = snippet.matchAll(/\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+){0,3})\b/g);
      for (const m of matches) {
        const e = m[1]!.trim();
        const eLc = e.toLowerCase();
        if (seenEntities.has(eLc)) continue;
        if (e.length < 4 || e.length > 50) continue;
        // Skip common English words that happen to be capitalized
        if (/^(The|This|That|These|Those|They|Their|There|When|Where|What|Which|Who|Whom|How|Why|Also|But|And|From|Into|Over|Under|About|After|Before|During|While|Between|Among|Through|Because|Although|Though|English|American|British|French|German|Italian|Spanish|Japanese|Chinese|Indian|Korean|Russian|European|African|Asian)$/i.test(e)) continue;
        seenEntities.add(eLc);
        hopEntities.push(e);
      }
    }
    // Retrieve top-1 per hop entity. Boost more when title matches exactly
    // (high-confidence hop), smaller boost when it's just body match.
    for (const he of hopEntities.slice(0, 15)) {
      const hits = this.bm25Top(he, 1);
      for (const doc of hits) {
        const isTitleMatch = doc.title.toLowerCase().includes(he.toLowerCase());
        const existing = foundBy.get(doc.id) ?? { doc, score: 0, foundBy: new Set<string>() };
        existing.score += isTitleMatch ? 0.9 : 0.4;
        existing.foundBy.add(`hop2:${he}`);
        foundBy.set(doc.id, existing);
      }
    }

    // Apply noise-doc penalty (anti-bias against globally over-retrieved
    // docs that cosine similarity collapses onto for short queries)
    if (opts.noiseDocs && opts.noiseDocs.size > 0) {
      for (const [id, v] of foundBy) {
        if (opts.noiseDocs.has(id)) v.score *= 0.3;
      }
    }

    const ranked: MultiHopDoc[] = Array.from(foundBy.values())
      .map((v) => ({
        docId: v.doc.id,
        title: v.doc.title,
        body: v.doc.body,
        score: v.score,
        foundBy: Array.from(v.foundBy),
      }))
      .sort((a, b) => b.score - a.score);

    // 4) Coverage guarantee — ensure one doc per entity is present in top-K
    const selected = new Map<string, MultiHopDoc>();
    for (const r of ranked) {
      if (selected.size >= topK) break;
      selected.set(r.docId, r);
    }
    if (entities.length >= 2 && topK >= 2) {
      for (const ent of entities) {
        const covers = (d: MultiHopDoc) => d.foundBy.includes(`entity:${ent}`);
        const hasCoverage = Array.from(selected.values()).some(covers);
        if (hasCoverage) continue;
        // Find this entity's top BM25 doc; insert it, evict the lowest-score non-entity doc
        const entTop = this.bm25Top(ent, 1)[0];
        if (!entTop || selected.has(entTop.id)) continue;
        const byScoreAsc = Array.from(selected.values()).sort((a, b) => a.score - b.score);
        for (const cand of byScoreAsc) {
          if (!entities.some((e) => cand.foundBy.includes(`entity:${e}`))) {
            selected.delete(cand.docId);
            selected.set(entTop.id, {
              docId: entTop.id,
              title: entTop.title,
              body: entTop.body,
              score: 0.5,
              foundBy: [`entity:${ent}-fallback`],
            });
            break;
          }
        }
      }
    }

    return Array.from(selected.values()).sort((a, b) => b.score - a.score);
  }
}
