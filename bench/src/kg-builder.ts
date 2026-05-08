/**
 * Heuristic knowledge-graph builder for GraphRAG-Bench (no LLM required).
 *
 * Approach:
 *   1. Extract entities per chunk via the same regex used by multi-hop
 *      (proper nouns, capitalized phrases, content tokens).
 *   2. Filter entities by document frequency (drop too-common, too-rare).
 *   3. Edges = co-occurrence within the same chunk (weighted by chunk
 *      frequency).
 *   4. Cache to disk; KGRetriever runs PPR on this.
 *
 * Why heuristic vs LLM extraction (HippoRAG-style):
 *   - System RAM constrained (2.7GB free), Llama-8B requires 4.4GB
 *   - Heuristic KG is fast (~5s for 1385 chunks vs ~70min for LLM)
 *   - Trades extraction quality for tractability; we still get a real KG
 *     (entity nodes + co-occurrence edges + PPR retrieval)
 */
import { writeFileSync, readFileSync, existsSync } from "node:fs";

export interface KGEntity {
  id: string;
  surface: string;
  chunkIds: string[];
  freq: number;
}

export interface KGEdge {
  src: string;
  predicate: string;
  dst: string;
  chunkId: string;
}

export interface KGStore {
  entities: Map<string, KGEntity>;
  edges: KGEdge[];
  chunkToEntities: Map<string, string[]>;
}

export interface KGSerialized {
  entities: KGEntity[];
  edges: KGEdge[];
  chunkToEntities: Array<[string, string[]]>;
}

export function emptyKG(): KGStore {
  return { entities: new Map(), edges: [], chunkToEntities: new Map() };
}

export function normalize(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9 \-]/g, "").trim().replace(/\s+/g, " ");
}

const STOP_CONTENT = new Set([
  "the","a","an","is","was","are","were","be","been","of","to","in","on","at",
  "for","by","with","as","that","this","it","its","and","or","but","not","from",
  "into","over","under","about","also","than","then","so","such","these","those",
  "there","they","them","their","does","do","did","has","have","had","can","could",
  "would","should","may","might","will","most","more","some","any","all","each",
  "which","who","whom","when","where","what","how","why","because","also",
  "include","includes","including","such","various","many","other","another",
  "one","two","three","four","five","first","second","third",
  "based","given","since","while","during","before","after","between","among",
]);

const STOP_CAP = new Set([
  "what","which","who","whom","when","where","how","why",
  "the","this","that","these","those","they","them","their","there",
  "are","is","was","were","be","been","being",
  "according","within","during","before","after","while",
]);

/** Extract proper-noun phrases + content tokens from a chunk. */
function extractFromChunk(text: string): string[] {
  const found = new Set<string>();
  // Proper noun phrases (capitalized, 1-4 words)
  for (const m of text.matchAll(/\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z0-9&][a-zA-Z0-9]*){0,3})\b/g)) {
    const phrase = m[1]!.trim();
    if (STOP_CAP.has(phrase.toLowerCase())) continue;
    if (phrase.length < 3 || phrase.length > 60) continue;
    found.add(phrase);
  }
  // Quoted strings
  for (const m of text.matchAll(/"([^"]{3,40})"/g)) found.add(m[1]!.trim());
  for (const m of text.matchAll(/'([^']{3,40})'/g)) found.add(m[1]!.trim());
  // Content tokens (lowercased domain words like "shehna", "biomass")
  for (const tok of text.toLowerCase().split(/[^a-z0-9\-]+/)) {
    if (tok.length < 5) continue; // 5+ chars only — fewer noise short words
    if (STOP_CONTENT.has(tok)) continue;
    if (STOP_CAP.has(tok)) continue;
    found.add(tok);
  }
  return Array.from(found);
}

export interface BuildKGOptions {
  /** Drop entities appearing in fewer than this many chunks */
  minDocFreq?: number;
  /** Drop entities appearing in more than this fraction of chunks */
  maxDocFreqRatio?: number;
  /** Cap entities per chunk (top by frequency-rare-first) */
  maxEntitiesPerChunk?: number;
}

export async function buildKG(
  _model: string, // unused — heuristic doesn't need an LLM
  chunks: Array<{ id: string; body: string; title?: string }>,
  cachePath?: string,
  onProgress?: (i: number, total: number) => void,
  opts: BuildKGOptions = {},
): Promise<KGStore> {
  if (cachePath && existsSync(cachePath)) {
    const data = JSON.parse(readFileSync(cachePath, "utf-8")) as KGSerialized;
    return {
      entities: new Map(data.entities.map((e) => [e.id, e])),
      edges: data.edges,
      chunkToEntities: new Map(data.chunkToEntities),
    };
  }

  const minDocFreq = opts.minDocFreq ?? 2;
  const maxRatio = opts.maxDocFreqRatio ?? 0.3;
  const maxPerChunk = opts.maxEntitiesPerChunk ?? 25;

  // Pass 1: per-chunk extraction, build raw entity → chunkIds map
  const rawEntChunks = new Map<string, { surface: string; chunkIds: Set<string> }>();
  const chunkRawEnts = new Map<string, string[]>();
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i]!;
    const surfaces = extractFromChunk(`${c.title ?? ""}\n${c.body}`);
    const idsForChunk: string[] = [];
    for (const s of surfaces) {
      const id = normalize(s);
      if (!id || id.length < 3) continue;
      const ent = rawEntChunks.get(id);
      if (ent) ent.chunkIds.add(c.id);
      else rawEntChunks.set(id, { surface: s, chunkIds: new Set([c.id]) });
      idsForChunk.push(id);
    }
    chunkRawEnts.set(c.id, Array.from(new Set(idsForChunk)));
    onProgress?.(i + 1, chunks.length);
  }

  // Pass 2: filter by doc frequency
  const totalChunks = chunks.length;
  const kg = emptyKG();
  for (const [id, ent] of rawEntChunks) {
    const df = ent.chunkIds.size;
    if (df < minDocFreq) continue;
    if (df / totalChunks > maxRatio) continue;
    kg.entities.set(id, {
      id,
      surface: ent.surface,
      chunkIds: Array.from(ent.chunkIds),
      freq: df,
    });
  }

  // Pass 3: chunkToEntities = filtered entities per chunk; cap by per-chunk count
  for (const [chunkId, ids] of chunkRawEnts) {
    const filtered = ids.filter((id) => kg.entities.has(id));
    // Sort by rarity (lower freq first) — favor distinctive entities
    filtered.sort((a, b) => kg.entities.get(a)!.freq - kg.entities.get(b)!.freq);
    kg.chunkToEntities.set(chunkId, filtered.slice(0, maxPerChunk));
  }

  // Pass 4: edges = co-occurrence (only between entities that survived filter)
  // Edge weight = co-occurrence count, but we store one row per chunk.
  for (const [chunkId, ids] of kg.chunkToEntities) {
    if (ids.length < 2) continue;
    // Build co-occurrence edges only among the most distinctive top-8 to
    // avoid quadratic blow-up
    const top = ids.slice(0, 8);
    for (let i = 0; i < top.length; i++) {
      for (let j = i + 1; j < top.length; j++) {
        kg.edges.push({ src: top[i]!, predicate: "co_occurs", dst: top[j]!, chunkId });
      }
    }
  }

  if (cachePath) {
    const ser: KGSerialized = {
      entities: Array.from(kg.entities.values()),
      edges: kg.edges,
      chunkToEntities: Array.from(kg.chunkToEntities.entries()),
    };
    writeFileSync(cachePath, JSON.stringify(ser));
  }
  return kg;
}
