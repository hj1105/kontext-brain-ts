import type { DataSource, DocumentContent, MetaDocument } from "../graph/layered-models.js";
import type { LLMAdapter } from "./llm-adapter.js";
import type { PromptTemplates } from "./prompt-templates.js";
import { DefaultPromptTemplates } from "./prompt-templates.js";

/**
 * L2 -> L3 selector port.
 * Given a list of candidate titles, picks which documents to actually fetch.
 */
export interface MetaDocumentSelector {
  select(query: string, candidates: readonly MetaDocument[], maxSelect: number): Promise<MetaDocument[]>;
}

/** Score-based selector — no LLM needed. */
export class ScoreBasedSelector implements MetaDocumentSelector {
  async select(
    query: string,
    candidates: readonly MetaDocument[],
    maxSelect: number,
  ): Promise<MetaDocument[]> {
    const q = query.toLowerCase();
    const queryWords = new Set(q.split(/\s+/).filter((w) => w.length > 0));

    return candidates
      .map((doc) => {
        const words = doc.title.toLowerCase().split(/\s+/);
        let overlap = 0;
        for (const w of words) if (queryWords.has(w)) overlap++;
        const termScore = overlap / (queryWords.size + 1);
        return { ...doc, score: doc.score * 0.6 + termScore * 0.4 };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, maxSelect);
  }
}

/**
 * Compressed-prompt LLM selector. Sends `i:title` lines, expects comma-separated indices back.
 * Saves ~80% tokens vs full descriptions.
 */
export class LLMMetaDocumentSelector implements MetaDocumentSelector {
  constructor(
    private readonly adapter: LLMAdapter,
    private readonly templates: PromptTemplates = DefaultPromptTemplates,
  ) {}

  async select(
    query: string,
    candidates: readonly MetaDocument[],
    maxSelect: number,
  ): Promise<MetaDocument[]> {
    if (candidates.length === 0) return [];

    const titleList = candidates
      .slice(0, 60)
      .map((doc, i) => `${i}:${doc.title}`)
      .join("\n");

    const response = await this.adapter.complete(
      this.templates.metaDocumentSelector,
      titleList,
      query,
    );

    const selectedIds = parseCompactIndices(response, candidates.length);
    const filtered = candidates.filter((_, i) => selectedIds.has(i)).slice(0, maxSelect);

    if (filtered.length === 0) {
      return new ScoreBasedSelector().select(query, candidates, maxSelect);
    }
    return filtered;
  }
}

function parseCompactIndices(response: string, size: number): Set<number> {
  const clean = response.trim().replace(/^```json/, "").replace(/```$/, "").trim();
  try {
    const parsed = JSON.parse(clean);
    if (Array.isArray(parsed)) {
      const set = new Set<number>();
      for (const v of parsed) {
        const n = Number(v);
        if (Number.isInteger(n) && n >= 0 && n < size) set.add(n);
      }
      if (set.size > 0) return set;
    }
  } catch {
    // fall through
  }
  const set = new Set<number>();
  for (const token of clean.split(",")) {
    const n = Number.parseInt(token.trim(), 10);
    if (Number.isInteger(n) && n >= 0 && n < size) set.add(n);
  }
  return set;
}

// ── BM25 body extractor ──────────────────────────────────────

/**
 * BM25-based body compression. Instead of sending the full body to the LLM,
 * extract the top-N relevant sentences.
 */
export const BM25BodyExtractor = {
  extract(body: string, query: string, maxSentences = 3): string {
    const sentences = body
      .split(/[.。!?\n]/)
      .map((s) => s.trim())
      .filter((s) => s.length > 5);

    if (sentences.length <= maxSentences) return body;

    const qTokens = tokenize(query);
    const docTokens = sentences.map(tokenize);
    const scores = bm25Scores(qTokens, docTokens);

    const topIdx = scores
      .map((score, index) => ({ score, index }))
      .sort((a, b) => b.score - a.score)
      .slice(0, maxSentences)
      .map((x) => x.index)
      .sort((a, b) => a - b);

    return topIdx.map((i) => sentences[i]).join(". ");
  },
};

function tokenize(text: string): string[] {
  const tokens: string[] = [];
  // Latin words
  tokens.push(
    ...text
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((t) => t.length > 1),
  );
  // Korean 2-grams
  let korean = "";
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code >= 0xac00 && code <= 0xd7a3) korean += text[i];
  }
  for (let i = 0; i < korean.length - 1; i++) {
    tokens.push(korean.substring(i, i + 2));
  }
  return tokens;
}

function bm25Scores(
  query: readonly string[],
  docs: readonly (readonly string[])[],
  k1 = 1.5,
  b = 0.75,
): number[] {
  const avgLen = Math.max(1, docs.reduce((sum, d) => sum + d.length, 0) / docs.length);
  const N = docs.length;
  const df = new Map<string, number>();
  for (const tokens of docs) {
    for (const t of new Set(tokens)) df.set(t, (df.get(t) ?? 0) + 1);
  }
  return docs.map((tokens) => {
    const tf = new Map<string, number>();
    for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
    const docLen = tokens.length;
    let score = 0;
    for (const term of query) {
      const tfVal = tf.get(term) ?? 0;
      const dfVal = df.get(term) ?? 0;
      const idf = Math.log((N - dfVal + 0.5) / (dfVal + 0.5) + 1);
      const tfNorm = (tfVal * (k1 + 1)) / (tfVal + k1 * (1 - b + b * (docLen / avgLen)));
      score += idf * tfNorm;
    }
    return score;
  });
}

// ── ContentFetcher ────────────────────────────────────────────

export interface ContentFetcher {
  readonly source: DataSource;
  fetch(metaDoc: MetaDocument): Promise<DocumentContent>;
}

export class ContentFetcherRegistry {
  private readonly fetchers = new Map<DataSource, ContentFetcher>();

  register(fetcher: ContentFetcher): void {
    this.fetchers.set(fetcher.source, fetcher);
  }

  async fetch(metaDoc: MetaDocument): Promise<DocumentContent> {
    const fetcher = this.fetchers.get(metaDoc.source);
    if (fetcher) return fetcher.fetch(metaDoc);
    return {
      metaDocumentId: metaDoc.id,
      title: metaDoc.title,
      body: `[fetcher not registered for ${metaDoc.source}]`,
      source: metaDoc.source,
      sectionContent: null,
      fetchedAt: new Date(),
    };
  }

  supports(source: DataSource): boolean {
    return this.fetchers.has(source);
  }
}
