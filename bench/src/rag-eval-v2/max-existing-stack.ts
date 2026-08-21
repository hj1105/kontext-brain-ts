export interface MaxStackDocument {
  readonly id: string;
  readonly text: string;
}

export interface WeightedRanking {
  readonly name: "vector" | "graph" | "bm25" | "context-rerank";
  readonly ids: readonly string[];
  readonly weight: number;
}

export interface FusedCandidate {
  readonly id: string;
  readonly score: number;
  readonly sourceRanks: Readonly<Record<string, number>>;
}

interface IndexedDocument {
  readonly id: string;
  readonly tokens: readonly string[];
  readonly frequencies: ReadonlyMap<string, number>;
}

/** Corpus-level BM25 used as both a candidate generator and a fusion signal. */
export class CorpusBm25Ranker {
  private readonly documents: readonly IndexedDocument[];
  private readonly documentFrequency = new Map<string, number>();
  private readonly averageLength: number;

  constructor(documents: readonly MaxStackDocument[]) {
    this.documents = documents.map((document) => {
      const tokens = tokenize(document.text);
      const frequencies = frequenciesOf(tokens);
      for (const token of frequencies.keys()) {
        this.documentFrequency.set(token, (this.documentFrequency.get(token) ?? 0) + 1);
      }
      return { id: document.id, tokens, frequencies };
    });
    this.averageLength =
      this.documents.length === 0
        ? 0
        : this.documents.reduce((total, document) => total + document.tokens.length, 0) /
          this.documents.length;
  }

  rank(query: string, limit: number): string[] {
    if (limit <= 0 || this.documents.length === 0) return [];
    const queryTokens = [...new Set(tokenize(query))];
    return this.documents
      .map((document, index) => ({
        id: document.id,
        index,
        score: this.score(document, queryTokens),
      }))
      .sort(
        (left, right) =>
          right.score - left.score || left.index - right.index || left.id.localeCompare(right.id),
      )
      .slice(0, limit)
      .map((item) => item.id);
  }

  private score(document: IndexedDocument, queryTokens: readonly string[]): number {
    let score = 0;
    for (const token of queryTokens) {
      const frequency = document.frequencies.get(token) ?? 0;
      if (frequency === 0) continue;
      const containingDocuments = this.documentFrequency.get(token) ?? 0;
      const inverseDocumentFrequency = Math.log(
        1 + (this.documents.length - containingDocuments + 0.5) / (containingDocuments + 0.5),
      );
      const denominator =
        frequency +
        1.2 * (1 - 0.75 + 0.75 * (document.tokens.length / Math.max(1, this.averageLength)));
      score += inverseDocumentFrequency * ((frequency * 2.2) / denominator);
    }
    return score;
  }
}

/**
 * Weighted reciprocal-rank fusion over already bounded, independently ranked
 * candidate lists. Duplicate ids contribute once per source.
 */
export function fuseRankings(
  rankings: readonly WeightedRanking[],
  limit: number,
  reciprocalRankConstant = 60,
): FusedCandidate[] {
  if (limit <= 0) return [];
  const byId = new Map<string, { score: number; sourceRanks: Record<string, number> }>();
  for (const ranking of rankings) {
    if (ranking.weight <= 0) continue;
    const seen = new Set<string>();
    for (let index = 0; index < ranking.ids.length; index += 1) {
      const id = ranking.ids[index];
      if (!id || seen.has(id)) continue;
      seen.add(id);
      const rank = index + 1;
      const current = byId.get(id) ?? { score: 0, sourceRanks: {} };
      current.score += ranking.weight / (reciprocalRankConstant + rank);
      current.sourceRanks[ranking.name] = rank;
      byId.set(id, current);
    }
  }
  return Array.from(byId, ([id, value]) => ({ id, ...value }))
    .sort((left, right) => right.score - left.score || left.id.localeCompare(right.id))
    .slice(0, limit);
}

function tokenize(value: string): string[] {
  return value
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter((token) => token.length >= 2);
}

function frequenciesOf(tokens: readonly string[]): ReadonlyMap<string, number> {
  const output = new Map<string, number>();
  for (const token of tokens) output.set(token, (output.get(token) ?? 0) + 1);
  return output;
}
