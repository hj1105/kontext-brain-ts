/**
 * VectorStore port.
 *
 * `similaritySearchWithPrefix` is used by the VECTOR pipeline step:
 * it scopes the search to keys matching a prefix (e.g., "content:nodeId:")
 * and returns the complete suffix after that prefix. Keeping the full suffix
 * is important for URI document IDs such as `notion://page`.
 */
export interface VectorStore {
  embed(text: string): Promise<Float32Array>;
  upsert(key: string, embedding: Float32Array, metadata?: Record<string, string>): Promise<void>;
  similaritySearch(query: string, topK: number): Promise<string[]>;
  similaritySearchWithPrefix(
    query: string,
    prefix: string,
    topK: number,
    threshold?: number,
  ): Promise<string[]>;
  delete?(key: string): Promise<void>;
  deleteByPrefix?(prefix: string): Promise<void>;
}

/**
 * In-memory vector store with cosine similarity.
 * Requires an external embedder function.
 */
export class InMemoryVectorStore implements VectorStore {
  private readonly index = new Map<
    string,
    { embedding: Float32Array; metadata: Record<string, string> }
  >();

  constructor(private readonly embedder: (text: string) => Promise<Float32Array>) {}

  async embed(text: string): Promise<Float32Array> {
    return this.embedder(text);
  }

  async upsert(
    key: string,
    embedding: Float32Array,
    metadata: Record<string, string> = {},
  ): Promise<void> {
    this.index.set(key, { embedding, metadata });
  }

  async similaritySearch(query: string, topK: number): Promise<string[]> {
    return this.similaritySearchWithPrefix(query, "", topK);
  }

  async similaritySearchWithPrefix(
    query: string,
    prefix: string,
    topK: number,
    threshold = 0,
  ): Promise<string[]> {
    const queryVec = await this.embed(query);
    const scored: Array<{ key: string; score: number }> = [];

    for (const [key, { embedding }] of this.index) {
      if (prefix && !key.startsWith(prefix)) continue;
      const score = cosineSimilarity(queryVec, embedding);
      if (threshold > 0 && score < threshold) continue;
      scored.push({ key, score });
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map(({ key }) => (prefix ? key.slice(prefix.length) : key));
  }

  async delete(key: string): Promise<void> {
    this.index.delete(key);
  }

  async deleteByPrefix(prefix: string): Promise<void> {
    for (const key of this.index.keys()) {
      if (key.startsWith(prefix)) this.index.delete(key);
    }
  }
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
