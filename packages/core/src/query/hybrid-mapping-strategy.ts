import type { OntologyNode } from "../graph/ontology-node.js";
import { KeywordMappingStrategy, type NodeMappingStrategy } from "./node-mapping-strategy.js";
import type { VectorStore } from "./vector-store.js";

/**
 * Ensemble of keyword and vector scoring for L1 node routing.
 *
 * Keyword alone misses semantically-phrased queries; vector alone is noisy on
 * tiny (<20-word) node descriptions. Combining both with a weighted sum
 * recovers both signals.
 *
 * For each node:
 *   keyword_score = |query_words ∩ description_words| / |query_words|
 *   vector_score  = cosine(embed(query), embed(description))
 *   final_score   = keywordWeight * keyword_score + (1-keywordWeight) * vector_score
 */
export class HybridMappingStrategy implements NodeMappingStrategy {
  readonly strategyName = "hybrid";

  constructor(
    private readonly vectorStore: VectorStore,
    private readonly keywordWeight = 0.5,
    private readonly topK = 3,
  ) {}

  async findStartNodes(
    query: string,
    nodes: ReadonlyMap<string, OntologyNode>,
  ): Promise<string[]> {
    const q = query.toLowerCase();
    const qWords = new Set(q.split(/\s+/).filter((w) => w.length > 1));

    // Keyword scores
    const kwScores = new Map<string, number>();
    let kwMax = 0;
    for (const [id, node] of nodes) {
      const descWords = node.description.toLowerCase().split(/\s+/);
      let overlap = 0;
      for (const w of descWords) if (qWords.has(w)) overlap++;
      if (id.toLowerCase().split(/\s+/).some((w) => qWords.has(w))) overlap += 1;
      const score = overlap / Math.max(qWords.size, 1);
      kwScores.set(id, score);
      if (score > kwMax) kwMax = score;
    }

    // Vector scores via VectorStore.similaritySearch — assumes node
    // descriptions have been embedded with the node.id as key.
    const ranked = await this.vectorStore.similaritySearch(query, nodes.size);
    const vecScores = new Map<string, number>();
    ranked.forEach((nodeId, idx) => {
      // Linear decay: first result = 1.0, last ≈ 0
      if (nodes.has(nodeId)) vecScores.set(nodeId, 1 - idx / Math.max(ranked.length, 1));
    });

    // Combined
    const combined: Array<{ id: string; score: number }> = [];
    for (const [id] of nodes) {
      const kw = (kwScores.get(id) ?? 0) / (kwMax > 0 ? kwMax : 1);
      const vec = vecScores.get(id) ?? 0;
      const score = this.keywordWeight * kw + (1 - this.keywordWeight) * vec;
      combined.push({ id, score });
    }
    combined.sort((a, b) => b.score - a.score);
    const top = combined.filter((x) => x.score > 0).slice(0, this.topK);
    if (top.length > 0) return top.map((x) => x.id);

    return new KeywordMappingStrategy().findStartNodes(query, nodes);
  }
}
