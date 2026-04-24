import type { OntologyNode } from "../graph/ontology-node.js";
import { KeywordMappingStrategy, type NodeMappingStrategy } from "./node-mapping-strategy.js";

/**
 * BM25-based L1 node routing.
 *
 * Improves over `KeywordMappingStrategy` by weighting rare query terms more
 * than common ones via inverse document frequency. On the bench corpus,
 * "JWT" should pull harder than "the" or "use".
 *
 * Treats each node's (id + description + keywords) as a "document" and runs
 * BM25 over the node corpus.
 */
export class Bm25NodeMappingStrategy implements NodeMappingStrategy {
  readonly strategyName = "bm25";

  constructor(
    private readonly k1 = 1.5,
    private readonly b = 0.75,
    private readonly topK = 3,
    private readonly minScore = 0.1,
  ) {}

  async findStartNodes(
    query: string,
    nodes: ReadonlyMap<string, OntologyNode>,
  ): Promise<string[]> {
    if (nodes.size === 0) return [];
    const qTokens = tokenize(query);
    if (qTokens.length === 0) {
      return new KeywordMappingStrategy().findStartNodes(query, nodes);
    }

    // Build the per-node "document" tokens
    const docs: Array<{ id: string; tokens: string[] }> = [];
    for (const [id, node] of nodes) {
      const text = `${id} ${node.description} ${(node.keywords ?? []).join(" ")}`;
      docs.push({ id, tokens: tokenize(text) });
    }

    const N = docs.length;
    const avgLen = docs.reduce((s, d) => s + d.tokens.length, 0) / Math.max(N, 1);

    // Document frequency
    const df = new Map<string, number>();
    for (const d of docs) {
      for (const t of new Set(d.tokens)) df.set(t, (df.get(t) ?? 0) + 1);
    }

    // Score each node
    const scored = docs.map((d) => {
      const tf = new Map<string, number>();
      for (const t of d.tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
      const docLen = d.tokens.length;
      let score = 0;
      for (const term of qTokens) {
        const tfVal = tf.get(term) ?? 0;
        if (tfVal === 0) continue;
        const dfVal = df.get(term) ?? 0;
        // BM25 IDF — log((N - df + 0.5) / (df + 0.5) + 1)
        const idf = Math.log((N - dfVal + 0.5) / (dfVal + 0.5) + 1);
        const tfNorm =
          (tfVal * (this.k1 + 1)) /
          (tfVal + this.k1 * (1 - this.b + (this.b * docLen) / Math.max(avgLen, 1)));
        score += idf * tfNorm;
      }
      return { id: d.id, score };
    });

    scored.sort((a, b) => b.score - a.score);
    const top = scored.filter((x) => x.score >= this.minScore).slice(0, this.topK);
    if (top.length > 0) return top.map((x) => x.id);

    // Fallback to keyword if BM25 found nothing
    return new KeywordMappingStrategy().findStartNodes(query, nodes);
  }
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[\s\p{P}]+/u)
    .filter((w) => w.length > 1);
}
