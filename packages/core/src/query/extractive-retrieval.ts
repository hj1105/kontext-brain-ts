import type { MetaDocument } from "../graph/layered-models.js";
import type { ContentFetcherRegistry } from "./content-fetcher.js";

/**
 * Extractive retrieval: skip the final LLM and return the most relevant
 * sentences from retrieved documents as the answer.
 *
 * Benchmark rationale: on factual tech-docs QA against a small-to-medium
 * corpus, a 1.5B LLM doing "generation" barely outperforms simple
 * sentence extraction by keyword overlap. Skipping the final LLM cuts
 * latency by ~99% (ms vs seconds) with no quality loss on extractive-style
 * questions.
 *
 * Limitations:
 *   - Fails when the answer requires synthesis across multiple sentences
 *     or paraphrasing of scattered facts (see v12 vs v13 on "rotate secrets"
 *     in the bench corpus).
 *   - Returns raw corpus text — not suitable when the user expects
 *     conversational answers.
 */
export class ExtractiveRetriever {
  constructor(
    private readonly fetcherRegistry: ContentFetcherRegistry,
    private readonly topSentences = 2,
    private readonly minSentenceLength = 15,
  ) {}

  async answer(
    query: string,
    metaDocs: readonly MetaDocument[],
  ): Promise<{ answer: string; retrievedDocIds: string[]; contextChars: number }> {
    const q = query
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 2);

    const retrievedIds = new Set<string>();
    type Scored = { sentence: string; score: number };
    const scored: Scored[] = [];

    for (const meta of metaDocs) {
      const content = await this.fetcherRegistry.fetch(meta);
      retrievedIds.add(meta.id);
      const sentences = content.body
        .split(/(?<=[.!?])\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > this.minSentenceLength);
      for (const s of sentences) {
        const lc = s.toLowerCase();
        const hits = q.filter((w) => lc.includes(w)).length;
        if (hits === 0) continue;
        // Proximity bonus: reward density (more hits per unit length)
        const density = hits / Math.max(s.length / 100, 1);
        scored.push({ sentence: s, score: hits + density * 0.5 });
      }
    }

    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, this.topSentences).map((x) => x.sentence);
    const answer = top.join(". ");
    return {
      answer,
      retrievedDocIds: Array.from(retrievedIds),
      contextChars: answer.length,
    };
  }
}
