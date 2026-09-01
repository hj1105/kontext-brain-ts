import { describe, expect, it } from "vitest";
import type { EmbeddingClient, EmbeddingInput } from "../rag-eval-v2/openai-embeddings.js";
import {
  EmbeddingRagRetriever,
  buildRagCorpus,
  renderRetrievedContext,
  retrievalQuery,
} from "./rag-context.js";
import { codeQualityScenarios } from "./scenarios.js";

/**
 * A deterministic stand-in for the embedding model: each text becomes a bag of
 * word counts, so cosine similarity behaves like lexical overlap. It is weaker
 * than a real embedding model, which makes the retrieval assertions below a
 * lower bound rather than a flattering one.
 */
function lexicalClient(): EmbeddingClient {
  const vocabulary = new Map<string, number>();
  const vectorFor = (text: string): number[] => {
    const counts = new Map<number, number>();
    for (const word of text.toLowerCase().match(/[a-z][a-z0-9]{2,}/g) ?? []) {
      let index = vocabulary.get(word);
      if (index === undefined) {
        index = vocabulary.size;
        vocabulary.set(word, index);
      }
      counts.set(index, (counts.get(index) ?? 0) + 1);
    }
    const values = new Array<number>(4096).fill(0);
    for (const [index, count] of counts) values[index % 4096] = count;
    return values;
  };
  return {
    model: "lexical-fixture",
    dimensions: 4096,
    async embed(inputs: readonly EmbeddingInput[]) {
      return inputs.map((input) => ({ id: input.id, values: vectorFor(input.text) }));
    },
    getUsage() {
      return { requests: 0, inputTokens: 0, totalTokens: 0 };
    },
  };
}

describe("retrieval arm corpus", () => {
  it("holds every scenario's rules so retrieval faces real distractors", () => {
    const corpus = buildRagCorpus(codeQualityScenarios);
    const expected = codeQualityScenarios.reduce(
      (total, scenario) => total + scenario.rules.length,
      0,
    );
    expect(corpus).toHaveLength(expected);
    // More than one scenario must be present, or the arm is an oracle.
    expect(new Set(corpus.map((document) => document.scenarioId)).size).toBe(
      codeQualityScenarios.length,
    );
  });

  it("never leaks a canonical term through the retrieval query", () => {
    for (const scenario of codeQualityScenarios) {
      const query = retrievalQuery(scenario);
      for (const term of scenario.canonicalTerms) expect(query).not.toContain(term);
    }
  });

  it("returns a ranked top-k drawn from the whole corpus", async () => {
    const corpus = buildRagCorpus(codeQualityScenarios);
    const retriever = new EmbeddingRagRetriever(corpus, lexicalClient());
    const retrieved = await retriever.retrieve(codeQualityScenarios[0] as never, 5);
    expect(retrieved).toHaveLength(5);
    const scores = retrieved.map((rule) => rule.score);
    expect([...scores].sort((left, right) => right - left)).toEqual(scores);
    expect(new Set(retrieved.map((rule) => rule.ruleId)).size).toBe(retrieved.length);
    // Retrieval quality belongs to the embedding model, not to this lexical
    // stand-in, so it is measured against the real model and recorded as run
    // metadata rather than asserted here.
  });

  it("renders retrieved rules as numbered documentation", () => {
    const rendered = renderRetrievedContext([
      { ruleId: "decision:x", scenarioId: "s", text: "Decision: do the thing", score: 1 },
    ]);
    expect(rendered).toContain("[1] decision:x");
    expect(rendered).toContain("Decision: do the thing");
  });
});
