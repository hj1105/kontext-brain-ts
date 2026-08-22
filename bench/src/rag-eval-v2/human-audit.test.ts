import { describe, expect, it } from "vitest";
import type { AnswerResult, DatasetBundle, FrameworkId, RetrievalResult } from "./contracts.js";
import { createBlindHumanAuditSample } from "./human-audit.js";

describe("human audit sampling", () => {
  it("balances frameworks and hides their identity from audit rows", () => {
    const frameworks: FrameworkId[] = [
      "kontext-brain",
      "vector-rag-reranker",
      "microsoft-graphrag",
      "lightrag",
      "hipporag2",
    ];
    const queries = Array.from({ length: 30 }, (_, index) => ({
      id: `q${index}`,
      text: `question ${index}`,
      referenceAnswer: `answer ${index}`,
      goldEvidenceIds: [`e${index}`],
      goldEvidenceText: [`evidence ${index}`],
      answerable: true,
      category: index % 2 === 0 ? "Fact" : "Complex",
      metadata: {},
    }));
    const bundle: DatasetBundle = {
      id: "frames",
      track: "static-kb",
      documents: [],
      queries,
      provenance: { source: "test", version: "1", license: "test" },
    };
    const retrievals: RetrievalResult[] = [];
    const answers: AnswerResult[] = [];
    for (const frameworkId of frameworks) {
      for (const query of queries) {
        retrievals.push({
          datasetId: bundle.id,
          frameworkId,
          queryId: query.id,
          status: "ok",
          evidence: [
            {
              id: query.goldEvidenceIds[0]!,
              sourceId: "s",
              text: "evidence",
              score: 1,
              rank: 1,
              metadata: {},
            },
          ],
          latencyMs: 1,
          inputTokens: null,
          error: null,
          frameworkVersion: "test",
          configDigest: "test",
        });
        answers.push({
          datasetId: bundle.id,
          frameworkId,
          queryId: query.id,
          status: "ok",
          output: {
            answer: "answer",
            citations: [query.goldEvidenceIds[0]!],
            abstained: false,
            abstentionReason: null,
          },
          latencyMs: 1,
          inputTokens: null,
          outputTokens: null,
          error: null,
          inputDigest: "answer-input",
        });
      }
    }
    const sample = createBlindHumanAuditSample(bundle, retrievals, answers, 100);
    expect(sample.rows).toHaveLength(100);
    expect(sample.mapping).toHaveLength(100);
    expect(sample.rows.every((row) => !("frameworkId" in row))).toBe(true);
    const counts = new Map<FrameworkId, number>();
    for (const mapping of sample.mapping)
      counts.set(mapping.frameworkId, (counts.get(mapping.frameworkId) ?? 0) + 1);
    expect([...counts.values()]).toEqual([20, 20, 20, 20, 20]);
  });
});
