import { describe, expect, it } from "vitest";
import type {
  AnswerResult,
  BenchmarkQuery,
  DatasetBundle,
  JudgeContract,
  JudgeResult,
  RetrievalResult,
} from "./contracts.js";
import {
  bootstrapMean95Ci,
  compareFrameworkPairs,
  contextPrecisionForQuery,
  evidenceRecallForQuery,
  scoreDatasetFramework,
} from "./metrics.js";

const primaryQuery: BenchmarkQuery = {
  id: "q1",
  text: "question",
  referenceAnswer: "answer",
  goldEvidenceIds: ["e1", "e2"],
  goldEvidenceText: ["gold evidence"],
  answerable: true,
  category: "Fact",
  metadata: {},
};

const bundle: DatasetBundle = {
  id: "graphrag-bench-medical",
  track: "static-kb",
  documents: [],
  provenance: { source: "test", version: "1", license: "test" },
  queries: [primaryQuery],
};

const retrieval: RetrievalResult = {
  datasetId: bundle.id,
  frameworkId: "kontext-brain",
  queryId: "q1",
  status: "ok",
  evidence: [
    { id: "e1", sourceId: "s", text: "gold evidence", score: 1, rank: 1, metadata: {} },
    { id: "noise", sourceId: "s", text: "noise", score: 0.5, rank: 2, metadata: {} },
  ],
  latencyMs: 10,
  inputTokens: null,
  error: null,
  frameworkVersion: "test",
  configDigest: "test",
};

const answer: AnswerResult = {
  datasetId: bundle.id,
  frameworkId: "kontext-brain",
  queryId: "q1",
  status: "ok",
  output: { answer: "answer", citations: ["e1"], abstained: false, abstentionReason: null },
  latencyMs: 20,
  inputTokens: 10,
  outputTokens: 2,
  error: null,
  inputDigest: "answer-input",
};

const judgementOutput: JudgeContract = {
  answerCorrectness: 1,
  completeness: 0.8,
  strictFaithfulness: 1,
  citationPrecision: 1,
  citationRecall: 0.5,
  acceptableAbstention: false,
  claims: [
    { claim: "answer", supported: true, correct: true, citations: ["e1"], reason: "entailed" },
  ],
};

const judgement: JudgeResult = {
  datasetId: bundle.id,
  frameworkId: "kontext-brain",
  queryId: "q1",
  status: "ok",
  output: judgementOutput,
  latencyMs: 30,
  inputTokens: 20,
  outputTokens: 5,
  error: null,
  inputDigest: "judge-input",
};

describe("rag eval metrics", () => {
  it("separates evidence recall from context precision", () => {
    expect(evidenceRecallForQuery(primaryQuery, retrieval.evidence)).toBe(1);
    expect(contextPrecisionForQuery(primaryQuery, retrieval.evidence)).toBe(0.5);
  });

  it("credits every provenance source represented by a bundled native context", () => {
    const query = {
      ...primaryQuery,
      goldEvidenceIds: ["document-a", "document-b"],
      goldEvidenceText: [],
    };
    const bundledEvidence = [
      {
        id: "native-context",
        sourceId: "bundled-context",
        sourceIds: ["document-a", "document-b"],
        text: "Combined native context",
        score: 1,
        rank: 1,
        metadata: {},
      },
    ];

    expect(evidenceRecallForQuery(query, bundledEvidence)).toBe(1);
    expect(contextPrecisionForQuery(query, bundledEvidence)).toBe(1);
  });

  it("reports per-dataset metrics without a combined score", () => {
    const score = scoreDatasetFramework(
      bundle,
      "kontext-brain",
      [retrieval],
      [answer],
      [judgement],
    );
    expect(score.answerCorrectness).toBe(1);
    expect(score.retrievalQueries).toBe(1);
    expect(score.retrievalCompleted).toBe(1);
    expect(score.claimRecall).toBe(0.8);
    expect(score.claimF1).toBeCloseTo(8 / 9);
    expect(score.citationF1).toBeCloseTo(2 / 3);
    expect(score.endToEndLatencyP95Ms).toBe(60);
    expect(score).not.toHaveProperty("overallScore");
  });

  it("scores full retrieval separately from the answer/judge sample", () => {
    const secondQuery = { ...primaryQuery, id: "q2" };
    const sampledBundle = { ...bundle, queries: [primaryQuery, secondQuery] };
    const secondRetrieval = { ...retrieval, queryId: "q2" };
    const score = scoreDatasetFramework(
      sampledBundle,
      "kontext-brain",
      [retrieval, secondRetrieval],
      [answer],
      [judgement],
      [primaryQuery],
    );

    expect(score).toMatchObject({
      retrievalQueries: 2,
      retrievalCompleted: 2,
      queries: 1,
      completed: 1,
      blocked: 0,
    });
  });

  it("scores failed retrieval queries as zero instead of dropping them from averages", () => {
    const secondQuery = { ...primaryQuery, id: "q2" };
    const expandedBundle = { ...bundle, queries: [primaryQuery, secondQuery] };
    const failedRetrieval: RetrievalResult = {
      ...retrieval,
      queryId: "q2",
      status: "error",
      evidence: [],
      error: "failed",
    };

    const score = scoreDatasetFramework(
      expandedBundle,
      "kontext-brain",
      [retrieval, failedRetrieval],
      [answer],
      [judgement],
      [primaryQuery],
    );

    expect(score.evidenceRecallAtK).toBe(0.5);
    expect(score.contextPrecision).toBe(0.25);
    expect(score.retrievalErrors).toBe(1);
  });

  it("reports paired framework differences on shared completed queries", () => {
    const otherRetrieval = { ...retrieval, frameworkId: "vector-rag-reranker" as const };
    const otherJudgement = {
      ...judgement,
      frameworkId: "vector-rag-reranker" as const,
      output: { ...judgementOutput, answerCorrectness: 0 },
    };
    const comparisons = compareFrameworkPairs(
      bundle,
      ["kontext-brain", "vector-rag-reranker"],
      [retrieval, otherRetrieval],
      [judgement, otherJudgement],
    );
    const correctness = comparisons.find(
      (comparison) => comparison.metric === "answer-correctness",
    );
    expect(correctness).toMatchObject({
      pairedQueries: 1,
      meanDifferenceLeftMinusRight: 1,
      difference95Ci: { low: 1, high: 1 },
    });
  });

  it("bootstraps deterministically", () => {
    expect(bootstrapMean95Ci([0, 1, 1], 100, 42)).toEqual(bootstrapMean95Ci([0, 1, 1], 100, 42));
  });
});
