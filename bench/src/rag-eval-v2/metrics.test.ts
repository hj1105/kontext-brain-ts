import { describe, expect, it } from "vitest";
import type {
  AnswerResult,
  BenchmarkQuery,
  DatasetBundle,
  JudgeContract,
  JudgeResult,
  RetrievalResult,
  RetrievedEvidence,
} from "./contracts.js";
import {
  answerabilityJointAccuracyForResults,
  bootstrapMean95Ci,
  compareFrameworkPairs,
  contextPrecisionForQuery,
  evidenceRecallForQuery,
  ndcgForQuery,
  nearestRankPercentileOrNull,
  robustnessDropForQueries,
  scoreDatasetFramework,
} from "./metrics.js";

const baseQuery: BenchmarkQuery = {
  id: "q1",
  text: "question",
  referenceAnswer: "answer",
  goldEvidenceIds: ["e1", "e2"],
  goldEvidenceText: ["gold evidence"],
  answerable: true,
  category: "Fact",
  metadata: {},
};

const relevantEvidence: RetrievedEvidence = {
  id: "e1",
  sourceId: "s",
  text: "gold evidence",
  score: 1,
  rank: 1,
  metadata: {},
};

const noiseEvidence: RetrievedEvidence = {
  id: "noise",
  sourceId: "s",
  text: "noise",
  score: 0.5,
  rank: 2,
  metadata: {},
};

const bundle: DatasetBundle = {
  id: "graphrag-bench-medical",
  track: "static-kb",
  documents: [],
  provenance: { source: "test", version: "1", license: "test" },
  queries: [baseQuery],
};

const retrieval: RetrievalResult = {
  datasetId: bundle.id,
  frameworkId: "kontext-brain",
  queryId: "q1",
  status: "ok",
  evidence: [relevantEvidence, noiseEvidence],
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
  clarity: 0.9,
  conciseness: 0.8,
  fluency: 0.95,
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
    expect(evidenceRecallForQuery(baseQuery, retrieval.evidence)).toBe(1);
    expect(contextPrecisionForQuery(baseQuery, retrieval.evidence)).toBe(0.5);
    expect(ndcgForQuery(baseQuery, retrieval.evidence)).toBe(1);
  });

  it("discounts relevant evidence that appears later in the ranking", () => {
    const evidence = [
      { ...noiseEvidence, rank: 1 },
      { ...relevantEvidence, rank: 2 },
    ];
    expect(ndcgForQuery(baseQuery, evidence)).toBeCloseTo(1 / Math.log2(3));
  });

  it("does not let duplicate evidence inflate nDCG", () => {
    const query = {
      ...baseQuery,
      goldEvidenceIds: ["gold-source"],
      goldEvidenceText: [],
    };
    const evidence = [
      { ...noiseEvidence, rank: 1 },
      { ...relevantEvidence, id: "first-copy", sourceId: "gold-source", rank: 2 },
      { ...relevantEvidence, id: "second-copy", sourceId: "gold-source", rank: 3 },
    ];

    expect(ndcgForQuery(query, evidence)).toBeCloseTo(1 / Math.log2(3));
  });

  it("credits every provenance source represented by a bundled native context", () => {
    const query = {
      ...baseQuery,
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
    expect(score.claimSupportPrecision).toBe(1);
    expect(score.claimF1).toBeCloseTo(8 / 9);
    expect(score.citationPrecision).toBe(1);
    expect(score.citationRecall).toBe(0.5);
    expect(score.citationF1).toBeCloseTo(2 / 3);
    expect(score.clarity).toBe(0.9);
    expect(score.conciseness).toBe(0.8);
    expect(score.fluency).toBe(0.95);
    expect(score.answerabilityJointAccuracy).toBeNull();
    expect(score.robustnessDrop).toBeNull();
    expect(score.retrievalLatencyP95Ms).toBe(10);
    expect(score.queryToAnswerLatencyP95Ms).toBe(30);
    expect(score.endToEndLatencyP95Ms).toBe(60);
    expect(score).not.toHaveProperty("overallScore");
  });

  it("uses the frozen nearest-rank percentile definition", () => {
    expect(nearestRankPercentileOrNull([], 0.95)).toBeNull();
    expect(
      nearestRankPercentileOrNull(
        Array.from({ length: 20 }, (_, index) => index + 1),
        0.95,
      ),
    ).toBe(19);
    expect(nearestRankPercentileOrNull([30, 10, 20], 0.5)).toBe(20);
    expect(() => nearestRankPercentileOrNull([1], 0)).toThrow("percentile must be in (0, 1]");
  });

  it("keeps user-facing query-to-answer latency independent of the judge", () => {
    const score = scoreDatasetFramework(bundle, "kontext-brain", [retrieval], [answer], []);

    expect(score.queryToAnswerLatencyP95Ms).toBe(30);
    expect(score.endToEndLatencyP95Ms).toBeNull();
  });

  it("scores full retrieval separately from the answer/judge sample", () => {
    const secondQuery = { ...baseQuery, id: "q2" };
    const sampledBundle = { ...bundle, queries: [baseQuery, secondQuery] };
    const secondRetrieval = { ...retrieval, queryId: "q2" };
    const score = scoreDatasetFramework(
      sampledBundle,
      "kontext-brain",
      [retrieval, secondRetrieval],
      [answer],
      [judgement],
      [baseQuery],
    );

    expect(score).toMatchObject({
      retrievalQueries: 2,
      retrievalCompleted: 2,
      queries: 1,
      completed: 1,
      blocked: 0,
    });
  });

  it("class-balances answerable answers and acceptable unanswerable abstentions", () => {
    const answerableQuery = baseQuery;
    const unanswerableQuery = {
      ...answerableQuery,
      id: "q-unanswerable",
      answerable: false,
      referenceAnswer: null,
    };
    const answers = new Map<string, AnswerResult>([
      [answerableQuery.id, answer],
      [
        unanswerableQuery.id,
        {
          ...answer,
          queryId: unanswerableQuery.id,
          output: {
            answer: "",
            citations: [],
            abstained: true,
            abstentionReason: "Outside the supplied KB.",
          },
        },
      ],
    ]);
    const judgements = new Map<string, JudgeResult>([
      [answerableQuery.id, judgement],
      [
        unanswerableQuery.id,
        {
          ...judgement,
          queryId: unanswerableQuery.id,
          output: { ...judgementOutput, acceptableAbstention: true },
        },
      ],
    ]);

    expect(
      answerabilityJointAccuracyForResults(
        [answerableQuery, unanswerableQuery],
        answers,
        judgements,
      ),
    ).toBe(1);
  });

  it("measures paired robustness drop from an explicit baseline", () => {
    const baseline = {
      ...baseQuery,
      id: "robust-base",
      metadata: { robustnessGroupId: "g1", robustnessRole: "baseline" },
    };
    const perturbation = {
      ...baseline,
      id: "robust-order",
      metadata: { robustnessGroupId: "g1", robustnessRole: "perturbed" },
    };
    const judgements = new Map<string, JudgeResult>([
      [
        baseline.id,
        {
          ...judgement,
          queryId: baseline.id,
          output: { ...judgementOutput, answerCorrectness: 0.9 },
        },
      ],
      [
        perturbation.id,
        {
          ...judgement,
          queryId: perturbation.id,
          output: { ...judgementOutput, answerCorrectness: 0.6 },
        },
      ],
    ]);

    expect(robustnessDropForQueries([baseline, perturbation], judgements)).toBeCloseTo(0.3);
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
    expect(comparisons.find((comparison) => comparison.metric === "ndcg-at-k")).toMatchObject({
      pairedQueries: 1,
      meanDifferenceLeftMinusRight: 0,
    });
    expect(
      comparisons.find((comparison) => comparison.metric === "claim-support-precision"),
    ).toMatchObject({ pairedQueries: 1, meanDifferenceLeftMinusRight: 0 });
  });

  it("bootstraps deterministically", () => {
    expect(bootstrapMean95Ci([0, 1, 1], 100, 42)).toEqual(bootstrapMean95Ci([0, 1, 1], 100, 42));
  });
});
