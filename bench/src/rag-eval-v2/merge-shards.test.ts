import { describe, expect, it } from "vitest";
import type { AnswerResult, DatasetBundle, JudgeResult, RetrievalResult } from "./contracts.js";
import { DEFAULT_RAG_EVAL_MANIFEST } from "./manifest.js";
import { type IndexedShard, mergeAnswerShardRecords } from "./merge-answer-shards.js";
import { mergeJudgeShardRecords } from "./merge-judge-shards.js";
import { answerInputDigest, judgeInputDigest } from "./pipeline.js";

describe("answer shard merge validation", () => {
  it("rejects stale digests instead of accepting a primary fallback", () => {
    const fixture = mergeFixture();
    const answer0 = requiredRecord(fixture.answers, 0);
    const answer1 = requiredRecord(fixture.answers, 1);
    const stale = { ...answer0, inputDigest: "stale" };

    expect(() =>
      mergeAnswerShardRecords({
        manifest: DEFAULT_RAG_EVAL_MANIFEST,
        bundle: fixture.bundle,
        evaluationQueries: fixture.bundle.queries,
        retrievals: fixture.retrievals,
        shardCount: 2,
        shards: [
          { shardIndex: 0, records: [stale] },
          { shardIndex: 1, records: [answer1] },
        ],
      }),
    ).toThrow(/input digest mismatch/i);
  });

  it("rejects missing and misassigned shard records", () => {
    const fixture = mergeFixture();
    const answer0 = requiredRecord(fixture.answers, 0);
    const answer1 = requiredRecord(fixture.answers, 1);
    const missing: readonly IndexedShard<AnswerResult>[] = [
      { shardIndex: 0, records: [answer0] },
      { shardIndex: 1, records: [] },
    ];
    expect(() =>
      mergeAnswerShardRecords({
        manifest: DEFAULT_RAG_EVAL_MANIFEST,
        bundle: fixture.bundle,
        evaluationQueries: fixture.bundle.queries,
        retrievals: fixture.retrievals,
        shardCount: 2,
        shards: missing,
      }),
    ).toThrow(/missing answer shard record/i);

    expect(() =>
      mergeAnswerShardRecords({
        manifest: DEFAULT_RAG_EVAL_MANIFEST,
        bundle: fixture.bundle,
        evaluationQueries: fixture.bundle.queries,
        retrievals: fixture.retrievals,
        shardCount: 2,
        shards: [
          { shardIndex: 0, records: [answer1] },
          { shardIndex: 1, records: [answer0] },
        ],
      }),
    ).toThrow(/assigned to shard/i);
  });

  it("rejects a self-consistent answer shard copied from another dataset and framework", () => {
    const fixture = mergeFixture();
    const query0 = requiredRecord(fixture.bundle.queries, 0);
    const retrieval0 = requiredRecord(fixture.retrievals, 0);
    const answer0 = requiredRecord(fixture.answers, 0);
    const answer1 = requiredRecord(fixture.answers, 1);
    const foreignRetrieval: RetrievalResult = {
      ...retrieval0,
      datasetId: "graphrag-bench-novel",
      frameworkId: "vector-rag-reranker",
    };
    const foreignAnswer: AnswerResult = {
      ...answer0,
      datasetId: foreignRetrieval.datasetId,
      frameworkId: foreignRetrieval.frameworkId,
      inputDigest: answerInputDigest(DEFAULT_RAG_EVAL_MANIFEST, query0, foreignRetrieval),
    };

    expect(() =>
      mergeAnswerShardRecords({
        manifest: DEFAULT_RAG_EVAL_MANIFEST,
        bundle: fixture.bundle,
        evaluationQueries: fixture.bundle.queries,
        retrievals: [foreignRetrieval, requiredRecord(fixture.retrievals, 1)],
        shardCount: 2,
        shards: [
          { shardIndex: 0, records: [foreignAnswer] },
          { shardIndex: 1, records: [answer1] },
        ],
      }),
    ).toThrow(/retrieval identity mismatch/i);
  });
});

describe("judge shard merge validation", () => {
  it("rejects stale, missing, and misassigned judgement records", () => {
    const fixture = mergeFixture();
    const judgement0 = requiredRecord(fixture.judgements, 0);
    const judgement1 = requiredRecord(fixture.judgements, 1);
    const common = {
      manifest: DEFAULT_RAG_EVAL_MANIFEST,
      bundle: fixture.bundle,
      evaluationQueries: fixture.bundle.queries,
      retrievals: fixture.retrievals,
      answers: fixture.answers,
      shardCount: 2,
    } as const;
    expect(() =>
      mergeJudgeShardRecords({
        ...common,
        shards: [
          { shardIndex: 0, records: [{ ...judgement0, inputDigest: "stale" }] },
          { shardIndex: 1, records: [judgement1] },
        ],
      }),
    ).toThrow(/input digest mismatch/i);
    expect(() =>
      mergeJudgeShardRecords({
        ...common,
        shards: [
          { shardIndex: 0, records: [judgement0] },
          { shardIndex: 1, records: [] },
        ],
      }),
    ).toThrow(/missing judge shard record/i);
    expect(() =>
      mergeJudgeShardRecords({
        ...common,
        shards: [
          { shardIndex: 0, records: [judgement1] },
          { shardIndex: 1, records: [judgement0] },
        ],
      }),
    ).toThrow(/assigned to shard/i);
  });

  it("rejects current judgements when their upstream answer digest is stale", () => {
    const fixture = mergeFixture();
    const answer0 = requiredRecord(fixture.answers, 0);
    const answer1 = requiredRecord(fixture.answers, 1);
    const judgement0 = requiredRecord(fixture.judgements, 0);
    const judgement1 = requiredRecord(fixture.judgements, 1);
    expect(() =>
      mergeJudgeShardRecords({
        manifest: DEFAULT_RAG_EVAL_MANIFEST,
        bundle: fixture.bundle,
        evaluationQueries: fixture.bundle.queries,
        retrievals: fixture.retrievals,
        answers: [{ ...answer0, inputDigest: "stale" }, answer1],
        shardCount: 2,
        shards: [
          { shardIndex: 0, records: [judgement0] },
          { shardIndex: 1, records: [judgement1] },
        ],
      }),
    ).toThrow(/answer input digest mismatch/i);
  });

  it("rejects self-consistent judgements copied from another dataset and framework", () => {
    const fixture = mergeFixture();
    const query0 = requiredRecord(fixture.bundle.queries, 0);
    const retrieval0 = requiredRecord(fixture.retrievals, 0);
    const retrieval1 = requiredRecord(fixture.retrievals, 1);
    const answer0 = requiredRecord(fixture.answers, 0);
    const answer1 = requiredRecord(fixture.answers, 1);
    const judgement0 = requiredRecord(fixture.judgements, 0);
    const judgement1 = requiredRecord(fixture.judgements, 1);
    const foreignRetrieval: RetrievalResult = {
      ...retrieval0,
      datasetId: "graphrag-bench-novel",
      frameworkId: "vector-rag-reranker",
    };
    const foreignAnswer: AnswerResult = {
      ...answer0,
      datasetId: foreignRetrieval.datasetId,
      frameworkId: foreignRetrieval.frameworkId,
      inputDigest: answerInputDigest(DEFAULT_RAG_EVAL_MANIFEST, query0, foreignRetrieval),
    };
    const foreignJudgement: JudgeResult = {
      ...judgement0,
      datasetId: foreignRetrieval.datasetId,
      frameworkId: foreignRetrieval.frameworkId,
      inputDigest: judgeInputDigest(
        DEFAULT_RAG_EVAL_MANIFEST,
        query0,
        foreignRetrieval,
        foreignAnswer,
      ),
    };

    expect(() =>
      mergeJudgeShardRecords({
        manifest: DEFAULT_RAG_EVAL_MANIFEST,
        bundle: fixture.bundle,
        evaluationQueries: fixture.bundle.queries,
        retrievals: [foreignRetrieval, retrieval1],
        answers: [foreignAnswer, answer1],
        shardCount: 2,
        shards: [
          { shardIndex: 0, records: [foreignJudgement] },
          { shardIndex: 1, records: [judgement1] },
        ],
      }),
    ).toThrow(/retrieval identity mismatch/i);
  });
});

function mergeFixture(): {
  bundle: DatasetBundle;
  retrievals: RetrievalResult[];
  answers: AnswerResult[];
  judgements: JudgeResult[];
} {
  const bundle: DatasetBundle = {
    id: "graphrag-bench-medical",
    track: "static-kb",
    documents: [],
    queries: [query("q1"), query("q2")],
    provenance: { source: "test", version: "1", license: "test" },
  };
  const retrievals = bundle.queries.map(
    (item, index): RetrievalResult => ({
      datasetId: bundle.id,
      frameworkId: "kontext-brain",
      queryId: item.id,
      status: "ok",
      evidence: [
        {
          id: `e${index + 1}`,
          sourceId: "source",
          text: `Evidence ${index + 1}`,
          score: 1,
          rank: 1,
          metadata: {},
        },
      ],
      latencyMs: 1,
      inputTokens: 1,
      error: null,
      frameworkVersion: "v13",
      configDigest: "v13",
      answerPolicy: "supported-evidence-needs",
    }),
  );
  const answers = bundle.queries.map(
    (item, index): AnswerResult => ({
      datasetId: bundle.id,
      frameworkId: "kontext-brain",
      queryId: item.id,
      status: "ok",
      output: {
        answer: `Answer ${index + 1} [e${index + 1}]`,
        citations: [`e${index + 1}`],
        abstained: false,
        abstentionReason: null,
      },
      latencyMs: 1,
      inputTokens: 1,
      outputTokens: 1,
      error: null,
      inputDigest: answerInputDigest(DEFAULT_RAG_EVAL_MANIFEST, item, retrievals[index]),
    }),
  );
  const judgements = bundle.queries.map(
    (item, index): JudgeResult => ({
      datasetId: bundle.id,
      frameworkId: "kontext-brain",
      queryId: item.id,
      status: "ok",
      output: {
        answerCorrectness: 1,
        completeness: 1,
        strictFaithfulness: 1,
        citationPrecision: 1,
        citationRecall: 1,
        acceptableAbstention: false,
        claims: [],
      },
      latencyMs: 1,
      inputTokens: 1,
      outputTokens: 1,
      error: null,
      inputDigest: judgeInputDigest(
        DEFAULT_RAG_EVAL_MANIFEST,
        item,
        retrievals[index],
        answers[index],
      ),
    }),
  );
  return { bundle, retrievals, answers, judgements };
}

function query(id: string): DatasetBundle["queries"][number] {
  return {
    id,
    text: `Question ${id}?`,
    referenceAnswer: `Reference ${id}`,
    goldEvidenceIds: [],
    goldEvidenceText: [],
    answerable: true,
    category: "test",
    metadata: {},
  };
}

function requiredRecord<T>(records: readonly T[], index: number): T {
  const record = records[index];
  if (!record) throw new Error(`Missing test record ${index}`);
  return record;
}
