export const RAG_EVAL_SCHEMA_VERSION = "2.0.0" as const;

export type DatasetTrack = "static-kb" | "dynamic-api" | "large-corpus" | "multilingual-report";

export type DatasetId =
  | "graphrag-bench-medical"
  | "graphrag-bench-novel"
  | "beir-scifact"
  | "beir-nfcorpus"
  | "garage"
  | "frames"
  | "uaeval-kontext"
  | "stable-rag"
  | "crag"
  | "trec-rag"
  | "ragtime";

export type FrameworkId =
  | "kontext-brain"
  | "vector-rag-reranker"
  | "microsoft-graphrag"
  | "lightrag"
  | "hipporag2";

export type MetricId =
  | "evidence-recall-at-k"
  | "ndcg-at-k"
  | "context-precision"
  | "answer-correctness"
  | "claim-recall"
  | "claim-support-precision"
  | "claim-f1"
  | "strict-faithfulness"
  | "citation-precision"
  | "citation-recall"
  | "citation-f1"
  | "acceptable-abstention"
  | "answerability-joint-accuracy"
  | "permutation-sensitivity"
  | "robustness-drop"
  | "clarity"
  | "conciseness"
  | "fluency"
  | "crag-truthfulness"
  | "latency-p95"
  | "input-tokens"
  | "cost";

export interface CorpusDocument {
  readonly id: string;
  readonly title: string;
  readonly text: string;
  readonly sourceId: string;
  readonly metadata: Readonly<Record<string, string | number | boolean | null>>;
}

export interface BenchmarkQuery {
  readonly id: string;
  readonly text: string;
  readonly referenceAnswer: string | null;
  readonly goldEvidenceIds: readonly string[];
  readonly goldEvidenceText: readonly string[];
  readonly answerable: boolean;
  readonly category: string;
  readonly metadata: Readonly<Record<string, string | number | boolean | null>>;
}

export interface DatasetBundle {
  readonly id: DatasetId;
  readonly track: DatasetTrack;
  readonly documents: readonly CorpusDocument[];
  readonly queries: readonly BenchmarkQuery[];
  readonly provenance: {
    readonly source: string;
    readonly version: string;
    readonly license: string;
  };
}

export interface RetrievedEvidence {
  readonly id: string;
  readonly sourceId: string;
  readonly sourceIds?: readonly string[];
  readonly text: string;
  readonly score: number;
  readonly rank: number;
  readonly metadata: Readonly<Record<string, string | number | boolean | null>>;
}

export interface RetrievalResult {
  readonly datasetId: DatasetId;
  readonly frameworkId: FrameworkId;
  readonly queryId: string;
  readonly status: "ok" | "blocked" | "unsupported" | "error";
  readonly evidence: readonly RetrievedEvidence[];
  readonly latencyMs: number;
  readonly inputTokens: number | null;
  readonly error: string | null;
  readonly frameworkVersion: string;
  readonly configDigest: string;
  readonly answerPolicy?: AnswerPolicy;
  /** Optional wall-clock boundaries used by clean latency contamination checks. */
  readonly startedAt?: string;
  readonly completedAt?: string;
}

export type AnswerPolicy = "supported-evidence-needs";

export interface AnswerContract {
  readonly answer: string;
  readonly citations: readonly string[];
  readonly abstained: boolean;
  readonly abstentionReason: string | null;
}

export interface AnswerResult {
  readonly datasetId: DatasetId;
  readonly frameworkId: FrameworkId;
  readonly queryId: string;
  readonly status: "ok" | "blocked" | "unsupported" | "error";
  readonly output: AnswerContract;
  readonly latencyMs: number;
  readonly inputTokens: number | null;
  readonly outputTokens: number | null;
  readonly error: string | null;
  readonly inputDigest: string;
  /** Optional wall-clock boundaries used by clean latency contamination checks. */
  readonly startedAt?: string;
  readonly completedAt?: string;
}

export interface ClaimJudgement {
  readonly claim: string;
  readonly supported: boolean;
  readonly correct: boolean;
  readonly citations: readonly string[];
  readonly reason: string;
}

export interface JudgeContract {
  readonly answerCorrectness: number;
  readonly completeness: number;
  readonly strictFaithfulness: number;
  readonly citationPrecision: number;
  readonly citationRecall: number;
  readonly acceptableAbstention: boolean;
  /** Present for judge-policy-v3 and later. Older preserved artifacts omit these fields. */
  readonly clarity?: number;
  readonly conciseness?: number;
  readonly fluency?: number;
  readonly claims: readonly ClaimJudgement[];
}

export interface JudgeResult {
  readonly datasetId: DatasetId;
  readonly frameworkId: FrameworkId;
  readonly queryId: string;
  readonly status: "ok" | "blocked" | "unsupported" | "error";
  readonly output: JudgeContract | null;
  readonly latencyMs: number;
  readonly inputTokens: number | null;
  readonly outputTokens: number | null;
  readonly error: string | null;
  readonly inputDigest: string;
  /** Optional wall-clock boundaries used by clean latency contamination checks. */
  readonly startedAt?: string;
  readonly completedAt?: string;
}

export interface FrameworkDoctorResult {
  readonly frameworkId: FrameworkId;
  readonly status: "ready" | "blocked" | "unsupported";
  readonly version: string;
  readonly detail: string;
}

export interface DatasetDoctorResult {
  readonly datasetId: DatasetId;
  readonly status: "ready" | "blocked" | "unsupported";
  readonly detail: string;
}

export interface RunCheckpoint<T> {
  readonly schemaVersion: typeof RAG_EVAL_SCHEMA_VERSION;
  readonly manifestDigest: string;
  readonly createdAt: string;
  readonly updatedAt: string;
  readonly records: readonly T[];
}
