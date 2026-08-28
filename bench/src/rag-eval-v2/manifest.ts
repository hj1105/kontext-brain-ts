import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import {
  type DatasetId,
  type DatasetTrack,
  type FrameworkId,
  type MetricId,
  RAG_EVAL_SCHEMA_VERSION,
} from "./contracts.js";

export interface ModelManifest {
  readonly provider: "openai" | "codex-cli" | "anthropic";
  readonly model: string;
  readonly dimensions?: number;
  readonly reasoningEffort?: "low" | "medium" | "high" | "xhigh";
  readonly execution?: "api" | "codex-exec" | "anthropic-api";
}

export interface FrameworkManifest {
  readonly id: FrameworkId;
  readonly displayName: string;
  readonly implementation: "builtin" | "external-command";
  readonly versionPolicy: "workspace" | "official-pinned";
  readonly pinnedVersion?: string;
  readonly commandEnv?: string;
}

export interface DatasetManifest {
  readonly id: DatasetId;
  readonly displayName: string;
  readonly track: DatasetTrack;
  readonly metrics: readonly MetricId[];
  readonly requiredDataPath?: string;
}

export interface RagEvalManifest {
  readonly schemaVersion: typeof RAG_EVAL_SCHEMA_VERSION;
  readonly benchmarkPolicy: {
    readonly tuning: "development-iterative";
    readonly aggregateAcrossDatasets: false;
    readonly commonAnswerContract: true;
    readonly frameworkInternalChanges: true;
    readonly retrievalQueryScope: "all";
    readonly answerJudgeSamplePerDataset: number;
    readonly answerJudgeSampleSeed: number;
    readonly answerCodexBatchSize: number;
    readonly judgeCodexBatchSize: number;
    readonly codexConcurrency: number;
    readonly judgeCodexConcurrency?: number;
    readonly judgeTimeoutMs?: number;
    readonly humanAuditPerDataset: number;
    readonly maxRetries: number;
    readonly checkpointEvery: number;
  };
  readonly models: {
    readonly embedding: ModelManifest;
    readonly answer: ModelManifest;
    readonly judge: ModelManifest;
  };
  readonly frameworks: readonly FrameworkManifest[];
  readonly datasets: readonly DatasetManifest[];
}

const RETRIEVAL_METRICS = [
  "evidence-recall-at-k",
  "ndcg-at-k",
  "context-precision",
] as const satisfies readonly MetricId[];

const ANSWER_QUALITY_METRICS = [
  "answer-correctness",
  "claim-recall",
  "claim-support-precision",
  "claim-f1",
  "strict-faithfulness",
  "citation-precision",
  "citation-recall",
  "citation-f1",
  "clarity",
  "conciseness",
  "fluency",
] as const satisfies readonly MetricId[];

const RELIABILITY_METRICS = [
  ...ANSWER_QUALITY_METRICS,
  "latency-p95",
  "input-tokens",
  "cost",
] as const satisfies readonly MetricId[];

export const DEFAULT_RAG_EVAL_MANIFEST: RagEvalManifest = {
  schemaVersion: RAG_EVAL_SCHEMA_VERSION,
  benchmarkPolicy: {
    tuning: "development-iterative",
    aggregateAcrossDatasets: false,
    commonAnswerContract: true,
    frameworkInternalChanges: true,
    retrievalQueryScope: "all",
    answerJudgeSamplePerDataset: 200,
    answerJudgeSampleSeed: 20260814,
    answerCodexBatchSize: 1,
    judgeCodexBatchSize: 1,
    codexConcurrency: 1,
    humanAuditPerDataset: 100,
    maxRetries: 3,
    checkpointEvery: 1,
  },
  models: {
    embedding: {
      provider: "openai",
      model: "text-embedding-3-small",
      dimensions: 1536,
      execution: "api",
    },
    answer: {
      provider: "codex-cli",
      model: "gpt-5.6-terra",
      reasoningEffort: "medium",
      execution: "codex-exec",
    },
    judge: {
      provider: "codex-cli",
      model: "gpt-5.6-sol",
      reasoningEffort: "xhigh",
      execution: "codex-exec",
    },
  },
  frameworks: [
    {
      id: "kontext-brain",
      displayName: "kontext-brain",
      implementation: "builtin",
      versionPolicy: "workspace",
    },
    {
      id: "vector-rag-reranker",
      displayName: "Vector RAG + reranker",
      implementation: "builtin",
      versionPolicy: "workspace",
    },
    {
      id: "microsoft-graphrag",
      displayName: "Microsoft GraphRAG",
      implementation: "external-command",
      versionPolicy: "official-pinned",
      pinnedVersion: "3.1.1",
      commandEnv: "RAG_EVAL_GRAPHRAG_COMMAND",
    },
    {
      id: "lightrag",
      displayName: "LightRAG",
      implementation: "external-command",
      versionPolicy: "official-pinned",
      pinnedVersion: "1.5.6",
      commandEnv: "RAG_EVAL_LIGHTRAG_COMMAND",
    },
    {
      id: "hipporag2",
      displayName: "HippoRAG 2",
      implementation: "external-command",
      versionPolicy: "official-pinned",
      pinnedVersion: "2.0.0a4",
      commandEnv: "RAG_EVAL_HIPPORAG_COMMAND",
    },
  ],
  datasets: [
    {
      id: "graphrag-bench-medical",
      displayName: "GraphRAG-Bench Medical",
      track: "static-kb",
      metrics: [...RETRIEVAL_METRICS, ...RELIABILITY_METRICS],
    },
    {
      id: "graphrag-bench-novel",
      displayName: "GraphRAG-Bench Novel",
      track: "static-kb",
      metrics: [...RETRIEVAL_METRICS, ...RELIABILITY_METRICS],
    },
    {
      id: "beir-scifact",
      displayName: "BEIR SciFact",
      track: "static-kb",
      metrics: [...RETRIEVAL_METRICS, "latency-p95", "input-tokens", "cost"],
      requiredDataPath: "beir-scifact",
    },
    {
      id: "beir-nfcorpus",
      displayName: "BEIR NFCorpus",
      track: "static-kb",
      metrics: [...RETRIEVAL_METRICS, "latency-p95", "input-tokens", "cost"],
      requiredDataPath: "beir-nfcorpus",
    },
    {
      id: "garage",
      displayName: "GaRAGe",
      track: "static-kb",
      metrics: [...RELIABILITY_METRICS, "acceptable-abstention", "answerability-joint-accuracy"],
      requiredDataPath: "garage",
    },
    {
      id: "frames",
      displayName: "FRAMES",
      track: "static-kb",
      metrics: [...RETRIEVAL_METRICS, ...RELIABILITY_METRICS],
      requiredDataPath: "frames",
    },
    {
      id: "uaeval-kontext",
      displayName: "UAEval4RAG-style kontext corpus boundary",
      track: "static-kb",
      metrics: [...ANSWER_QUALITY_METRICS, "acceptable-abstention", "answerability-joint-accuracy"],
      requiredDataPath: "uaeval-kontext",
    },
    {
      id: "stable-rag",
      displayName: "Stable-RAG perturbations",
      track: "static-kb",
      metrics: [...ANSWER_QUALITY_METRICS, "permutation-sensitivity", "robustness-drop"],
      requiredDataPath: "stable-rag",
    },
    {
      id: "crag",
      displayName: "CRAG",
      track: "dynamic-api",
      metrics: [
        "answer-correctness",
        "acceptable-abstention",
        "answerability-joint-accuracy",
        "crag-truthfulness",
        "latency-p95",
      ],
      requiredDataPath: "crag",
    },
    {
      id: "trec-rag",
      displayName: "TREC RAG",
      track: "large-corpus",
      metrics: [
        ...RETRIEVAL_METRICS,
        "answer-correctness",
        "citation-precision",
        "citation-recall",
        "citation-f1",
        "latency-p95",
        "cost",
      ],
      requiredDataPath: "trec-rag",
    },
    {
      id: "ragtime",
      displayName: "TREC RAGTIME",
      track: "multilingual-report",
      metrics: [...ANSWER_QUALITY_METRICS, "latency-p95", "cost"],
      requiredDataPath: "ragtime",
    },
  ],
};

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, nested]) => [key, stableValue(nested)]),
    );
  }
  return value;
}

export function manifestDigest(manifest: RagEvalManifest): string {
  return createHash("sha256")
    .update(JSON.stringify(stableValue(manifest)))
    .digest("hex");
}

export function loadFrozenRunManifest(path: string): RagEvalManifest {
  let value: unknown;
  try {
    value = JSON.parse(readFileSync(path, "utf8")) as unknown;
  } catch (error) {
    throw new Error(`Cannot read frozen run manifest envelope at ${path}`, { cause: error });
  }
  if (!isExactFrozenRunManifestEnvelope(value)) {
    throw new Error(`Malformed frozen run manifest envelope at ${path}`);
  }

  const manifest = value.manifest as RagEvalManifest;
  try {
    assertValidManifest(manifest);
  } catch (error) {
    throw new Error(`Invalid frozen run manifest at ${path}`, { cause: error });
  }
  const actualDigest = manifestDigest(manifest);
  if (value.manifestDigest !== actualDigest) {
    throw new Error(
      `Frozen run manifest digest mismatch at ${path}: stored ${value.manifestDigest}, canonical ${actualDigest}`,
    );
  }
  return manifest;
}

export function manifestForRunDirectory(
  defaultManifest: RagEvalManifest,
  workDirectory: string,
): RagEvalManifest {
  const frozenManifestPath = join(workDirectory, "run-manifest.json");
  return existsSync(frozenManifestPath)
    ? loadFrozenRunManifest(frozenManifestPath)
    : defaultManifest;
}

function isExactFrozenRunManifestEnvelope(
  value: unknown,
): value is { readonly manifestDigest: string; readonly manifest: object } {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const record = value as Record<string, unknown>;
  const keys = Object.keys(record).sort();
  return (
    keys.length === 2 &&
    keys[0] === "manifest" &&
    keys[1] === "manifestDigest" &&
    typeof record.manifestDigest === "string" &&
    /^[0-9a-f]{64}$/.test(record.manifestDigest) &&
    !!record.manifest &&
    typeof record.manifest === "object" &&
    !Array.isArray(record.manifest)
  );
}

export function assertValidManifest(manifest: RagEvalManifest): void {
  if (manifest.schemaVersion !== RAG_EVAL_SCHEMA_VERSION) {
    throw new Error(`Unsupported manifest schema ${manifest.schemaVersion}`);
  }
  if (manifest.benchmarkPolicy.aggregateAcrossDatasets !== false) {
    throw new Error("Cross-dataset aggregation is forbidden by the benchmark protocol");
  }
  if (manifest.benchmarkPolicy.retrievalQueryScope !== "all") {
    throw new Error("Retrieval must run over every dataset query");
  }
  if (
    !Number.isInteger(manifest.benchmarkPolicy.answerJudgeSamplePerDataset) ||
    manifest.benchmarkPolicy.answerJudgeSamplePerDataset <= 0
  ) {
    throw new Error("answerJudgeSamplePerDataset must be a positive integer");
  }
  if (!Number.isInteger(manifest.benchmarkPolicy.answerJudgeSampleSeed)) {
    throw new Error("answerJudgeSampleSeed must be an integer");
  }
  if (
    !Number.isInteger(manifest.benchmarkPolicy.answerCodexBatchSize) ||
    manifest.benchmarkPolicy.answerCodexBatchSize <= 0
  ) {
    throw new Error("answerCodexBatchSize must be a positive integer");
  }
  if (
    !Number.isInteger(manifest.benchmarkPolicy.judgeCodexBatchSize) ||
    manifest.benchmarkPolicy.judgeCodexBatchSize <= 0
  ) {
    throw new Error("judgeCodexBatchSize must be a positive integer");
  }
  if (
    !Number.isInteger(manifest.benchmarkPolicy.codexConcurrency) ||
    manifest.benchmarkPolicy.codexConcurrency <= 0
  ) {
    throw new Error("codexConcurrency must be a positive integer");
  }
  if (
    manifest.benchmarkPolicy.judgeCodexConcurrency !== undefined &&
    (!Number.isInteger(manifest.benchmarkPolicy.judgeCodexConcurrency) ||
      manifest.benchmarkPolicy.judgeCodexConcurrency <= 0)
  ) {
    throw new Error("judgeCodexConcurrency must be a positive integer when set");
  }
  if (
    manifest.benchmarkPolicy.judgeTimeoutMs !== undefined &&
    (!Number.isInteger(manifest.benchmarkPolicy.judgeTimeoutMs) ||
      manifest.benchmarkPolicy.judgeTimeoutMs <= 0)
  ) {
    throw new Error("judgeTimeoutMs must be a positive integer when set");
  }
  if (manifest.models.embedding.provider !== "openai") {
    throw new Error("The shared embedding provider must be OpenAI");
  }
  if (manifest.models.embedding.model !== "text-embedding-3-small") {
    throw new Error("The shared embedding model must be text-embedding-3-small");
  }
  if (manifest.models.embedding.dimensions !== 1536) {
    throw new Error("The shared embedding dimensionality must be 1536");
  }
  const frameworkIds = new Set(manifest.frameworks.map((framework) => framework.id));
  if (frameworkIds.size !== manifest.frameworks.length) throw new Error("Duplicate framework id");
  for (const framework of manifest.frameworks) {
    if (framework.versionPolicy === "official-pinned" && !framework.pinnedVersion) {
      throw new Error(`Official framework ${framework.id} must have a pinned version`);
    }
  }
  const datasetIds = new Set(manifest.datasets.map((dataset) => dataset.id));
  if (datasetIds.size !== manifest.datasets.length) throw new Error("Duplicate dataset id");
}
