import { createHash } from "node:crypto";
import { existsSync, lstatSync, readFileSync, readdirSync, realpathSync } from "node:fs";
import { arch, cpus, loadavg, platform, release, totalmem } from "node:os";
import { join, relative, resolve } from "node:path";
import { runCommand } from "./codex-json.js";
import type {
  AnswerResult,
  DatasetBundle,
  DatasetId,
  FrameworkId,
  JudgeResult,
  RetrievalResult,
} from "./contracts.js";
import { defaultDatasetPaths, loadDataset } from "./datasets.js";
import { createEvaluationSample } from "./evaluation-sample.js";
import {
  ExternalCommandFrameworkAdapter,
  type FrameworkAdapter,
  type FrameworkRunOptions,
} from "./frameworks.js";
import { readJsonLines, writeJsonAtomic, writeJsonLines } from "./jsonl.js";
import { KontextBrainAdapter, type KontextRetrievalMode } from "./kontext-framework.js";
import { type JsonLlmClient, createJsonLlmClient } from "./llm-json-client.js";
import { DEFAULT_RAG_EVAL_MANIFEST, type RagEvalManifest, manifestDigest } from "./manifest.js";
import { nearestRankPercentileOrNull } from "./metrics.js";
import { answerQueries, judgeAnswers } from "./pipeline.js";

export const CLEAN_LATENCY_PROTOCOL_VERSION = "clean-latency-v1" as const;
export const CLEAN_LATENCY_SAMPLE_SIZE = 200;
export const CLEAN_LATENCY_QUERY_CONCURRENCY = 1;
export const CLEAN_LATENCY_TAIL_LIMIT_MS = 600_000;
export const CLEAN_LATENCY_ANTHROPIC_ANSWER_MODEL = "claude-sonnet-5" as const;

export type CleanLatencyCompletionBackend = "codex-exec" | "anthropic-api";

export type CleanLatencySystem =
  | "kontext-v15"
  | "kontext-v13"
  | "lightrag-1.5.6"
  | "microsoft-graphrag-3.1.1";

export interface CleanLatencyRunOptions {
  readonly repositoryRoot: string;
  readonly workDirectory: string;
  readonly datasetId: Extract<DatasetId, "graphrag-bench-medical" | "graphrag-bench-novel">;
  readonly system: CleanLatencySystem;
  readonly indexSourceDirectory: string;
  readonly skipHostGuard?: boolean;
  readonly completionBackend?: CleanLatencyCompletionBackend;
}

export interface LatencyStageSummary {
  readonly completed: number;
  readonly p50Ms: number | null;
  readonly p95Ms: number | null;
  readonly maxMs: number | null;
  readonly over600Seconds: number;
}

export interface CleanLatencySummary {
  readonly queries: number;
  readonly retrieval: LatencyStageSummary;
  readonly answer: LatencyStageSummary;
  readonly judge: LatencyStageSummary;
  readonly queryToAnswer: LatencyStageSummary;
  readonly judgeInclusiveEvaluationEndToEnd: LatencyStageSummary;
}

export interface CleanLatencyAssessment {
  readonly status: "valid" | "invalid";
  readonly reasons: readonly string[];
  readonly suspiciousCompletionWaveGaps: readonly {
    readonly stage: "retrieval" | "answer" | "judge";
    readonly gapMs: number;
    readonly previousCompletedAt: string;
    readonly completedAt: string;
  }[];
}

export interface CleanLatencyReport {
  readonly schemaVersion: "1.0.0";
  readonly protocolVersion: typeof CLEAN_LATENCY_PROTOCOL_VERSION;
  readonly datasetId: CleanLatencyRunOptions["datasetId"];
  readonly system: CleanLatencySystem;
  readonly frameworkId: FrameworkId;
  readonly startedAt: string;
  readonly completedAt: string;
  readonly sampleDigest: string;
  readonly sampleQueryIds: readonly string[];
  readonly manifestDigest: string;
  readonly conditions: {
    readonly warmIndex: true;
    readonly indexBuildIncluded: false;
    readonly newIndexEmbeddingsAllowed: false;
    readonly retrievalQueryConcurrency: 1;
    readonly answerConcurrency: 1;
    readonly judgeConcurrency: 1;
    readonly answerBatchSize: 1;
    readonly judgeBatchSize: 1;
    readonly maxRetries: 0;
    readonly percentile: "nearest-rank";
    readonly queryToAnswerDefinition: "retrieval latency + answer latency; judge excluded";
    readonly evaluationEndToEndDefinition: "retrieval latency + answer latency + judge latency";
    readonly completionBackend: CleanLatencyCompletionBackend;
    readonly answerModel: string;
    readonly judgeModel: string;
    readonly answerModelMatchesIndexBuild: boolean;
  };
  readonly indexSource: {
    readonly path: string;
    readonly runManifestPath: string;
    readonly runManifestDigest: string;
    readonly metadataDigestBefore: string;
    readonly metadataDigestAfter: string;
    readonly unchanged: boolean;
    readonly indexBuildModels: IndexBuildModels | null;
  };
  readonly environment: Awaited<ReturnType<typeof environmentProvenance>>;
  readonly summary: CleanLatencySummary;
  readonly assessment: CleanLatencyAssessment;
}

export async function runCleanLatency(
  options: CleanLatencyRunOptions,
): Promise<CleanLatencyReport> {
  if (!options.skipHostGuard) await assertNoCompetingBenchmarkProcesses();
  const reportPath = join(options.workDirectory, "clean-latency-report.json");
  if (existsSync(reportPath))
    throw new Error(
      `Clean latency report already exists; preserve it and use a new path: ${reportPath}`,
    );
  const backend = options.completionBackend ?? "codex-exec";
  const indexSourceDirectory = realpathSync(resolve(options.indexSourceDirectory));
  const metadataDigestBefore = directoryMetadataDigest(indexSourceDirectory);
  const manifest = cleanLatencyManifest(backend);
  const sourceManifest = validateIndexSourceProvenance(indexSourceDirectory, manifest, {
    requireAnswerModelMatch: backend === "codex-exec",
  });
  const fullBundle = loadDataset(
    options.datasetId,
    defaultDatasetPaths(resolve(options.repositoryRoot)),
  );
  const sample = createEvaluationSample(
    fullBundle,
    CLEAN_LATENCY_SAMPLE_SIZE,
    manifest.benchmarkPolicy.answerJudgeSampleSeed,
  );
  if (sample.queries.length !== CLEAN_LATENCY_SAMPLE_SIZE)
    throw new Error(
      `Clean latency requires exactly ${CLEAN_LATENCY_SAMPLE_SIZE} queries, found ${sample.queries.length}`,
    );
  const sampledBundle: DatasetBundle = { ...fullBundle, queries: sample.queries };
  const answerClient = createJsonLlmClient(
    manifest.models.answer.execution === "anthropic-api" ? "anthropic-api" : "codex-exec",
  );
  const judgeClient = createJsonLlmClient(
    manifest.models.judge.execution === "anthropic-api" ? "anthropic-api" : "codex-exec",
  );
  const { adapter, frameworkId } = createCleanAdapter(
    options.system,
    manifest,
    indexSourceDirectory,
    answerClient,
  );
  const doctor = await adapter.doctor();
  if (doctor.status !== "ready")
    throw new Error(`${options.system} doctor ${doctor.status}: ${doctor.detail}`);
  const startedAt = new Date().toISOString();
  const environment = await environmentProvenance(options.repositoryRoot, options.skipHostGuard);
  const protocol = {
    schemaVersion: "1.0.0",
    protocolVersion: CLEAN_LATENCY_PROTOCOL_VERSION,
    datasetId: options.datasetId,
    system: options.system,
    frameworkId,
    sample: sample.manifest,
    manifestDigest: manifestDigest(manifest),
    conditions: frozenConditions(backend, manifest),
    indexSource: {
      path: indexSourceDirectory,
      runManifestPath: sourceManifest.path,
      runManifestDigest: sourceManifest.digest,
      metadataDigestBefore,
      indexBuildModels: sourceManifest.indexBuildModels,
    },
    environment,
  };
  const frozenProtocol = freezeProtocol(
    join(options.workDirectory, "clean-latency-protocol.json"),
    protocol,
  );
  writeJsonAtomic(join(options.workDirectory, "evaluation-sample.json"), sample.manifest);

  const frameworkDirectory = join(options.workDirectory, sampledBundle.id, frameworkId);
  const retrievalPath = join(frameworkDirectory, "retrieval.jsonl");
  const retrievals =
    loadCompletedStage<RetrievalResult>(retrievalPath, sample.manifest.queryIds, frameworkId) ??
    (await adapter.retrieve(sampledBundle, {
      workDirectory: options.workDirectory,
      topK: 10,
      candidateK: 50,
      queryConcurrency: CLEAN_LATENCY_QUERY_CONCURRENCY,
      indexSourceDirectory,
      requireWarmIndex: true,
      indexQueryUniverse: fullBundle.queries,
    } satisfies FrameworkRunOptions));
  writeJsonLines(retrievalPath, retrievals);

  const answers = await answerQueries(
    manifest,
    sampledBundle,
    frameworkId,
    retrievals,
    sample.queries,
    frameworkDirectory,
    answerClient,
  );
  const judgements = await judgeAnswers(
    manifest,
    sampledBundle,
    frameworkId,
    retrievals,
    answers,
    sample.queries,
    frameworkDirectory,
    judgeClient,
  );
  const metadataDigestAfter = directoryMetadataDigest(indexSourceDirectory);
  const summary = summarizeCleanLatency(sample.manifest.queryIds, retrievals, answers, judgements);
  const assessment = assessCleanLatency(
    sample.manifest.queryIds,
    retrievals,
    answers,
    judgements,
    metadataDigestBefore === metadataDigestAfter,
  );
  const report: CleanLatencyReport = {
    schemaVersion: "1.0.0",
    protocolVersion: CLEAN_LATENCY_PROTOCOL_VERSION,
    datasetId: options.datasetId,
    system: options.system,
    frameworkId,
    startedAt,
    completedAt: new Date().toISOString(),
    sampleDigest: sample.manifest.sampleDigest,
    sampleQueryIds: sample.manifest.queryIds,
    manifestDigest: manifestDigest(manifest),
    conditions: frozenConditions(backend, manifest),
    indexSource: {
      path: indexSourceDirectory,
      runManifestPath: sourceManifest.path,
      runManifestDigest: sourceManifest.digest,
      metadataDigestBefore,
      metadataDigestAfter,
      unchanged: metadataDigestBefore === metadataDigestAfter,
      indexBuildModels: sourceManifest.indexBuildModels,
    },
    environment: frozenProtocol.environment,
    summary,
    assessment,
  };
  writeJsonAtomic(reportPath, report);
  return report;
}

export function cleanLatencyManifest(
  backend: CleanLatencyCompletionBackend = "codex-exec",
): RagEvalManifest {
  const base = DEFAULT_RAG_EVAL_MANIFEST;
  return {
    ...base,
    benchmarkPolicy: {
      ...base.benchmarkPolicy,
      answerJudgeSamplePerDataset: CLEAN_LATENCY_SAMPLE_SIZE,
      answerCodexBatchSize: 1,
      judgeCodexBatchSize: 1,
      codexConcurrency: 1,
      maxRetries: 0,
    },
    models:
      backend === "anthropic-api"
        ? {
            ...base.models,
            answer: {
              provider: "anthropic",
              model: CLEAN_LATENCY_ANTHROPIC_ANSWER_MODEL,
              reasoningEffort: "medium",
              execution: "anthropic-api",
            },
          }
        : base.models,
  };
}

export function summarizeCleanLatency(
  queryIds: readonly string[],
  retrievals: readonly RetrievalResult[],
  answers: readonly AnswerResult[],
  judgements: readonly JudgeResult[],
): CleanLatencySummary {
  const retrievalById = new Map(retrievals.map((result) => [result.queryId, result]));
  const answerById = new Map(answers.map((result) => [result.queryId, result]));
  const judgeById = new Map(judgements.map((result) => [result.queryId, result]));
  const retrievalLatencies = successfulLatencies(retrievals);
  const answerLatencies = successfulLatencies(answers);
  const judgeLatencies = successfulLatencies(judgements);
  const queryToAnswerLatencies = queryIds.flatMap((queryId) => {
    const retrieval = retrievalById.get(queryId);
    const answer = answerById.get(queryId);
    return retrieval?.status === "ok" && answer?.status === "ok"
      ? [retrieval.latencyMs + answer.latencyMs]
      : [];
  });
  const evaluationEndToEndLatencies = queryIds.flatMap((queryId) => {
    const retrieval = retrievalById.get(queryId);
    const answer = answerById.get(queryId);
    const judge = judgeById.get(queryId);
    return retrieval?.status === "ok" && answer?.status === "ok" && judge?.status === "ok"
      ? [retrieval.latencyMs + answer.latencyMs + judge.latencyMs]
      : [];
  });
  return {
    queries: queryIds.length,
    retrieval: stageSummary(retrievalLatencies),
    answer: stageSummary(answerLatencies),
    judge: stageSummary(judgeLatencies),
    queryToAnswer: stageSummary(queryToAnswerLatencies),
    judgeInclusiveEvaluationEndToEnd: stageSummary(evaluationEndToEndLatencies),
  };
}

export function assessCleanLatency(
  queryIds: readonly string[],
  retrievals: readonly RetrievalResult[],
  answers: readonly AnswerResult[],
  judgements: readonly JudgeResult[],
  indexSourceUnchanged: boolean,
): CleanLatencyAssessment {
  const reasons: string[] = [];
  const expectedIds = new Set(queryIds);
  for (const [stage, records] of [
    ["retrieval", retrievals],
    ["answer", answers],
    ["judge", judgements],
  ] as const) {
    const uniqueIds = new Set(records.map((record) => record.queryId));
    const completed = records.filter((record) => record.status === "ok").length;
    if (
      records.length !== queryIds.length ||
      uniqueIds.size !== queryIds.length ||
      [...uniqueIds].some((queryId) => !expectedIds.has(queryId))
    )
      reasons.push(`${stage} identity coverage ${uniqueIds.size}/${queryIds.length}`);
    if (completed !== queryIds.length)
      reasons.push(`${stage} completed ${completed}/${queryIds.length}`);
    const tails = records.filter(
      (record) => record.status === "ok" && record.latencyMs > CLEAN_LATENCY_TAIL_LIMIT_MS,
    ).length;
    if (tails > 0) reasons.push(`${stage} has ${tails} latency values over 600 seconds`);
    const invalidLatencies = records.filter(
      (record) =>
        record.status === "ok" && (!Number.isFinite(record.latencyMs) || record.latencyMs < 0),
    ).length;
    if (invalidLatencies > 0)
      reasons.push(`${stage} has ${invalidLatencies} invalid latency values`);
    const queueErrors = records.filter((record) =>
      /usage.?limit|rate.?limit|quota|queue|throttl|retry.?after/i.test(record.error ?? ""),
    );
    if (queueErrors.length > 0)
      reasons.push(
        `${stage} has ${queueErrors.length} queue, throttle, quota, or usage-limit errors`,
      );
  }
  if (!indexSourceUnchanged) reasons.push("read-only index source metadata changed during the run");
  const suspiciousCompletionWaveGaps = [
    ...completionWaveGaps("retrieval", retrievals),
    ...completionWaveGaps("answer", answers),
    ...completionWaveGaps("judge", judgements),
  ];
  if (suspiciousCompletionWaveGaps.length > 0)
    reasons.push(
      `${suspiciousCompletionWaveGaps.length} inter-completion gaps fell in the 10-20 minute throttle-wave band`,
    );
  return {
    status: reasons.length === 0 ? "valid" : "invalid",
    reasons,
    suspiciousCompletionWaveGaps,
  };
}

export function directoryMetadataDigest(directory: string): string {
  const root = realpathSync(directory);
  const hash = createHash("sha256");
  const visit = (path: string): void => {
    const stats = lstatSync(path, { bigint: true });
    const name = relative(root, path) || ".";
    hash
      .update(name)
      .update("\0")
      .update(stats.mode.toString())
      .update("\0")
      .update(stats.size.toString())
      .update("\0")
      .update(stats.mtimeNs.toString())
      .update("\0");
    if (stats.isDirectory()) {
      for (const entry of readdirSync(path).sort()) visit(join(path, entry));
    }
  };
  visit(root);
  return hash.digest("hex");
}

export interface IndexBuildModels {
  readonly embedding: Readonly<Record<string, unknown>> | null;
  readonly answer: Readonly<Record<string, unknown>> | null;
}

export function validateIndexSourceProvenance(
  indexSourceDirectory: string,
  manifest: RagEvalManifest,
  options: { readonly requireAnswerModelMatch: boolean } = { requireAnswerModelMatch: true },
): { readonly path: string; readonly digest: string; readonly indexBuildModels: IndexBuildModels } {
  let cursor = resolve(indexSourceDirectory);
  let manifestPath: string | null = null;
  for (let depth = 0; depth <= 6; depth += 1) {
    const candidate = join(cursor, "run-manifest.json");
    if (existsSync(candidate)) {
      manifestPath = candidate;
      break;
    }
    const parent = resolve(cursor, "..");
    if (parent === cursor) break;
    cursor = parent;
  }
  if (!manifestPath)
    throw new Error(`Warm index source has no ancestor run-manifest.json: ${indexSourceDirectory}`);
  const bytes = readFileSync(manifestPath);
  const source = JSON.parse(bytes.toString("utf8")) as {
    readonly manifest?: Pick<RagEvalManifest, "models">;
  };
  const actualModels: IndexBuildModels | null = source.manifest
    ? {
        embedding: comparableModel(source.manifest.models.embedding),
        answer: comparableModel(source.manifest.models.answer),
      }
    : null;
  const expectedModels = {
    embedding: comparableModel(manifest.models.embedding),
    answer: options.requireAnswerModelMatch ? comparableModel(manifest.models.answer) : null,
  };
  const foundModels = actualModels
    ? {
        embedding: actualModels.embedding,
        answer: options.requireAnswerModelMatch ? actualModels.answer : null,
      }
    : null;
  if (JSON.stringify(foundModels) !== JSON.stringify(expectedModels))
    throw new Error(
      `Warm index source model provenance mismatch at ${manifestPath}: expected ${JSON.stringify(expectedModels)}, found ${JSON.stringify(foundModels)}`,
    );
  return {
    path: realpathSync(manifestPath),
    digest: createHash("sha256").update(bytes).digest("hex"),
    indexBuildModels: actualModels ?? { embedding: null, answer: null },
  };
}

function comparableModel(model: unknown): Readonly<Record<string, unknown>> | null {
  if (!model || typeof model !== "object") return null;
  const value = model as Readonly<Record<string, unknown>>;
  return {
    provider: value.provider ?? null,
    model: value.model ?? null,
    dimensions: value.dimensions ?? null,
    reasoningEffort: value.reasoningEffort ?? null,
    execution: value.execution ?? null,
  };
}

export async function assertNoCompetingBenchmarkProcesses(): Promise<void> {
  const result = await runCommand("ps", ["-axo", "pid=,command="], "", 10_000);
  if (result.exitCode !== 0)
    throw new Error(`Unable to verify clean host process state: ${result.stderr.trim()}`);
  const conflicts = result.stdout
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !line.startsWith(`${process.pid} `))
    .filter((line) =>
      /rag-eval-v2\/cli\.ts|framework-adapters\/(?:graphrag|lightrag)\/adapter\.py|rag-eval-(?:graphrag|lightrag|kontext)/i.test(
        line,
      ),
    );
  if (conflicts.length > 0)
    throw new Error(`Competing benchmark processes detected:\n${conflicts.join("\n")}`);
}

function createCleanAdapter(
  system: CleanLatencySystem,
  manifest: RagEvalManifest,
  indexSourceDirectory: string,
  answerClient: JsonLlmClient,
): { readonly adapter: FrameworkAdapter; readonly frameworkId: FrameworkId } {
  const kontextModeBySystem: Partial<Record<CleanLatencySystem, KontextRetrievalMode>> = {
    "kontext-v13": "multi-query-anchored-evidence-answer-stack",
    "kontext-v15": "corpus-complete-anchored-evidence-answer-stack",
  };
  const kontextMode = kontextModeBySystem[system];
  if (kontextMode) {
    return {
      adapter: new KontextBrainAdapter(manifest, {
        codexClient: answerClient,
        embeddingClient: null,
        retrievalMode: kontextMode,
        precomputedIndexDirectory: indexSourceDirectory,
        strictWarmIndex: true,
      }),
      frameworkId: "kontext-brain",
    };
  }
  const frameworkId = system === "lightrag-1.5.6" ? "lightrag" : "microsoft-graphrag";
  const framework = manifest.frameworks.find((candidate) => candidate.id === frameworkId);
  if (!framework) throw new Error(`Framework manifest missing for ${frameworkId}`);
  return {
    adapter: new ExternalCommandFrameworkAdapter(framework, manifest),
    frameworkId,
  };
}

export function frozenConditions(
  backend: CleanLatencyCompletionBackend,
  manifest: RagEvalManifest,
): CleanLatencyReport["conditions"] {
  return {
    warmIndex: true,
    indexBuildIncluded: false,
    newIndexEmbeddingsAllowed: false,
    retrievalQueryConcurrency: 1,
    answerConcurrency: 1,
    judgeConcurrency: 1,
    answerBatchSize: 1,
    judgeBatchSize: 1,
    maxRetries: 0,
    percentile: "nearest-rank",
    queryToAnswerDefinition: "retrieval latency + answer latency; judge excluded",
    evaluationEndToEndDefinition: "retrieval latency + answer latency + judge latency",
    completionBackend: backend,
    answerModel: manifest.models.answer.model,
    judgeModel: manifest.models.judge.model,
    answerModelMatchesIndexBuild: backend === "codex-exec",
  };
}

function freezeProtocol<T extends { readonly environment: unknown }>(path: string, protocol: T): T {
  if (existsSync(path)) {
    const existing = JSON.parse(readFileSync(path, "utf8")) as T;
    const { environment: _existingEnvironment, ...existingContract } = existing;
    const { environment: _currentEnvironment, ...currentContract } = protocol;
    if (JSON.stringify(existingContract) !== JSON.stringify(currentContract))
      throw new Error(`Clean latency protocol mismatch at ${path}; use a new run directory`);
    return existing;
  }
  writeJsonAtomic(path, protocol);
  return protocol;
}

function loadCompletedStage<T extends { readonly queryId: string; readonly status: string }>(
  path: string,
  queryIds: readonly string[],
  frameworkId: FrameworkId,
): T[] | null {
  if (!existsSync(path)) return null;
  const records = readJsonLines<T & { readonly frameworkId?: FrameworkId }>(path);
  if (
    records.length !== queryIds.length ||
    records.some(
      (record, index) =>
        record.queryId !== queryIds[index] ||
        record.status !== "ok" ||
        record.frameworkId !== frameworkId,
    )
  )
    return null;
  return records;
}

function successfulLatencies(
  records: readonly { readonly status: string; readonly latencyMs: number }[],
): number[] {
  return records.filter((record) => record.status === "ok").map((record) => record.latencyMs);
}

function stageSummary(latencies: readonly number[]): LatencyStageSummary {
  return {
    completed: latencies.length,
    p50Ms: nearestRankPercentileOrNull(latencies, 0.5),
    p95Ms: nearestRankPercentileOrNull(latencies, 0.95),
    maxMs: latencies.length === 0 ? null : Math.max(...latencies),
    over600Seconds: latencies.filter((latency) => latency > CLEAN_LATENCY_TAIL_LIMIT_MS).length,
  };
}

function completionWaveGaps(
  stage: "retrieval" | "answer" | "judge",
  records: readonly { readonly completedAt?: string }[],
): CleanLatencyAssessment["suspiciousCompletionWaveGaps"] {
  const completed = records
    .flatMap((record) => {
      if (!record.completedAt) return [];
      const value = Date.parse(record.completedAt);
      return Number.isFinite(value) ? [{ text: record.completedAt, value }] : [];
    })
    .sort((left, right) => left.value - right.value);
  const gaps: Array<CleanLatencyAssessment["suspiciousCompletionWaveGaps"][number]> = [];
  for (let index = 1; index < completed.length; index += 1) {
    const previous = completed[index - 1];
    const current = completed[index];
    if (!previous || !current) continue;
    const gapMs = current.value - previous.value;
    if (gapMs >= 10 * 60_000 && gapMs <= 20 * 60_000) {
      gaps.push({
        stage,
        gapMs,
        previousCompletedAt: previous.text,
        completedAt: current.text,
      });
    }
  }
  return gaps;
}

async function environmentProvenance(
  repositoryRoot: string,
  hostGuardSkipped = false,
): Promise<{
  readonly capturedAt: string;
  readonly gitCommit: string;
  readonly node: string;
  readonly platform: string;
  readonly release: string;
  readonly arch: string;
  readonly cpuModel: string;
  readonly logicalCpus: number;
  readonly totalMemoryBytes: number;
  readonly loadAverage: readonly number[];
  readonly hostGuard: "automatic" | "operator-attested";
}> {
  const git = await runCommand(
    "git",
    ["-C", resolve(repositoryRoot), "rev-parse", "HEAD"],
    "",
    10_000,
  );
  if (git.exitCode !== 0) throw new Error(`Unable to capture git commit: ${git.stderr.trim()}`);
  const processors = cpus();
  return {
    capturedAt: new Date().toISOString(),
    gitCommit: git.stdout.trim(),
    node: process.version,
    platform: platform(),
    release: release(),
    arch: arch(),
    cpuModel: processors[0]?.model ?? "unknown",
    logicalCpus: processors.length,
    totalMemoryBytes: totalmem(),
    loadAverage: loadavg(),
    hostGuard: hostGuardSkipped ? "operator-attested" : "automatic",
  };
}
