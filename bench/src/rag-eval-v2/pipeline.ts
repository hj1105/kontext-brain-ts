import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { CodexJsonClient, runCommand } from "./codex-json.js";
import type {
  AnswerContract,
  AnswerResult,
  BenchmarkQuery,
  DatasetDoctorResult,
  DatasetId,
  FrameworkDoctorResult,
  FrameworkId,
  JudgeResult,
  RetrievalResult,
} from "./contracts.js";
import {
  type DatasetLoadOptions,
  type DatasetPaths,
  doctorDatasets,
  loadDataset,
} from "./datasets.js";
import { type EvaluationSampleManifest, createEvaluationSample } from "./evaluation-sample.js";
import { type FrameworkAdapter, createFrameworkAdapters } from "./frameworks.js";
import { createBlindHumanAuditSample } from "./human-audit.js";
import { readJsonLines, writeJsonAtomic, writeJsonLines } from "./jsonl.js";
import type { JsonLlmClient } from "./llm-json-client.js";
import {
  type RagEvalManifest,
  assertValidManifest,
  loadFrozenRunManifest,
  manifestDigest,
} from "./manifest.js";
import type { DatasetFrameworkScore, PairedFrameworkComparison } from "./metrics.js";
import { compareFrameworkPairs, scoreDatasetFramework } from "./metrics.js";

export interface ModelDoctorResult {
  readonly component: "embedding" | "answer" | "judge";
  readonly status: "ready" | "blocked";
  readonly detail: string;
}

export interface DoctorReport {
  readonly manifestDigest: string;
  readonly models: readonly ModelDoctorResult[];
  readonly datasets: readonly DatasetDoctorResult[];
  readonly frameworks: readonly FrameworkDoctorResult[];
}

export interface BenchmarkRunOptions {
  readonly workDirectory: string;
  readonly datasetPaths: DatasetPaths;
  readonly stage?: "retrieval" | "full";
  readonly datasetIds?: readonly DatasetId[];
  readonly frameworkIds?: readonly FrameworkId[];
  readonly datasetLoad?: DatasetLoadOptions;
  readonly topK?: number;
  readonly candidateK?: number;
}

export interface DatasetRunReport {
  readonly datasetId: DatasetId;
  readonly status: "completed" | "partial" | "blocked";
  readonly detail: string;
  readonly scores: readonly DatasetFrameworkScore[];
  readonly pairedComparisons: readonly PairedFrameworkComparison[];
}

export interface BenchmarkRunReport {
  readonly manifestDigest: string;
  readonly startedAt: string;
  readonly completedAt: string;
  readonly workDirectory: string;
  readonly datasets: readonly DatasetRunReport[];
}

const EMPTY_ANSWER: AnswerContract = {
  answer: "",
  citations: [],
  abstained: true,
  abstentionReason: "benchmark stage unavailable",
};
const ANSWER_INPUT_DIGEST_VERSION = "answer-input-v2";
// v3 adds clarity, conciseness, and fluency to the frozen judge contract.
const JUDGE_INPUT_DIGEST_VERSION = "judge-input-v3";

export async function doctorBenchmark(
  manifest: RagEvalManifest,
  datasetPaths: DatasetPaths,
): Promise<DoctorReport> {
  assertValidManifest(manifest);
  const adapters = createFrameworkAdapters(manifest);
  const codex = await runCommand("codex", ["--version"], "", 10_000).catch((error: Error) => ({
    exitCode: 1,
    stdout: "",
    stderr: error.message,
    durationMs: 0,
  }));
  const embeddingKey = process.env.OPENAI_API_KEY;
  return {
    manifestDigest: manifestDigest(manifest),
    models: [
      {
        component: "embedding",
        status: embeddingKey ? "ready" : "blocked",
        detail: embeddingKey ? "OpenAI API key present" : "OPENAI_API_KEY is not set",
      },
      {
        component: "answer",
        status: codex.exitCode === 0 ? "ready" : "blocked",
        detail: codex.exitCode === 0 ? codex.stdout.trim() : codex.stderr.trim(),
      },
      {
        component: "judge",
        status: codex.exitCode === 0 ? "ready" : "blocked",
        detail: codex.exitCode === 0 ? codex.stdout.trim() : codex.stderr.trim(),
      },
    ],
    datasets: doctorDatasets(manifest, datasetPaths),
    frameworks: await Promise.all(
      adapters.map((adapter) => doctorFramework(adapter, manifest.benchmarkPolicy.maxRetries)),
    ),
  };
}

export async function runBenchmark(
  manifest: RagEvalManifest,
  options: BenchmarkRunOptions,
): Promise<BenchmarkRunReport> {
  assertValidManifest(manifest);
  freezeRunManifest(manifest, options.workDirectory);
  const startedAt = new Date().toISOString();
  const digest = manifestDigest(manifest);
  const datasetDoctors = new Map(
    doctorDatasets(manifest, options.datasetPaths).map((result) => [result.datasetId, result]),
  );
  const selectedDatasetIds = new Set(
    options.datasetIds ?? manifest.datasets.map((dataset) => dataset.id),
  );
  const selectedFrameworkIds = new Set(
    options.frameworkIds ?? manifest.frameworks.map((framework) => framework.id),
  );
  const retrievalOnly = options.stage === "retrieval";
  const adapters = createFrameworkAdapters(manifest).filter((adapter) =>
    selectedFrameworkIds.has(adapter.id),
  );
  logProgress(
    `run start datasets=${[...selectedDatasetIds].join(",")} frameworks=${adapters.map((adapter) => adapter.id).join(",")}`,
  );
  const adapterDoctors = new Map(
    (
      await Promise.all(
        adapters.map(async (adapter) => {
          logProgress(`doctor start framework=${adapter.id}`);
          const result = await doctorFramework(adapter, manifest.benchmarkPolicy.maxRetries);
          logProgress(
            `doctor complete framework=${adapter.id} status=${result.status} version=${result.version}`,
          );
          return result;
        }),
      )
    ).map((result) => [result.frameworkId, result]),
  );
  const codexClient = new CodexJsonClient();
  const reports: DatasetRunReport[] = [];

  for (const dataset of manifest.datasets) {
    if (!selectedDatasetIds.has(dataset.id)) continue;
    logProgress(`dataset start dataset=${dataset.id}`);
    const datasetDoctor = datasetDoctors.get(dataset.id);
    if (!datasetDoctor || datasetDoctor.status !== "ready") {
      reports.push({
        datasetId: dataset.id,
        status: "blocked",
        detail: datasetDoctor?.detail ?? "Dataset doctor result missing",
        scores: [],
        pairedComparisons: [],
      });
      continue;
    }

    const bundle = loadDataset(dataset.id, options.datasetPaths, options.datasetLoad);
    const datasetDirectory = join(options.workDirectory, dataset.id);
    const evaluationSample = createEvaluationSample(
      bundle,
      manifest.benchmarkPolicy.answerJudgeSamplePerDataset,
      manifest.benchmarkPolicy.answerJudgeSampleSeed,
    );
    freezeEvaluationSample(evaluationSample.manifest, datasetDirectory);
    const allRetrievals: RetrievalResult[] = [];
    const allAnswers: AnswerResult[] = [];
    const allJudgements: JudgeResult[] = [];
    const scores: DatasetFrameworkScore[] = [];

    for (const adapter of adapters) {
      logProgress(`framework start dataset=${dataset.id} framework=${adapter.id}`);
      const frameworkDirectory = join(datasetDirectory, adapter.id);
      const retrievalPath = join(frameworkDirectory, "retrieval.jsonl");
      const doctor = adapterDoctors.get(adapter.id);
      const retrievals =
        doctor?.status === "ready"
          ? await retrieveOrLoadCompleted(adapter, bundle, options, manifest, retrievalPath)
          : blockedRetrievals(
              bundle.id,
              adapter.id,
              bundle.queries.map((query) => query.id),
              doctor,
            );
      logProgress(
        `retrieval complete dataset=${dataset.id} framework=${adapter.id} ok=${retrievals.filter((result) => result.status === "ok").length}/${retrievals.length}`,
      );
      writeJsonLines(retrievalPath, retrievals);
      if (doctor?.status === "ready") {
        writeJsonAtomic(join(frameworkDirectory, "retrieval-cache.json"), {
          schemaVersion: "1.0.0",
          cacheDigest: retrievalCacheDigest(bundle, adapter.id, options, manifest),
          records: retrievals.length,
        });
      }
      allRetrievals.push(...retrievals);

      if (retrievalOnly) {
        logProgress(
          `framework complete dataset=${dataset.id} framework=${adapter.id} stage=retrieval`,
        );
        continue;
      }

      const answers = await answerQueries(
        manifest,
        bundle,
        adapter.id,
        retrievals,
        evaluationSample.queries,
        frameworkDirectory,
        codexClient,
      );
      logProgress(
        `answers complete dataset=${dataset.id} framework=${adapter.id} ok=${answers.filter((result) => result.status === "ok").length}/${answers.length}`,
      );
      allAnswers.push(...answers);
      const judgements = await judgeAnswers(
        manifest,
        bundle,
        adapter.id,
        retrievals,
        answers,
        evaluationSample.queries,
        frameworkDirectory,
        codexClient,
      );
      logProgress(
        `judgements complete dataset=${dataset.id} framework=${adapter.id} ok=${judgements.filter((result) => result.status === "ok").length}/${judgements.length}`,
      );
      allJudgements.push(...judgements);
      const score = scoreDatasetFramework(
        bundle,
        adapter.id,
        retrievals,
        answers,
        judgements,
        evaluationSample.queries,
      );
      scores.push(score);
      writeJsonAtomic(join(frameworkDirectory, "score.json"), score);
      logProgress(`framework complete dataset=${dataset.id} framework=${adapter.id}`);
    }

    if (retrievalOnly) {
      reports.push({
        datasetId: dataset.id,
        status: "partial",
        detail: `${bundle.documents.length} documents, ${bundle.queries.length} retrieval queries; retrieval stage complete`,
        scores: [],
        pairedComparisons: [],
      });
      logProgress(`dataset complete dataset=${dataset.id} status=partial stage=retrieval`);
      continue;
    }

    const audit = createBlindHumanAuditSample(
      bundle,
      allRetrievals,
      allAnswers,
      manifest.benchmarkPolicy.humanAuditPerDataset,
    );
    writeJsonLines(join(datasetDirectory, "human-audit.blind.jsonl"), audit.rows);
    writeJsonLines(join(datasetDirectory, "human-audit.mapping.private.jsonl"), audit.mapping);
    const hasCompleted = scores.some(
      (score) => score.retrievalCompleted > 0 || score.completed > 0,
    );
    const allCompleted =
      scores.length > 0 &&
      scores.every(
        (score) =>
          score.retrievalCompleted === score.retrievalQueries && score.completed === score.queries,
      );
    reports.push({
      datasetId: dataset.id,
      status: allCompleted ? "completed" : hasCompleted ? "partial" : "blocked",
      detail: `${bundle.documents.length} documents, ${bundle.queries.length} retrieval queries, ${evaluationSample.queries.length} answer/judge queries`,
      scores,
      pairedComparisons: compareFrameworkPairs(
        bundle,
        adapters.map((adapter) => adapter.id),
        allRetrievals,
        allJudgements,
      ),
    });
    logProgress(`dataset complete dataset=${dataset.id} status=${reports.at(-1)!.status}`);
  }

  const report: BenchmarkRunReport = {
    manifestDigest: digest,
    startedAt,
    completedAt: new Date().toISOString(),
    workDirectory: options.workDirectory,
    datasets: reports,
  };
  const reportName = retrievalOnly
    ? `run-report.retrieval.${[...selectedDatasetIds].sort().join("+")}.${[...selectedFrameworkIds].sort().join("+")}.json`
    : "run-report.json";
  writeJsonAtomic(join(options.workDirectory, reportName), report);
  return report;
}

export function freezeRunManifest(manifest: RagEvalManifest, workDirectory: string): void {
  const digest = manifestDigest(manifest);
  const manifestPath = join(workDirectory, "run-manifest.json");
  const reportPath = join(workDirectory, "run-report.json");
  if (existsSync(manifestPath)) {
    const existingDigest = manifestDigest(loadFrozenRunManifest(manifestPath));
    if (existingDigest !== digest) {
      throw new Error(
        `Run directory manifest mismatch: ${workDirectory} contains ${existingDigest}, current is ${digest}. Use a new run directory.`,
      );
    }
    return;
  }
  const existingDigest = existsSync(reportPath)
    ? (JSON.parse(readFileSync(reportPath, "utf8")) as { manifestDigest?: string }).manifestDigest
    : undefined;
  if (existingDigest && existingDigest !== digest) {
    throw new Error(
      `Run directory manifest mismatch: ${workDirectory} contains ${existingDigest}, current is ${digest}. Use a new run directory.`,
    );
  }
  writeJsonAtomic(manifestPath, { manifestDigest: digest, manifest });
}

export function freezeEvaluationSample(
  sample: EvaluationSampleManifest,
  datasetDirectory: string,
): void {
  const path = join(datasetDirectory, "evaluation-sample.json");
  if (existsSync(path)) {
    const existing = JSON.parse(readFileSync(path, "utf8")) as { sampleDigest?: string };
    if (existing.sampleDigest !== sample.sampleDigest) {
      throw new Error(
        `Evaluation sample mismatch in ${datasetDirectory}: expected ${sample.sampleDigest}, found ${existing.sampleDigest ?? "none"}. Use a new run directory.`,
      );
    }
  }
  writeJsonAtomic(path, sample);
}

async function retrieveOrRecordErrors(
  adapter: FrameworkAdapter,
  bundle: ReturnType<typeof loadDataset>,
  options: BenchmarkRunOptions,
  manifest: RagEvalManifest,
): Promise<RetrievalResult[]> {
  try {
    return await retry(
      () =>
        adapter.retrieve(bundle, {
          workDirectory: options.workDirectory,
          topK: options.topK ?? 10,
          candidateK: options.candidateK ?? 50,
        }),
      manifest.benchmarkPolicy.maxRetries,
    );
  } catch (error) {
    return bundle.queries.map((query) => ({
      datasetId: bundle.id,
      frameworkId: adapter.id,
      queryId: query.id,
      status: "error",
      evidence: [],
      latencyMs: 0,
      inputTokens: null,
      error: (error as Error).message,
      frameworkVersion: "unresolved",
      configDigest: manifestDigest(manifest),
    }));
  }
}

async function retrieveOrLoadCompleted(
  adapter: FrameworkAdapter,
  bundle: ReturnType<typeof loadDataset>,
  options: BenchmarkRunOptions,
  manifest: RagEvalManifest,
  retrievalPath: string,
): Promise<RetrievalResult[]> {
  const cached = loadCompletedRetrieval(adapter.id, bundle, options, manifest, retrievalPath);
  return cached ?? (await retrieveOrRecordErrors(adapter, bundle, options, manifest));
}

export function loadCompletedRetrieval(
  frameworkId: FrameworkId,
  bundle: ReturnType<typeof loadDataset>,
  options: BenchmarkRunOptions,
  manifest: RagEvalManifest,
  retrievalPath: string,
): RetrievalResult[] | null {
  if (!existsSync(retrievalPath)) return null;
  try {
    const records = readJsonLines<RetrievalResult>(retrievalPath);
    const expectedIds = bundle.queries.map((query) => query.id);
    if (
      records.length !== expectedIds.length ||
      records.some(
        (record, index) =>
          record.datasetId !== bundle.id ||
          record.frameworkId !== frameworkId ||
          record.queryId !== expectedIds[index] ||
          record.status === "error" ||
          record.status === "blocked" ||
          record.configDigest !== manifestDigest(manifest),
      )
    ) {
      return null;
    }
    const metadataPath = join(dirname(retrievalPath), "retrieval-cache.json");
    if (existsSync(metadataPath)) {
      const metadata = JSON.parse(readFileSync(metadataPath, "utf8")) as { cacheDigest?: string };
      if (metadata.cacheDigest !== retrievalCacheDigest(bundle, frameworkId, options, manifest))
        return null;
    }
    return records;
  } catch {
    return null;
  }
}

function retrievalCacheDigest(
  bundle: ReturnType<typeof loadDataset>,
  frameworkId: FrameworkId,
  options: BenchmarkRunOptions,
  manifest: RagEvalManifest,
): string {
  const hash = createHash("sha256");
  hash.update(manifestDigest(manifest)).update("\0");
  hash.update(frameworkId).update("\0");
  hash.update(String(options.topK ?? 10)).update("\0");
  hash.update(String(options.candidateK ?? 50)).update("\0");
  hash.update(bundle.provenance.version).update("\0");
  for (const document of bundle.documents) {
    hash.update(document.id).update("\0").update(document.text).update("\0");
  }
  for (const query of bundle.queries) {
    hash.update(query.id).update("\0").update(query.text).update("\0");
  }
  return hash.digest("hex");
}

function blockedRetrievals(
  datasetId: DatasetId,
  frameworkId: FrameworkId,
  queryIds: readonly string[],
  doctor: FrameworkDoctorResult | undefined,
): RetrievalResult[] {
  return queryIds.map((queryId) => ({
    datasetId,
    frameworkId,
    queryId,
    status: doctor?.status === "unsupported" ? "unsupported" : "blocked",
    evidence: [],
    latencyMs: 0,
    inputTokens: null,
    error: doctor?.detail ?? "Framework doctor result missing",
    frameworkVersion: doctor?.version ?? "unresolved",
    configDigest: "unavailable",
  }));
}

export async function answerQueries(
  manifest: RagEvalManifest,
  bundle: ReturnType<typeof loadDataset>,
  expectedFrameworkId: FrameworkId,
  retrievals: readonly RetrievalResult[],
  evaluationQueries: readonly BenchmarkQuery[],
  frameworkDirectory: string,
  codexClient: JsonLlmClient,
): Promise<AnswerResult[]> {
  const outputPath = join(frameworkDirectory, "answers.jsonl");
  const existing = existsSync(outputPath) ? readJsonLines<AnswerResult>(outputPath) : [];
  const byQuery = new Map(existing.map((result) => [result.queryId, result]));
  const retrievalById = new Map(retrievals.map((result) => [result.queryId, result]));
  const evaluationIds = evaluationQueries.map((query) => query.id);
  const inputDigestByQuery = new Map(
    evaluationQueries.map((query) => [
      query.id,
      answerInputDigest(manifest, query, retrievalById.get(query.id)),
    ]),
  );

  for (const query of evaluationQueries) {
    const inputDigest = requiredMapValue(inputDigestByQuery, query.id);
    const checkpoint = byQuery.get(query.id);
    const retrieval = retrievalById.get(query.id);
    const identityError = prerequisiteIdentityError(
      bundle.id,
      expectedFrameworkId,
      query.id,
      retrieval,
    );
    if (identityError) {
      byQuery.set(query.id, {
        datasetId: bundle.id,
        frameworkId: expectedFrameworkId,
        queryId: query.id,
        status: "blocked",
        output: EMPTY_ANSWER,
        latencyMs: 0,
        inputTokens: null,
        outputTokens: null,
        error: identityError,
        inputDigest,
      });
      continue;
    }
    if (checkpoint?.status === "ok" && checkpoint.inputDigest === inputDigest) continue;
    byQuery.delete(query.id);
    if (!retrieval || retrieval.status !== "ok") {
      byQuery.set(query.id, {
        datasetId: bundle.id,
        frameworkId: retrieval?.frameworkId ?? inferFrameworkId(retrievals),
        queryId: query.id,
        status: retrieval?.status ?? "blocked",
        output: EMPTY_ANSWER,
        latencyMs: 0,
        inputTokens: null,
        outputTokens: null,
        error: retrieval?.error ?? "Retrieval result missing",
        inputDigest,
      });
    }
  }
  writeJsonLines(outputPath, orderedValues(evaluationIds, byQuery));

  const pending = evaluationQueries.filter((query) => {
    const checkpoint = byQuery.get(query.id);
    if (
      checkpoint?.status === "ok" &&
      checkpoint.inputDigest === requiredMapValue(inputDigestByQuery, query.id)
    ) {
      return false;
    }
    const retrieval = retrievalById.get(query.id);
    return (
      retrieval?.status === "ok" &&
      !prerequisiteIdentityError(bundle.id, expectedFrameworkId, query.id, retrieval)
    );
  });
  const answerBatches = batches(pending, manifest.benchmarkPolicy.answerCodexBatchSize);
  for (const wave of batches(answerBatches, manifest.benchmarkPolicy.codexConcurrency)) {
    const outcomes = await Promise.all(
      wave.map(async (batch) => {
        const startedAt = new Date().toISOString();
        try {
          const result = await retry(
            () =>
              codexClient.answerBatch(
                {
                  model: manifest.models.answer.model,
                  reasoningEffort: manifest.models.answer.reasoningEffort!,
                },
                batch.map((query) => {
                  const retrieval = retrievalById.get(query.id);
                  if (!retrieval) throw new Error(`Retrieval result missing for ${query.id}`);
                  return {
                    query,
                    evidence: retrieval.evidence,
                    answerPolicy: retrieval.answerPolicy,
                  };
                }),
              ),
            manifest.benchmarkPolicy.maxRetries,
          );
          return {
            status: "ok",
            batch,
            result,
            startedAt,
            completedAt: new Date().toISOString(),
          } as const;
        } catch (error) {
          return {
            status: "error",
            batch,
            error: error as Error,
            startedAt,
            completedAt: new Date().toISOString(),
          } as const;
        }
      }),
    );
    for (const outcome of outcomes) {
      if (outcome.status === "ok") {
        outcome.result.value.forEach((item, index) => {
          const retrieval = retrievalById.get(item.queryId)!;
          byQuery.set(item.queryId, {
            datasetId: bundle.id,
            frameworkId: retrieval.frameworkId,
            queryId: item.queryId,
            status: "ok",
            output: item.value,
            latencyMs: outcome.result.latencyMs / outcome.result.value.length,
            inputTokens: distributedValue(
              outcome.result.inputTokens,
              index,
              outcome.result.value.length,
            ),
            outputTokens: distributedValue(
              outcome.result.outputTokens,
              index,
              outcome.result.value.length,
            ),
            error: null,
            inputDigest: requiredMapValue(inputDigestByQuery, item.queryId),
            startedAt: outcome.startedAt,
            completedAt: outcome.completedAt,
          });
        });
        continue;
      }
      for (const query of outcome.batch) {
        byQuery.set(query.id, {
          datasetId: bundle.id,
          frameworkId: retrievalById.get(query.id)!.frameworkId,
          queryId: query.id,
          status: "error",
          output: EMPTY_ANSWER,
          latencyMs: 0,
          inputTokens: null,
          outputTokens: null,
          error: outcome.error.message,
          inputDigest: requiredMapValue(inputDigestByQuery, query.id),
          startedAt: outcome.startedAt,
          completedAt: outcome.completedAt,
        });
      }
    }
    writeJsonLines(outputPath, orderedValues(evaluationIds, byQuery));
  }
  return orderedValues(evaluationIds, byQuery);
}

export async function judgeAnswers(
  manifest: RagEvalManifest,
  bundle: ReturnType<typeof loadDataset>,
  expectedFrameworkId: FrameworkId,
  retrievals: readonly RetrievalResult[],
  answers: readonly AnswerResult[],
  evaluationQueries: readonly BenchmarkQuery[],
  frameworkDirectory: string,
  codexClient: JsonLlmClient,
): Promise<JudgeResult[]> {
  const outputPath = join(frameworkDirectory, "judgements.jsonl");
  const existing = existsSync(outputPath) ? readJsonLines<JudgeResult>(outputPath) : [];
  const byQuery = new Map(existing.map((result) => [result.queryId, result]));
  const retrievalById = new Map(retrievals.map((result) => [result.queryId, result]));
  const answerById = new Map(answers.map((result) => [result.queryId, result]));
  const evaluationIds = evaluationQueries.map((query) => query.id);
  const answerInputDigestByQuery = new Map(
    evaluationQueries.map((query) => [
      query.id,
      answerInputDigest(manifest, query, retrievalById.get(query.id)),
    ]),
  );
  const inputDigestByQuery = new Map(
    evaluationQueries.map((query) => [
      query.id,
      judgeInputDigest(manifest, query, retrievalById.get(query.id), answerById.get(query.id)),
    ]),
  );

  for (const query of evaluationQueries) {
    const inputDigest = requiredMapValue(inputDigestByQuery, query.id);
    const checkpoint = byQuery.get(query.id);
    const answer = answerById.get(query.id);
    const retrieval = retrievalById.get(query.id);
    const identityError = prerequisiteIdentityError(
      bundle.id,
      expectedFrameworkId,
      query.id,
      retrieval,
      answer,
    );
    if (identityError) {
      byQuery.set(query.id, {
        datasetId: bundle.id,
        frameworkId: expectedFrameworkId,
        queryId: query.id,
        status: "blocked",
        output: null,
        latencyMs: 0,
        inputTokens: null,
        outputTokens: null,
        error: identityError,
        inputDigest,
      });
      continue;
    }
    if (answer && answer.inputDigest !== requiredMapValue(answerInputDigestByQuery, query.id)) {
      byQuery.set(query.id, {
        datasetId: bundle.id,
        frameworkId: answer.frameworkId,
        queryId: query.id,
        status: "blocked",
        output: null,
        latencyMs: 0,
        inputTokens: null,
        outputTokens: null,
        error: "Answer input digest mismatch; rerun the answer stage",
        inputDigest,
      });
      continue;
    }
    if (checkpoint?.status === "ok" && checkpoint.inputDigest === inputDigest) continue;
    byQuery.delete(query.id);
    if (!answer) {
      byQuery.set(query.id, {
        datasetId: bundle.id,
        frameworkId: retrieval?.frameworkId ?? inferFrameworkId(retrievals),
        queryId: query.id,
        status: "blocked",
        output: null,
        latencyMs: 0,
        inputTokens: null,
        outputTokens: null,
        error: "Answer result missing",
        inputDigest,
      });
      continue;
    }
    if (answer.status !== "ok" || !retrieval || retrieval.status !== "ok") {
      const status = answer.status === "ok" ? (retrieval?.status ?? "blocked") : answer.status;
      byQuery.set(query.id, {
        datasetId: bundle.id,
        frameworkId: answer.frameworkId,
        queryId: query.id,
        status,
        output: null,
        latencyMs: 0,
        inputTokens: null,
        outputTokens: null,
        error: answer.error ?? retrieval?.error ?? "Prerequisite stage unavailable",
        inputDigest,
      });
    }
  }
  writeJsonLines(outputPath, orderedValues(evaluationIds, byQuery));

  const pending = evaluationQueries.filter((query) => {
    const checkpoint = byQuery.get(query.id);
    if (
      checkpoint?.status === "ok" &&
      checkpoint.inputDigest === requiredMapValue(inputDigestByQuery, query.id)
    ) {
      return false;
    }
    return (
      answerById.get(query.id)?.status === "ok" &&
      answerById.get(query.id)?.inputDigest ===
        requiredMapValue(answerInputDigestByQuery, query.id) &&
      retrievalById.get(query.id)?.status === "ok" &&
      !prerequisiteIdentityError(
        bundle.id,
        expectedFrameworkId,
        query.id,
        retrievalById.get(query.id),
        answerById.get(query.id),
      )
    );
  });
  const judgeBatches = batches(pending, manifest.benchmarkPolicy.judgeCodexBatchSize);
  for (const wave of batches(
    judgeBatches,
    manifest.benchmarkPolicy.judgeCodexConcurrency ?? manifest.benchmarkPolicy.codexConcurrency,
  )) {
    const outcomes = await Promise.all(
      wave.map(async (batch) => {
        const startedAt = new Date().toISOString();
        try {
          const result = await retry(
            () =>
              codexClient.judgeBatch(
                {
                  model: manifest.models.judge.model,
                  reasoningEffort: manifest.models.judge.reasoningEffort!,
                  timeoutMs: manifest.benchmarkPolicy.judgeTimeoutMs ?? 1_800_000,
                },
                batch.map((query) => ({
                  query,
                  evidence: retrievalById.get(query.id)!.evidence,
                  answer: answerById.get(query.id)!.output,
                })),
              ),
            manifest.benchmarkPolicy.maxRetries,
          );
          return {
            status: "ok",
            batch,
            result,
            startedAt,
            completedAt: new Date().toISOString(),
          } as const;
        } catch (error) {
          return {
            status: "error",
            batch,
            error: error as Error,
            startedAt,
            completedAt: new Date().toISOString(),
          } as const;
        }
      }),
    );
    for (const outcome of outcomes) {
      if (outcome.status === "ok") {
        outcome.result.value.forEach((item, index) => {
          const answer = answerById.get(item.queryId)!;
          byQuery.set(item.queryId, {
            datasetId: bundle.id,
            frameworkId: answer.frameworkId,
            queryId: item.queryId,
            status: "ok",
            output: item.value,
            latencyMs: outcome.result.latencyMs / outcome.result.value.length,
            inputTokens: distributedValue(
              outcome.result.inputTokens,
              index,
              outcome.result.value.length,
            ),
            outputTokens: distributedValue(
              outcome.result.outputTokens,
              index,
              outcome.result.value.length,
            ),
            error: null,
            inputDigest: requiredMapValue(inputDigestByQuery, item.queryId),
            startedAt: outcome.startedAt,
            completedAt: outcome.completedAt,
          });
        });
        continue;
      }
      for (const query of outcome.batch) {
        const answer = answerById.get(query.id)!;
        byQuery.set(query.id, {
          datasetId: bundle.id,
          frameworkId: answer.frameworkId,
          queryId: query.id,
          status: "error",
          output: null,
          latencyMs: 0,
          inputTokens: null,
          outputTokens: null,
          error: outcome.error.message,
          inputDigest: requiredMapValue(inputDigestByQuery, query.id),
          startedAt: outcome.startedAt,
          completedAt: outcome.completedAt,
        });
      }
    }
    writeJsonLines(outputPath, orderedValues(evaluationIds, byQuery));
  }
  return orderedValues(evaluationIds, byQuery);
}

export function answerInputDigest(
  manifest: RagEvalManifest,
  query: BenchmarkQuery,
  retrieval: RetrievalResult | undefined,
): string {
  return stageInputDigest(ANSWER_INPUT_DIGEST_VERSION, {
    model: {
      model: manifest.models.answer.model,
      reasoningEffort: manifest.models.answer.reasoningEffort,
      execution: manifest.models.answer.execution,
    },
    query: { id: query.id, text: query.text },
    retrieval: retrieval
      ? {
          datasetId: retrieval.datasetId,
          frameworkId: retrieval.frameworkId,
          queryId: retrieval.queryId,
          status: retrieval.status,
          frameworkVersion: retrieval.frameworkVersion,
          configDigest: retrieval.configDigest,
          answerPolicy: retrieval.answerPolicy ?? null,
          evidence: retrieval.evidence.map((item) => ({ id: item.id, text: item.text })),
        }
      : null,
  });
}

export function judgeInputDigest(
  manifest: RagEvalManifest,
  query: BenchmarkQuery,
  retrieval: RetrievalResult | undefined,
  answer: AnswerResult | undefined,
): string {
  return stageInputDigest(JUDGE_INPUT_DIGEST_VERSION, {
    model: {
      model: manifest.models.judge.model,
      reasoningEffort: manifest.models.judge.reasoningEffort,
      execution: manifest.models.judge.execution,
    },
    query: {
      id: query.id,
      text: query.text,
      answerable: query.answerable,
      referenceAnswer: query.referenceAnswer,
      goldEvidenceText: query.goldEvidenceText,
    },
    retrieval: retrieval
      ? {
          datasetId: retrieval.datasetId,
          frameworkId: retrieval.frameworkId,
          queryId: retrieval.queryId,
          status: retrieval.status,
          frameworkVersion: retrieval.frameworkVersion,
          configDigest: retrieval.configDigest,
          evidence: retrieval.evidence.map((item) => ({ id: item.id, text: item.text })),
        }
      : null,
    answer: answer
      ? {
          datasetId: answer.datasetId,
          frameworkId: answer.frameworkId,
          queryId: answer.queryId,
          status: answer.status,
          inputDigest: answer.inputDigest,
          output: answer.output,
        }
      : null,
  });
}

function stageInputDigest(version: string, value: unknown): string {
  return createHash("sha256")
    .update(version)
    .update("\0")
    .update(JSON.stringify(value))
    .digest("hex");
}

function prerequisiteIdentityError(
  datasetId: DatasetId,
  expectedFrameworkId: FrameworkId,
  queryId: string,
  retrieval: RetrievalResult | undefined,
  answer?: AnswerResult,
): string | null {
  if (
    retrieval &&
    (retrieval.datasetId !== datasetId ||
      retrieval.frameworkId !== expectedFrameworkId ||
      retrieval.queryId !== queryId)
  ) {
    return `Retrieval identity mismatch for ${queryId}`;
  }
  if (
    answer &&
    (answer.datasetId !== datasetId ||
      answer.frameworkId !== expectedFrameworkId ||
      (retrieval && answer.frameworkId !== retrieval.frameworkId) ||
      answer.queryId !== queryId)
  ) {
    return `Answer identity mismatch for ${queryId}`;
  }
  return null;
}

function requiredMapValue<K, V>(values: ReadonlyMap<K, V>, key: K): V {
  const value = values.get(key);
  if (value === undefined) throw new Error("Required checkpoint digest is missing");
  return value;
}

function inferFrameworkId(retrievals: readonly RetrievalResult[]): FrameworkId {
  const frameworkId = retrievals[0]?.frameworkId;
  if (!frameworkId) throw new Error("Cannot infer framework ID from empty retrieval results");
  return frameworkId;
}

function batches<T>(values: readonly T[], size: number): T[][] {
  const output: T[][] = [];
  for (let offset = 0; offset < values.length; offset += size) {
    output.push(values.slice(offset, offset + size));
  }
  return output;
}

function distributedValue(total: number | null, index: number, count: number): number | null {
  if (total === null) return null;
  return Math.floor(total / count) + (index < total % count ? 1 : 0);
}

function orderedValues<T>(ids: readonly string[], values: ReadonlyMap<string, T>): T[] {
  return ids.flatMap((id) => {
    const value = values.get(id);
    return value === undefined ? [] : [value];
  });
}

async function retry<T>(operation: () => Promise<T>, maxRetries: number): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      if (attempt < maxRetries) {
        const message = error instanceof Error ? error.message : String(error);
        process.stderr.write(
          `[rag-eval-v2] retry ${attempt + 1}/${maxRetries} after: ${message.slice(0, 1_000)}\n`,
        );
        await delay(Math.min(1_000 * 2 ** attempt, 4_000));
      }
    }
  }
  throw lastError;
}

async function doctorFramework(
  adapter: FrameworkAdapter,
  maxRetries: number,
): Promise<FrameworkDoctorResult> {
  let result = await adapter.doctor();
  for (
    let attempt = 0;
    result.status !== "ready" && result.version === "unresolved" && attempt < maxRetries;
    attempt += 1
  ) {
    logProgress(
      `doctor retry ${attempt + 1}/${maxRetries} framework=${adapter.id} after=${result.detail.slice(0, 500)}`,
    );
    await delay(Math.min(1_000 * 2 ** attempt, 4_000));
    result = await adapter.doctor();
  }
  return result;
}

async function delay(milliseconds: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function logProgress(message: string): void {
  process.stderr.write(`[rag-eval-v2] ${new Date().toISOString()} ${message}\n`);
}
