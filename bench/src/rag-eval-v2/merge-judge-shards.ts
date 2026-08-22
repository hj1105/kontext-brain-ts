import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import type {
  AnswerResult,
  BenchmarkQuery,
  DatasetBundle,
  DatasetId,
  JudgeResult,
  RetrievalResult,
} from "./contracts.js";
import { defaultDatasetPaths, loadDataset } from "./datasets.js";
import type { EvaluationSampleManifest } from "./evaluation-sample.js";
import { readJsonLines, writeJsonLines } from "./jsonl.js";
import { type RagEvalManifest, loadFrozenRunManifest } from "./manifest.js";
import type { IndexedShard } from "./merge-answer-shards.js";
import { answerInputDigest, judgeInputDigest } from "./pipeline.js";

export interface MergeJudgeShardRecordsOptions {
  readonly manifest: RagEvalManifest;
  readonly bundle: DatasetBundle;
  readonly evaluationQueries: readonly BenchmarkQuery[];
  readonly retrievals: readonly RetrievalResult[];
  readonly answers: readonly AnswerResult[];
  readonly shardCount: number;
  readonly shards: readonly IndexedShard<JudgeResult>[];
}

export function mergeJudgeShardRecords(options: MergeJudgeShardRecordsOptions): JudgeResult[] {
  assertCompleteShardSet(options.shards, options.shardCount);
  const queryIndex = uniqueQueryIndex(options.evaluationQueries);
  const retrievalById = uniqueRecordsByQuery(options.retrievals, "retrieval");
  const answerById = uniqueRecordsByQuery(options.answers, "answer");
  const mergedById = new Map<string, JudgeResult>();

  for (const query of options.evaluationQueries) {
    const retrieval = retrievalById.get(query.id);
    if (!retrieval) throw new Error(`Retrieval result missing for ${query.id}`);
    if (
      retrieval.datasetId !== options.bundle.id ||
      retrieval.frameworkId !== "kontext-brain" ||
      retrieval.queryId !== query.id
    ) {
      throw new Error(`Retrieval identity mismatch for ${query.id}`);
    }
    const answer = answerById.get(query.id);
    if (!answer) throw new Error(`Answer result missing for ${query.id}`);
    if (answer.datasetId !== options.bundle.id || answer.frameworkId !== retrieval.frameworkId) {
      throw new Error(`Answer identity mismatch for ${query.id}`);
    }
    if (answer.inputDigest !== answerInputDigest(options.manifest, query, retrieval)) {
      throw new Error(`Answer input digest mismatch for ${query.id}`);
    }
  }

  for (const shard of options.shards) {
    for (const result of shard.records) {
      const index = queryIndex.get(result.queryId);
      if (index === undefined) throw new Error(`Unexpected judge shard query ${result.queryId}`);
      const expectedShard = index % options.shardCount;
      if (shard.shardIndex !== expectedShard) {
        throw new Error(
          `Judge query ${result.queryId} is assigned to shard ${expectedShard}, not ${shard.shardIndex}`,
        );
      }
      if (mergedById.has(result.queryId)) {
        throw new Error(`Duplicate judge shard record for ${result.queryId}`);
      }
      const query = options.evaluationQueries[index];
      if (!query) throw new Error(`Evaluation query missing at index ${index}`);
      const retrieval = retrievalById.get(result.queryId);
      if (!retrieval) throw new Error(`Retrieval result missing for ${result.queryId}`);
      const answer = answerById.get(result.queryId);
      if (!answer) throw new Error(`Answer result missing for ${result.queryId}`);
      if (result.datasetId !== options.bundle.id || result.frameworkId !== answer.frameworkId) {
        throw new Error(`Judge shard identity mismatch for ${result.queryId}`);
      }
      const expectedDigest = judgeInputDigest(options.manifest, query, retrieval, answer);
      if (result.inputDigest !== expectedDigest) {
        throw new Error(`Judge input digest mismatch for ${result.queryId}`);
      }
      mergedById.set(result.queryId, result);
    }
  }

  return options.evaluationQueries.map((query) => {
    const result = mergedById.get(query.id);
    if (!result) throw new Error(`Missing judge shard record for ${query.id}`);
    return result;
  });
}

function main(): void {
  const repositoryRoot = resolve(import.meta.dirname, "../../..");
  const workDirectory = requiredArgument("--work-dir");
  const shardCount = positiveInteger(requiredArgument("--shard-count"), "--shard-count");
  const requestedDatasetId = optionalArgument("--dataset") ?? "graphrag-bench-medical";
  const frameworkId = "kontext-brain" as const;
  const manifest = loadFrozenRunManifest(join(workDirectory, "run-manifest.json"));
  const datasetId = requestedDatasetId as DatasetId;
  if (!manifest.datasets.some((dataset) => dataset.id === datasetId)) {
    throw new Error(`Dataset ${requestedDatasetId} is not present in the frozen run manifest`);
  }
  const datasetDirectory = join(workDirectory, datasetId);
  const frameworkDirectory = join(datasetDirectory, frameworkId);
  const sample = JSON.parse(
    readFileSync(join(datasetDirectory, "evaluation-sample.json"), "utf8"),
  ) as EvaluationSampleManifest;
  const bundle = loadDataset(datasetId, defaultDatasetPaths(repositoryRoot));
  const queryById = new Map(bundle.queries.map((query) => [query.id, query]));
  const evaluationQueries = sample.queryIds.map((queryId) => {
    const query = queryById.get(queryId);
    if (!query) throw new Error(`Evaluation query ${queryId} is missing from the dataset`);
    return query;
  });
  const retrievals = readJsonLines<RetrievalResult>(join(frameworkDirectory, "retrieval.jsonl"));
  const answers = readJsonLines<AnswerResult>(join(frameworkDirectory, "answers.jsonl"));
  const shards: IndexedShard<JudgeResult>[] = [];
  for (let shardIndex = 0; shardIndex < shardCount; shardIndex += 1) {
    const shardPath = join(
      frameworkDirectory,
      "judge-shards",
      shardDirectoryName(shardIndex, shardCount),
      "judgements.jsonl",
    );
    if (!existsSync(shardPath)) throw new Error(`Missing judge shard ${shardPath}`);
    shards.push({ shardIndex, records: readJsonLines<JudgeResult>(shardPath) });
  }
  const merged = mergeJudgeShardRecords({
    manifest,
    bundle,
    evaluationQueries,
    retrievals,
    answers,
    shardCount,
    shards,
  });
  const primaryPath = join(frameworkDirectory, "judgements.jsonl");
  const incomplete = merged
    .filter((result) => result.status !== "ok")
    .map((result) => result.queryId);
  writeJsonLines(primaryPath, merged);
  process.stdout.write(
    `${JSON.stringify({
      output: resolve(primaryPath),
      datasetId,
      records: merged.length,
      completed: merged.filter((result) => result.status === "ok").length,
      incomplete,
    })}\n`,
  );
  if (incomplete.length > 0) process.exitCode = 2;
}

function assertCompleteShardSet<T>(shards: readonly IndexedShard<T>[], shardCount: number): void {
  if (!Number.isSafeInteger(shardCount) || shardCount <= 0) {
    throw new Error("shardCount must be positive");
  }
  const indexes = new Set<number>();
  for (const shard of shards) {
    if (shard.shardIndex < 0 || shard.shardIndex >= shardCount) {
      throw new Error(`Invalid judge shard index ${shard.shardIndex}`);
    }
    if (indexes.has(shard.shardIndex)) throw new Error(`Duplicate judge shard ${shard.shardIndex}`);
    indexes.add(shard.shardIndex);
  }
  for (let index = 0; index < shardCount; index += 1) {
    if (!indexes.has(index)) throw new Error(`Missing judge shard ${index}`);
  }
}

function uniqueQueryIndex(queries: readonly BenchmarkQuery[]): Map<string, number> {
  const output = new Map<string, number>();
  queries.forEach((query, index) => {
    if (output.has(query.id)) throw new Error(`Duplicate evaluation query ${query.id}`);
    output.set(query.id, index);
  });
  return output;
}

function uniqueRecordsByQuery<T extends { readonly queryId: string }>(
  records: readonly T[],
  label: string,
): Map<string, T> {
  const output = new Map<string, T>();
  for (const record of records) {
    if (output.has(record.queryId))
      throw new Error(`Duplicate ${label} result for ${record.queryId}`);
    output.set(record.queryId, record);
  }
  return output;
}

function shardDirectoryName(shardIndex: number, shardCount: number): string {
  return `part-${String(shardIndex).padStart(2, "0")}-of-${String(shardCount).padStart(2, "0")}`;
}

function requiredArgument(name: string): string {
  const index = process.argv.indexOf(name);
  const value = index >= 0 ? process.argv[index + 1] : undefined;
  if (!value) throw new Error(`Missing ${name}`);
  return value;
}

function optionalArgument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  if (index < 0) return undefined;
  const value = process.argv[index + 1];
  if (!value) throw new Error(`Missing ${name}`);
  return value;
}

function positiveInteger(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) throw new Error(`${name} must be positive`);
  return parsed;
}

const entryPoint = process.argv[1];
if (entryPoint && import.meta.url === pathToFileURL(resolve(entryPoint)).href) main();
