import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { isDeepStrictEqual } from "node:util";
import type {
  AnswerResult,
  BenchmarkQuery,
  DatasetBundle,
  DatasetId,
  RetrievalResult,
} from "./contracts.js";
import { defaultDatasetPaths, loadDataset } from "./datasets.js";
import type { EvaluationSampleManifest } from "./evaluation-sample.js";
import { readJsonLines, writeJsonLines } from "./jsonl.js";
import { type RagEvalManifest, loadFrozenRunManifest } from "./manifest.js";
import { answerInputDigest } from "./pipeline.js";

export interface IndexedShard<T> {
  readonly shardIndex: number;
  readonly records: readonly T[];
}

export interface MergeAnswerShardRecordsOptions {
  readonly manifest: RagEvalManifest;
  readonly bundle: DatasetBundle;
  readonly evaluationQueries: readonly BenchmarkQuery[];
  readonly retrievals: readonly RetrievalResult[];
  readonly shardCount: number;
  readonly shards: readonly IndexedShard<AnswerResult>[];
}

export function mergeAnswerShardRecords(options: MergeAnswerShardRecordsOptions): AnswerResult[] {
  assertCompleteShardSet(options.shards, options.shardCount, "answer");
  const queryIndex = uniqueQueryIndex(options.evaluationQueries);
  const retrievalById = uniqueRecordsByQuery(options.retrievals, "retrieval");
  const mergedById = new Map<string, AnswerResult>();

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
  }

  for (const shard of options.shards) {
    for (const result of shard.records) {
      const index = queryIndex.get(result.queryId);
      if (index === undefined) throw new Error(`Unexpected answer shard query ${result.queryId}`);
      const expectedShard = index % options.shardCount;
      if (shard.shardIndex !== expectedShard) {
        throw new Error(
          `Answer query ${result.queryId} is assigned to shard ${expectedShard}, not ${shard.shardIndex}`,
        );
      }
      if (mergedById.has(result.queryId)) {
        throw new Error(`Duplicate answer shard record for ${result.queryId}`);
      }
      const query = options.evaluationQueries[index];
      if (!query) throw new Error(`Evaluation query missing at index ${index}`);
      const retrieval = retrievalById.get(result.queryId);
      if (!retrieval) throw new Error(`Retrieval result missing for ${result.queryId}`);
      if (result.datasetId !== options.bundle.id || result.frameworkId !== retrieval.frameworkId) {
        throw new Error(`Answer shard identity mismatch for ${result.queryId}`);
      }
      const expectedDigest = answerInputDigest(options.manifest, query, retrieval);
      if (result.inputDigest !== expectedDigest) {
        throw new Error(`Answer input digest mismatch for ${result.queryId}`);
      }
      mergedById.set(result.queryId, result);
    }
  }

  return options.evaluationQueries.map((query) => {
    const result = mergedById.get(query.id);
    if (!result) throw new Error(`Missing answer shard record for ${query.id}`);
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
  const shards: IndexedShard<AnswerResult>[] = [];
  for (let shardIndex = 0; shardIndex < shardCount; shardIndex += 1) {
    const shardPath = join(
      frameworkDirectory,
      "answer-shards",
      shardDirectoryName(shardIndex, shardCount),
      "answers.jsonl",
    );
    if (!existsSync(shardPath)) throw new Error(`Missing answer shard ${shardPath}`);
    shards.push({ shardIndex, records: readJsonLines<AnswerResult>(shardPath) });
  }
  const merged = mergeAnswerShardRecords({
    manifest,
    bundle,
    evaluationQueries,
    retrievals,
    shardCount,
    shards,
  });
  const primaryPath = join(frameworkDirectory, "answers.jsonl");
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

function assertCompleteShardSet<T>(
  shards: readonly IndexedShard<T>[],
  shardCount: number,
  stage: string,
): void {
  if (!Number.isSafeInteger(shardCount) || shardCount <= 0) {
    throw new Error("shardCount must be positive");
  }
  const indexes = new Set<number>();
  for (const shard of shards) {
    if (shard.shardIndex < 0 || shard.shardIndex >= shardCount) {
      throw new Error(`Invalid ${stage} shard index ${shard.shardIndex}`);
    }
    if (indexes.has(shard.shardIndex))
      throw new Error(`Duplicate ${stage} shard ${shard.shardIndex}`);
    indexes.add(shard.shardIndex);
  }
  for (let index = 0; index < shardCount; index += 1) {
    if (!indexes.has(index)) throw new Error(`Missing ${stage} shard ${index}`);
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
    const existing = output.get(record.queryId);
    if (existing && !isDeepStrictEqual(existing, record))
      throw new Error(`Duplicate ${label} result for ${record.queryId}`);
    if (!existing) output.set(record.queryId, record);
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
