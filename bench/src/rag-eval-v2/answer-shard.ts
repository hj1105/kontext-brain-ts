import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { CodexJsonClient } from "./codex-json.js";
import type { DatasetId, RetrievalResult } from "./contracts.js";
import { defaultDatasetPaths, loadDataset } from "./datasets.js";
import type { EvaluationSampleManifest } from "./evaluation-sample.js";
import { readJsonLines } from "./jsonl.js";
import { loadFrozenRunManifest } from "./manifest.js";
import { answerQueries } from "./pipeline.js";

async function main(): Promise<void> {
  const repositoryRoot = resolve(import.meta.dirname, "../../..");
  const localEnvironment = join(repositoryRoot, ".env.local");
  if (existsSync(localEnvironment)) process.loadEnvFile(localEnvironment);
  const workDirectory = requiredArgument("--work-dir");
  const shardIndex = nonNegativeInteger(requiredArgument("--shard-index"), "--shard-index");
  const shardCount = positiveInteger(requiredArgument("--shard-count"), "--shard-count");
  if (shardIndex >= shardCount) throw new Error("--shard-index must be smaller than --shard-count");

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
  const retrievals = readJsonLines<RetrievalResult>(join(frameworkDirectory, "retrieval.jsonl"));
  const sampleQueries = sample.queryIds.map((queryId) => {
    const query = queryById.get(queryId);
    if (!query) throw new Error(`Evaluation query ${queryId} is missing from the dataset`);
    return query;
  });
  const shardQueries = sampleQueries.filter((_query, index) => index % shardCount === shardIndex);
  const shardDirectory = join(
    frameworkDirectory,
    "answer-shards",
    `part-${String(shardIndex).padStart(2, "0")}-of-${String(shardCount).padStart(2, "0")}`,
  );
  const results = await answerQueries(
    manifest,
    bundle,
    frameworkId,
    retrievals,
    shardQueries,
    shardDirectory,
    new CodexJsonClient(),
  );
  process.stdout.write(
    `${JSON.stringify({
      shardIndex,
      shardCount,
      datasetId,
      assigned: shardQueries.length,
      completed: results.filter((result) => result.status === "ok").length,
      errors: results.filter((result) => result.status === "error").length,
      output: join(shardDirectory, "answers.jsonl"),
    })}\n`,
  );
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

function nonNegativeInteger(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0) throw new Error(`${name} must be non-negative`);
  return parsed;
}

main().catch((error) => {
  process.stderr.write(`${(error as Error).stack ?? (error as Error).message}\n`);
  process.exitCode = 1;
});
