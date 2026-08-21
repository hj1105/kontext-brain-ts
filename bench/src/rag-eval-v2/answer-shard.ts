import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { CodexJsonClient } from "./codex-json.js";
import type { AnswerResult, RetrievalResult } from "./contracts.js";
import { defaultDatasetPaths, loadDataset } from "./datasets.js";
import type { EvaluationSampleManifest } from "./evaluation-sample.js";
import { readJsonLines } from "./jsonl.js";
import { DEFAULT_RAG_EVAL_MANIFEST } from "./manifest.js";
import { answerQueries } from "./pipeline.js";

async function main(): Promise<void> {
  const repositoryRoot = resolve(import.meta.dirname, "../../..");
  const localEnvironment = join(repositoryRoot, ".env.local");
  if (existsSync(localEnvironment)) process.loadEnvFile(localEnvironment);
  const workDirectory = requiredArgument("--work-dir");
  const shardIndex = nonNegativeInteger(requiredArgument("--shard-index"), "--shard-index");
  const shardCount = positiveInteger(requiredArgument("--shard-count"), "--shard-count");
  if (shardIndex >= shardCount) throw new Error("--shard-index must be smaller than --shard-count");

  const datasetId = "graphrag-bench-medical" as const;
  const frameworkId = "kontext-brain" as const;
  const datasetDirectory = join(workDirectory, datasetId);
  const frameworkDirectory = join(datasetDirectory, frameworkId);
  const sample = JSON.parse(
    readFileSync(join(datasetDirectory, "evaluation-sample.json"), "utf8"),
  ) as EvaluationSampleManifest;
  const bundle = loadDataset(datasetId, defaultDatasetPaths(repositoryRoot));
  const queryById = new Map(bundle.queries.map((query) => [query.id, query]));
  const retrievals = readJsonLines<RetrievalResult>(join(frameworkDirectory, "retrieval.jsonl"));
  const primaryAnswersPath = join(frameworkDirectory, "answers.jsonl");
  const completedIds = new Set(
    (existsSync(primaryAnswersPath) ? readJsonLines<AnswerResult>(primaryAnswersPath) : [])
      .filter((result) => result.status === "ok")
      .map((result) => result.queryId),
  );
  const pending = sample.queryIds
    .filter((queryId) => !completedIds.has(queryId))
    .map((queryId) => {
      const query = queryById.get(queryId);
      if (!query) throw new Error(`Evaluation query ${queryId} is missing from the dataset`);
      return query;
    });
  const shardQueries = pending.filter((_query, index) => index % shardCount === shardIndex);
  const shardDirectory = join(
    frameworkDirectory,
    "answer-shards",
    `part-${String(shardIndex).padStart(2, "0")}-of-${String(shardCount).padStart(2, "0")}`,
  );
  const results = await answerQueries(
    DEFAULT_RAG_EVAL_MANIFEST,
    bundle,
    retrievals,
    shardQueries,
    shardDirectory,
    new CodexJsonClient(),
  );
  process.stdout.write(
    `${JSON.stringify({
      shardIndex,
      shardCount,
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
