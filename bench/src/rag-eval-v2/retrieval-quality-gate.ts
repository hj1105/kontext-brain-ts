import { createHash } from "node:crypto";
import type { BenchmarkQuery, RetrievalResult } from "./contracts.js";
import { readJsonLines } from "./jsonl.js";
import { contextPrecisionForQuery, evidenceRecallForQuery } from "./metrics.js";

interface SplitScore {
  readonly queries: number;
  readonly evidenceRecallAtK: number | null;
  readonly contextPrecision: number | null;
}

const queriesPath = requiredArgument("--queries");
const retrievalPath = requiredArgument("--retrieval");
const minimumHoldoutRecall = numberArgument("--min-holdout-recall", 0);
const holdoutPercent = numberArgument("--holdout-percent", 20);
if (holdoutPercent <= 0 || holdoutPercent >= 100) {
  throw new Error("--holdout-percent must be between 0 and 100");
}

const queries = readJsonLines<BenchmarkQuery>(queriesPath);
const retrievalByQuery = new Map(
  readJsonLines<RetrievalResult>(retrievalPath).map((retrieval) => [retrieval.queryId, retrieval]),
);
const development = queries.filter((query) => splitBucket(query.id) >= holdoutPercent);
const holdout = queries.filter((query) => splitBucket(query.id) < holdoutPercent);
const result = {
  retrieval: retrievalPath,
  holdoutPercent,
  all: score(queries, retrievalByQuery),
  development: score(development, retrievalByQuery),
  holdout: score(holdout, retrievalByQuery),
  minimumHoldoutRecall,
};
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
if ((result.holdout.evidenceRecallAtK ?? 0) < minimumHoldoutRecall) process.exitCode = 1;

function score(
  selectedQueries: readonly BenchmarkQuery[],
  retrievals: ReadonlyMap<string, RetrievalResult>,
): SplitScore {
  const recall: number[] = [];
  const precision: number[] = [];
  for (const query of selectedQueries) {
    const retrieval = retrievals.get(query.id);
    if (!retrieval || retrieval.status !== "ok") continue;
    const recallScore = evidenceRecallForQuery(query, retrieval.evidence);
    const precisionScore = contextPrecisionForQuery(query, retrieval.evidence);
    if (recallScore !== null) recall.push(recallScore);
    if (precisionScore !== null) precision.push(precisionScore);
  }
  return {
    queries: selectedQueries.length,
    evidenceRecallAtK: meanOrNull(recall),
    contextPrecision: meanOrNull(precision),
  };
}

function splitBucket(queryId: string): number {
  return createHash("sha256").update(queryId).digest().readUInt32BE(0) % 100;
}

function meanOrNull(values: readonly number[]): number | null {
  return values.length === 0
    ? null
    : values.reduce((total, value) => total + value, 0) / values.length;
}

function requiredArgument(name: string): string {
  const index = process.argv.indexOf(name);
  const value = index >= 0 ? process.argv[index + 1] : undefined;
  if (!value) throw new Error(`Missing ${name}`);
  return value;
}

function numberArgument(name: string, fallback: number): number {
  const index = process.argv.indexOf(name);
  if (index < 0) return fallback;
  const value = Number(requiredArgument(name));
  if (!Number.isFinite(value)) throw new Error(`${name} must be a number`);
  return value;
}
