import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type { JudgeResult } from "./contracts.js";
import type { EvaluationSampleManifest } from "./evaluation-sample.js";
import { readJsonLines, writeJsonLines } from "./jsonl.js";

function main(): void {
  const workDirectory = requiredArgument("--work-dir");
  const shardCount = positiveInteger(requiredArgument("--shard-count"), "--shard-count");
  const datasetDirectory = join(workDirectory, "graphrag-bench-medical");
  const frameworkDirectory = join(datasetDirectory, "kontext-brain");
  const sample = JSON.parse(
    readFileSync(join(datasetDirectory, "evaluation-sample.json"), "utf8"),
  ) as EvaluationSampleManifest;
  const primaryPath = join(frameworkDirectory, "judgements.jsonl");
  const byQuery = new Map<string, JudgeResult>();
  if (existsSync(primaryPath)) {
    for (const result of readJsonLines<JudgeResult>(primaryPath)) {
      if (result.status === "ok") byQuery.set(result.queryId, result);
    }
  }
  for (let shardIndex = 0; shardIndex < shardCount; shardIndex += 1) {
    const shardPath = join(
      frameworkDirectory,
      "judge-shards",
      `part-${String(shardIndex).padStart(2, "0")}-of-${String(shardCount).padStart(2, "0")}`,
      "judgements.jsonl",
    );
    if (!existsSync(shardPath)) throw new Error(`Missing judge shard ${shardPath}`);
    for (const result of readJsonLines<JudgeResult>(shardPath)) {
      const existing = byQuery.get(result.queryId);
      if (!existing || existing.status !== "ok") byQuery.set(result.queryId, result);
    }
  }
  const merged = sample.queryIds.flatMap((queryId) => {
    const result = byQuery.get(queryId);
    return result ? [result] : [];
  });
  const incomplete = sample.queryIds.filter((queryId) => byQuery.get(queryId)?.status !== "ok");
  writeJsonLines(primaryPath, merged);
  process.stdout.write(
    `${JSON.stringify({
      output: resolve(primaryPath),
      records: merged.length,
      completed: merged.filter((result) => result.status === "ok").length,
      incomplete,
    })}\n`,
  );
  if (incomplete.length > 0) process.exitCode = 2;
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

main();
