import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { DatasetId, FrameworkId } from "./contracts.js";
import { defaultDatasetPaths } from "./datasets.js";
import { DEFAULT_RAG_EVAL_MANIFEST, manifestForRunDirectory } from "./manifest.js";
import { doctorBenchmark, runBenchmark } from "./pipeline.js";
import { prepareFramesDataset } from "./prepare-frames.js";

interface CliOptions {
  readonly command: "doctor" | "prepare-frames" | "smoke" | "run";
  readonly workDirectory: string;
  readonly stage: "retrieval" | "full";
  readonly datasetIds?: readonly DatasetId[];
  readonly frameworkIds?: readonly FrameworkId[];
  readonly limit?: number;
  readonly topK: number;
  readonly candidateK: number;
}

async function main(): Promise<void> {
  const repositoryRoot = resolve(fileURLToPath(import.meta.url), "../../../..");
  const localEnvironment = resolve(repositoryRoot, ".env.local");
  if (existsSync(localEnvironment)) process.loadEnvFile(localEnvironment);
  const options = parseCli(process.argv.slice(2));
  const datasetPaths = defaultDatasetPaths(repositoryRoot);
  if (options.command === "prepare-frames") {
    const bundle = await prepareFramesDataset(resolve(datasetPaths.externalDataRoot, "frames"));
    process.stdout.write(
      `${JSON.stringify({ datasetId: bundle.id, documents: bundle.documents.length, queries: bundle.queries.length }, null, 2)}\n`,
    );
    return;
  }
  if (options.command === "doctor") {
    const report = await doctorBenchmark(DEFAULT_RAG_EVAL_MANIFEST, datasetPaths);
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
    return;
  }
  const runManifest = manifestForRunDirectory(DEFAULT_RAG_EVAL_MANIFEST, options.workDirectory);
  const report = await runBenchmark(runManifest, {
    workDirectory: options.workDirectory,
    stage: options.stage,
    datasetPaths,
    datasetIds:
      options.command === "smoke"
        ? (options.datasetIds ?? ["graphrag-bench-medical"])
        : options.datasetIds,
    frameworkIds: options.frameworkIds,
    datasetLoad: { limit: options.command === "smoke" ? (options.limit ?? 2) : options.limit },
    topK: options.topK,
    candidateK: options.candidateK,
  });
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
}

function parseCli(args: readonly string[]): CliOptions {
  const command = args[0] ?? "doctor";
  if (
    command !== "doctor" &&
    command !== "prepare-frames" &&
    command !== "smoke" &&
    command !== "run"
  ) {
    throw new Error(`Unknown command ${command}. Expected doctor, prepare-frames, smoke, or run.`);
  }
  const values = new Map<string, string>();
  for (let index = 1; index < args.length; index += 1) {
    const name = args[index]!;
    if (!name.startsWith("--")) throw new Error(`Unexpected argument ${name}`);
    const value = args[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for ${name}`);
    values.set(name, value);
    index += 1;
  }
  return {
    command,
    workDirectory: resolve(values.get("--work-dir") ?? "/tmp/kontext-rag-eval-v2"),
    stage: benchmarkStage(values.get("--stage")),
    datasetIds: csv(values.get("--datasets")) as DatasetId[] | undefined,
    frameworkIds: csv(values.get("--frameworks")) as FrameworkId[] | undefined,
    limit: optionalPositiveInteger(values.get("--limit"), "--limit"),
    topK: positiveInteger(values.get("--top-k") ?? "10", "--top-k"),
    candidateK: positiveInteger(values.get("--candidate-k") ?? "50", "--candidate-k"),
  };
}

function benchmarkStage(value: string | undefined): "retrieval" | "full" {
  if (value === undefined || value === "full") return "full";
  if (value === "retrieval") return "retrieval";
  throw new Error(`--stage must be retrieval or full, found ${value}`);
}

function csv(value: string | undefined): string[] | undefined {
  return value
    ?.split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function optionalPositiveInteger(value: string | undefined, name: string): number | undefined {
  return value === undefined ? undefined : positiveInteger(value, name);
}

function positiveInteger(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0)
    throw new Error(`${name} must be a positive integer`);
  return parsed;
}

main().catch((error) => {
  process.stderr.write(`${(error as Error).stack ?? (error as Error).message}\n`);
  process.exitCode = 1;
});
