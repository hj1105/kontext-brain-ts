#!/usr/bin/env node
import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { runWorkspaceCommand } from "../workspace.js";
import type { DeepSweArm, DeepSwePrepareOptions } from "./contracts.js";
import { prepareDeepSweEvaluation } from "./prepare.js";
import { renderDeepSweMarkdown } from "./report.js";
import { runPreparedDeepSweEvaluation } from "./runner.js";

interface CliOptions
  extends Omit<
    DeepSwePrepareOptions,
    "repositoryRoot" | "jobsDirectory" | "pierRevision" | "adapterRevision"
  > {
  readonly outputPath: string;
  readonly dryRun: boolean;
  readonly declaredPierRevision?: string;
}

async function main(): Promise<void> {
  if (process.argv.includes("--help") || process.argv.includes("-h")) {
    process.stdout.write(helpText());
    return;
  }
  const repositoryRoot = fileURLToPath(new URL("../../../../", import.meta.url));
  const options = parseOptions(process.argv.slice(2), repositoryRoot);
  const localEnvironment = path.join(repositoryRoot, ".env.local");
  if (!options.envFile && existsSync(localEnvironment)) process.loadEnvFile(localEnvironment);
  const adapterRevision = await resolveAdapterRevision(repositoryRoot, options.dryRun);
  const pierRevision = await resolvePierRevision(
    options.pierBinary,
    options.declaredPierRevision,
    options.dryRun,
  );
  const jobsDirectory = path.join(options.runDirectory, "jobs");
  const manifest = await prepareDeepSweEvaluation({
    repositoryRoot,
    datasetTasksPath: options.datasetTasksPath,
    corpusRoot: options.corpusRoot,
    runDirectory: options.runDirectory,
    jobsDirectory,
    pierBinary: options.pierBinary,
    model: options.model,
    reasoningEffort: options.reasoningEffort,
    attempts: options.attempts,
    concurrency: options.concurrency,
    sampleSeed: options.sampleSeed,
    arms: options.arms,
    ...(options.taskIds ? { taskIds: options.taskIds } : {}),
    ...(options.taskLimit ? { taskLimit: options.taskLimit } : {}),
    environment: options.environment,
    ...(options.envFile ? { envFile: options.envFile } : {}),
    miniSweAgentVersion: options.miniSweAgentVersion,
    ...(options.deepSweRevision ? { deepSweRevision: options.deepSweRevision } : {}),
    pierRevision,
    adapterRevision,
  });
  if (options.dryRun) {
    process.stdout.write(`${JSON.stringify(manifest, null, 2)}\n`);
    return;
  }
  const report = await runPreparedDeepSweEvaluation({
    repositoryRoot,
    manifest,
    ...(options.envFile ? { envFile: options.envFile } : {}),
    dependencies: { onProgress: (message) => process.stderr.write(`${message}\n`) },
  });
  if (!report) throw new Error("DeepSWE runner returned no report");
  const outputPath = path.resolve(options.outputPath);
  const markdownPath = `${outputPath.replace(/\.json$/i, "")}.md`;
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  await writeFile(markdownPath, renderDeepSweMarkdown(report), "utf8");
  process.stdout.write(
    `${JSON.stringify(
      { outputPath, markdownPath, summaries: report.summaries, comparisons: report.comparisons },
      null,
      2,
    )}\n`,
  );
}

function parseOptions(args: readonly string[], repositoryRoot: string): CliOptions {
  let datasetTasksPath: string | undefined;
  let corpusRoot: string | undefined;
  let runDirectory = path.join(
    repositoryRoot,
    "bench",
    "data",
    "code-quality-eval",
    `deepswe-${compactTimestamp(new Date())}`,
  );
  let outputPath: string | undefined;
  let pierBinary = "pier";
  let model = "openai/gpt-5.5";
  let reasoningEffort = "medium";
  let attempts = 4;
  let concurrency = 1;
  let sampleSeed = 0;
  let arms: readonly DeepSweArm[] = ["baseline", "rag", "kontext"];
  let taskIds: readonly string[] | undefined;
  let taskLimit: number | undefined;
  let environment: "docker" | "modal" = "docker";
  let envFile: string | undefined;
  let miniSweAgentVersion: string | undefined;
  let deepSweRevision: string | undefined;
  let declaredPierRevision: string | undefined;
  let dryRun = false;
  for (let index = 0; index < args.length; index += 1) {
    const option = args[index];
    if (option === "--dry-run") {
      dryRun = true;
      continue;
    }
    const value = args[index + 1];
    if (!value) throw new Error(`Missing value for ${option}`);
    switch (option) {
      case "--dataset":
        datasetTasksPath = path.resolve(value);
        break;
      case "--corpus":
        corpusRoot = path.resolve(value);
        break;
      case "--run-dir":
        runDirectory = path.resolve(value);
        break;
      case "--output":
        outputPath = path.resolve(value);
        break;
      case "--pier-bin":
        pierBinary = value;
        break;
      case "--pier-revision":
        declaredPierRevision = value;
        break;
      case "--deepswe-revision":
        deepSweRevision = value;
        break;
      case "--mini-swe-version":
        miniSweAgentVersion = value;
        break;
      case "--model":
        model = value;
        break;
      case "--reasoning":
        reasoningEffort = value;
        break;
      case "--attempts":
        attempts = positiveInteger(value, option);
        break;
      case "--concurrency":
        concurrency = positiveInteger(value, option);
        break;
      case "--sample-seed":
        sampleSeed = integer(value, option);
        break;
      case "--arms":
        arms = parseArms(value);
        break;
      case "--tasks":
        taskIds = nonEmptyList(value, option);
        break;
      case "--task-limit":
        taskLimit = positiveInteger(value, option);
        break;
      case "--environment":
        if (value !== "docker" && value !== "modal") {
          throw new Error("--environment must be docker or modal");
        }
        environment = value;
        break;
      case "--env-file":
        envFile = path.resolve(value);
        break;
      default:
        throw new Error(`Unknown option: ${option}`);
    }
    index += 1;
  }
  if (!datasetTasksPath) throw new Error("--dataset is required");
  if (!corpusRoot) throw new Error("--corpus is required");
  if (!miniSweAgentVersion) throw new Error("--mini-swe-version is required for replayability");
  if (!model.includes("/")) throw new Error("--model must use provider/model format");
  return {
    datasetTasksPath,
    corpusRoot,
    runDirectory,
    pierBinary,
    model,
    reasoningEffort,
    attempts,
    concurrency,
    sampleSeed,
    arms,
    ...(taskIds ? { taskIds } : {}),
    ...(taskLimit ? { taskLimit } : {}),
    environment,
    ...(envFile ? { envFile } : {}),
    miniSweAgentVersion,
    ...(deepSweRevision ? { deepSweRevision } : {}),
    outputPath: outputPath ?? path.join(runDirectory, "report.json"),
    dryRun,
    ...(declaredPierRevision ? { declaredPierRevision } : {}),
  };
}

function parseArms(value: string): readonly DeepSweArm[] {
  const arms = nonEmptyList(value, "--arms");
  if (arms.some((arm) => arm !== "baseline" && arm !== "rag" && arm !== "kontext")) {
    throw new Error("--arms must be a comma-separated subset of baseline,rag,kontext");
  }
  return [...new Set(arms)] as DeepSweArm[];
}

async function resolvePierRevision(
  binary: string,
  declaredRevision: string | undefined,
  dryRun: boolean,
): Promise<string> {
  if (dryRun && declaredRevision) return declaredRevision;
  const result = await runWorkspaceCommand(process.cwd(), binary, ["--version"]);
  if (result.exitCode !== 0 || !result.stdout.trim()) {
    throw new Error(
      `Cannot resolve Pier version from ${binary}: ${result.stderr || result.stdout}`,
    );
  }
  const actualRevision = result.stdout.trim();
  if (declaredRevision && actualRevision !== declaredRevision) {
    throw new Error(`Pier revision mismatch: ${actualRevision} != ${declaredRevision}`);
  }
  return actualRevision;
}

async function resolveAdapterRevision(
  repositoryRoot: string,
  allowDirty: boolean,
): Promise<string> {
  const [revision, trackedStatus, adapterStatus] = await Promise.all([
    runWorkspaceCommand(repositoryRoot, "git", ["rev-parse", "HEAD"]),
    runWorkspaceCommand(repositoryRoot, "git", ["status", "--porcelain", "--untracked-files=no"]),
    runWorkspaceCommand(repositoryRoot, "git", [
      "status",
      "--porcelain",
      "--untracked-files=all",
      "--",
      "bench/src/code-quality-eval/deepswe",
      "bench/package.json",
    ]),
  ]);
  if (revision.exitCode !== 0 || trackedStatus.exitCode !== 0 || adapterStatus.exitCode !== 0) {
    throw new Error("Cannot resolve Kontext adapter revision");
  }
  const dirty = Boolean(trackedStatus.stdout.trim() || adapterStatus.stdout.trim());
  if (dirty && !allowDirty) {
    throw new Error("Refusing a scored DeepSWE run from a dirty Kontext checkout");
  }
  return `${revision.stdout.trim()}${dirty ? "+dirty" : ""}`;
}

function nonEmptyList(value: string, option: string): string[] {
  const values = value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  if (!values.length) throw new Error(`${option} requires at least one value`);
  return values;
}

function positiveInteger(value: string, option: string): number {
  const parsed = integer(value, option);
  if (parsed <= 0) throw new Error(`${option} must be positive`);
  return parsed;
}

function integer(value: string, option: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${option} must be an integer`);
  return parsed;
}

function compactTimestamp(date: Date): string {
  return date
    .toISOString()
    .replace(/[-:]/g, "")
    .replace(/\.\d{3}Z$/, "Z");
}

function helpText(): string {
  return `Usage: pnpm --filter @kontext-brain/bench code-quality:deepswe -- [options]

Runs a DeepSWE-based paired context evaluation through the official Pier task
and verifier isolation. The custom mini-swe-agent adapter is identical across
arms; only its offline context projection changes.

Required:
  --dataset <path>          Pinned deep-swe/tasks directory
  --corpus <path>           Separate task corpus directory
  --mini-swe-version <ver>  Exact mini-swe-agent package version

Options:
  --tasks <ids>             Explicit comma-separated task ids
  --task-limit <count>      Deterministic SHA-256 sample size
  --sample-seed <integer>   Sampling and arm-order seed (default: 0)
  --attempts <count>        Rollouts per task and arm (default: 4)
  --arms <list>             baseline,rag,kontext (default: all)
  --model <provider/model>  Pier model route (default: openai/gpt-5.5)
  --reasoning <effort>      Fixed effort recorded for every arm (default: medium)
  --environment <name>      docker or modal (default: docker)
  --concurrency <count>     Concurrent Pier trials per arm (default: 1)
  --env-file <path>         Provider credentials loaded by Pier
  --deepswe-revision <sha>  Refuse a dataset revision mismatch
  --pier-revision <value>   Explicit Pier version/revision; otherwise --version
  --pier-bin <path>         Pier executable (default: pier)
  --run-dir <path>          Manifests, Pier jobs, and report directory
  --output <path>           Final report JSON path
  --dry-run                 Prepare and print immutable run manifests only
  -h, --help                Show this help
`;
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
});
