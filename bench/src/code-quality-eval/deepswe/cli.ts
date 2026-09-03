#!/usr/bin/env node
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { codexSubscriptionEnvironment } from "../codex-runner.js";
import { runWorkspaceCommand } from "../workspace.js";
import type { DeepSweArm, DeepSwePrepareOptions, DeepSweRuntime } from "./contracts.js";
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
  readonly allowApiBilling: boolean;
}

async function main(): Promise<void> {
  if (process.argv.includes("--help") || process.argv.includes("-h")) {
    process.stdout.write(helpText());
    return;
  }
  const repositoryRoot = fileURLToPath(new URL("../../../../", import.meta.url));
  const options = parseOptions(process.argv.slice(2), repositoryRoot);
  const codexVersion =
    options.runtime === "codex-subscription"
      ? await resolveCodexSubscription(options.codexVersion)
      : undefined;
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
    runtime: options.runtime,
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
    ...(codexVersion ? { codexVersion } : {}),
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
  let runtime: DeepSweRuntime = "codex-subscription";
  let model: string | undefined;
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
  let codexVersion: string | undefined;
  let deepSweRevision: string | undefined;
  let declaredPierRevision: string | undefined;
  let allowApiBilling = false;
  let dryRun = false;
  for (let index = 0; index < args.length; index += 1) {
    const option = args[index];
    if (option === "--dry-run") {
      dryRun = true;
      continue;
    }
    if (option === "--allow-api-billing") {
      allowApiBilling = true;
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
      case "--codex-version":
        codexVersion = value;
        break;
      case "--runtime":
        if (value !== "codex-subscription" && value !== "mini-swe-api") {
          throw new Error("--runtime must be codex-subscription or mini-swe-api");
        }
        runtime = value;
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
  if (runtime === "mini-swe-api" && !allowApiBilling) {
    throw new Error("--runtime mini-swe-api requires explicit --allow-api-billing");
  }
  if (runtime === "mini-swe-api" && !miniSweAgentVersion) {
    throw new Error("--mini-swe-version is required for the API runtime");
  }
  if (runtime === "codex-subscription" && envFile) {
    throw new Error("--env-file is forbidden for the Codex subscription runtime");
  }
  const resolvedModel = model ?? (runtime === "codex-subscription" ? "gpt-5.5" : "openai/gpt-5.5");
  if (runtime === "mini-swe-api" && !resolvedModel.includes("/")) {
    throw new Error("API runtime model must use provider/model format");
  }
  if (runtime === "codex-subscription" && resolvedModel.includes("/")) {
    throw new Error("Codex subscription model must use a bare model name");
  }
  return {
    datasetTasksPath,
    corpusRoot,
    runDirectory,
    pierBinary,
    runtime,
    model: resolvedModel,
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
    ...(codexVersion ? { codexVersion } : {}),
    ...(deepSweRevision ? { deepSweRevision } : {}),
    outputPath: outputPath ?? path.join(runDirectory, "report.json"),
    dryRun,
    allowApiBilling,
    ...(declaredPierRevision ? { declaredPierRevision } : {}),
  };
}

async function resolveCodexSubscription(declaredVersion?: string): Promise<string> {
  const environment = codexSubscriptionEnvironment(process.env);
  const [status, version] = await Promise.all([
    runWorkspaceCommand(process.cwd(), "codex", ["login", "status"], environment),
    runWorkspaceCommand(process.cwd(), "codex", ["--version"], environment),
  ]);
  const statusText = `${status.stdout}\n${status.stderr}`;
  if (status.exitCode !== 0 || !/Logged in using ChatGPT/i.test(statusText)) {
    throw new Error(
      "Codex subscription runtime requires `codex login status` to report ChatGPT login",
    );
  }
  const match = `${version.stdout}\n${version.stderr}`.match(/codex-cli\s+([^\s]+)/i);
  const actualVersion = match?.[1];
  if (version.exitCode !== 0 || !actualVersion) {
    throw new Error(`Cannot resolve Codex CLI version: ${version.stderr || version.stdout}`);
  }
  if (declaredVersion && declaredVersion !== actualVersion) {
    throw new Error(`Codex version mismatch: ${actualVersion} != ${declaredVersion}`);
  }
  return actualVersion;
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
and verifier isolation. It defaults to the Codex CLI authenticated with the
user's ChatGPT subscription; only the offline context projection changes.

Required:
  --dataset <path>          Pinned deep-swe/tasks directory
  --corpus <path>           Separate task corpus directory

Options:
  --runtime <name>          codex-subscription (default) or mini-swe-api
  --codex-version <ver>     Refuse a Codex CLI version mismatch
  --mini-swe-version <ver>  Required only for mini-swe-api
  --allow-api-billing       Required acknowledgement for mini-swe-api
  --tasks <ids>             Explicit comma-separated task ids
  --task-limit <count>      Deterministic SHA-256 sample size
  --sample-seed <integer>   Sampling and arm-order seed (default: 0)
  --attempts <count>        Rollouts per task and arm (default: 4)
  --arms <list>             baseline,rag,kontext (default: all)
  --model <name>            Codex model, or provider/model for mini-swe-api
  --reasoning <effort>      Fixed effort recorded for every arm (default: medium)
  --environment <name>      docker or modal (default: docker)
  --concurrency <count>     Concurrent Pier trials per arm (default: 1)
  --env-file <path>         API credentials; forbidden for codex-subscription
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
