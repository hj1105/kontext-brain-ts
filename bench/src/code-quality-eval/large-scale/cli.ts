#!/usr/bin/env node
import { existsSync } from "node:fs";
import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { CodeQualityArm, CodeQualityRunConfig } from "../contracts.js";
import type { LargeScaleReport, LargeScaleRunConfig } from "./contracts.js";
import { renderLargeScaleMarkdown } from "./report.js";
import { runLargeScaleEvaluation } from "./runner.js";

interface CliOptions extends LargeScaleRunConfig {
  readonly outputPath: string;
  readonly resumePath?: string;
}

async function main(): Promise<void> {
  if (process.argv.includes("--help") || process.argv.includes("-h")) {
    process.stdout.write(helpText());
    return;
  }
  const repositoryRoot = fileURLToPath(new URL("../../../../", import.meta.url));
  const localEnvironment = path.join(repositoryRoot, ".env.local");
  if (existsSync(localEnvironment)) process.loadEnvFile(localEnvironment);
  const options = parseOptions(process.argv.slice(2), repositoryRoot);
  const previous = options.resumePath ? await loadReport(options.resumePath) : undefined;
  if (previous) assertCompatible(previous, options);
  const config: LargeScaleRunConfig = {
    runtime: options.runtime,
    model: options.model,
    reasoningEffort: options.reasoningEffort,
    repetitions: Math.max(options.repetitions, previous?.config.repetitions ?? 0),
    timeoutMilliseconds: options.timeoutMilliseconds,
    arms: [...new Set([...(previous?.config.arms ?? []), ...options.arms])],
  };
  await verifyPreconditions(repositoryRoot, config);
  const report = await runLargeScaleEvaluation({
    repositoryRoot,
    config,
    ...(previous ? { existingRuns: previous.runs.filter((run) => run.evaluationEligible) } : {}),
    dependencies: {
      onProgress: (message) => process.stderr.write(`${message}\n`),
    },
  });
  const outputPath = path.resolve(options.outputPath);
  const markdownPath = `${outputPath.replace(/\.json$/i, "")}.md`;
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  await writeFile(markdownPath, renderLargeScaleMarkdown(report), "utf8");
  process.stdout.write(
    `${JSON.stringify(
      {
        outputPath,
        markdownPath,
        evidenceStrength: report.evidenceStrength,
        summaries: report.summaries,
        comparisons: report.comparisons,
      },
      null,
      2,
    )}\n`,
  );
}

function parseOptions(args: readonly string[], repositoryRoot: string): CliOptions {
  let runtime: CodeQualityRunConfig["runtime"] = "codex";
  let model: string | undefined;
  let reasoningEffort: CodeQualityRunConfig["reasoningEffort"] = "medium";
  let repetitions = 1;
  let timeoutMilliseconds = 600_000;
  let arms: readonly CodeQualityArm[] = ["baseline", "rag", "kontext"];
  let outputPath = path.join(
    repositoryRoot,
    "bench",
    "data",
    "code-quality-eval",
    "large-scale-latest.json",
  );
  let resumePath: string | undefined;
  for (let index = 0; index < args.length; index += 1) {
    const option = args[index];
    const value = args[index + 1];
    if (!value) throw new Error(`Missing value for ${option}`);
    switch (option) {
      case "--runtime":
        if (value !== "codex" && value !== "claude") {
          throw new Error(`Unsupported runtime: ${value}`);
        }
        runtime = value;
        break;
      case "--model":
        model = value;
        break;
      case "--reasoning":
        if (!isReasoningEffort(value)) throw new Error(`Unsupported reasoning effort: ${value}`);
        reasoningEffort = value;
        break;
      case "--repetitions":
        repetitions = positiveInteger(value, option);
        break;
      case "--timeout-ms":
        timeoutMilliseconds = positiveInteger(value, option);
        break;
      case "--arms":
        arms = parseArms(value);
        break;
      case "--output":
        outputPath = value;
        break;
      case "--resume":
        resumePath = value;
        break;
      default:
        throw new Error(`Unknown option: ${option}`);
    }
    index += 1;
  }
  return {
    runtime,
    model: model ?? (runtime === "claude" ? "claude-opus-5" : "gpt-5.6-terra"),
    reasoningEffort,
    repetitions,
    timeoutMilliseconds,
    arms,
    outputPath,
    ...(resumePath ? { resumePath } : {}),
  };
}

async function loadReport(reportPath: string): Promise<LargeScaleReport> {
  const parsed = JSON.parse(await readFile(path.resolve(reportPath), "utf8")) as LargeScaleReport;
  if (parsed.benchmark !== "large-scale-code-quality" || parsed.schemaVersion !== 1) {
    throw new Error(`Not a large-scale code-quality report: ${reportPath}`);
  }
  return parsed;
}

function assertCompatible(previous: LargeScaleReport, options: LargeScaleRunConfig): void {
  for (const key of ["runtime", "model", "reasoningEffort"] as const) {
    if (previous.config[key] !== options[key]) {
      throw new Error(
        `Cannot resume with a different ${key}: ${previous.config[key]} != ${options[key]}`,
      );
    }
  }
}

function parseArms(value: string): readonly CodeQualityArm[] {
  const requested = value.split(",").map((arm) => arm.trim());
  if (requested.length === 0 || requested.some((arm) => !isArm(arm))) {
    throw new Error("--arms must be a comma-separated subset of baseline,rag,kontext");
  }
  return [...new Set(requested)] as CodeQualityArm[];
}

function isArm(value: string): value is CodeQualityArm {
  return value === "baseline" || value === "rag" || value === "kontext";
}

function isReasoningEffort(value: string): value is CodeQualityRunConfig["reasoningEffort"] {
  return value === "low" || value === "medium" || value === "high" || value === "xhigh";
}

function positiveInteger(value: string, option: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${option} must be a positive integer`);
  }
  return parsed;
}

async function verifyPreconditions(
  repositoryRoot: string,
  options: LargeScaleRunConfig,
): Promise<void> {
  const required = [
    path.join(repositoryRoot, "packages", "local", "dist", "cli.js"),
    path.join(repositoryRoot, "plugins", "kontext-brain", "server.mjs"),
  ];
  try {
    await Promise.all(required.map((artifact) => access(artifact)));
  } catch {
    throw new Error("Kontext artifacts are missing. Run `corepack pnpm build` first.");
  }
  if (options.arms.includes("rag") && !process.env.OPENAI_API_KEY?.trim()) {
    throw new Error(
      "The RAG arm needs OPENAI_API_KEY for embeddings. Omit it with --arms baseline,kontext to use subscriptions only.",
    );
  }
}

function helpText(): string {
  return `Usage: node --import tsx bench/src/code-quality-eval/large-scale/cli.ts [options]

Runs a repository-scale, paired code-generation benchmark. Generation uses the
authenticated Codex or Claude subscription; provider API keys are removed from
the agent process. The RAG arm alone uses OPENAI_API_KEY for embeddings.

Options:
  --runtime <name>        codex or claude (default: codex)
  --model <name>          model id (default: gpt-5.6-terra, or claude-opus-5)
  --reasoning <effort>    low, medium, high, or xhigh (default: medium)
  --arms <list>           baseline,rag,kontext subset (default: all three)
  --repetitions <count>   Paired repetitions (default: 1)
  --timeout-ms <ms>       Per-arm timeout (default: 600000)
  --output <path>         JSON output path
  --resume <path>         Replace rerun arms in an existing compatible report
  -h, --help              Show this help
`;
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
});
