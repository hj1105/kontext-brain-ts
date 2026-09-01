#!/usr/bin/env node
import { access, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { CodeQualityRunConfig } from "./contracts.js";
import { runCodeQualityEvaluation } from "./harness.js";
import { renderCodeQualityMarkdown } from "./report.js";
import { codeQualityScenarios } from "./scenarios.js";

interface CliOptions extends CodeQualityRunConfig {
  readonly outputPath: string;
}

async function main(): Promise<void> {
  if (process.argv.includes("--help") || process.argv.includes("-h")) {
    process.stdout.write(helpText());
    return;
  }
  const repositoryRoot = fileURLToPath(new URL("../../../", import.meta.url));
  const options = parseOptions(process.argv.slice(2), repositoryRoot);
  await verifyBuiltArtifacts(repositoryRoot);
  const report = await runCodeQualityEvaluation({
    repositoryRoot,
    scenarios: codeQualityScenarios,
    config: options,
    dependencies: {
      onProgress: (message) => process.stderr.write(`${message}\n`),
    },
  });
  const outputPath = path.resolve(options.outputPath);
  const markdownPath = `${outputPath.replace(/\.json$/i, "")}.md`;
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  await writeFile(markdownPath, renderCodeQualityMarkdown(report), "utf8");
  process.stdout.write(
    `${JSON.stringify(
      {
        outputPath,
        markdownPath,
        verdict: report.verdict,
        evidenceStrength: report.evidenceStrength,
        hiddenAssertionUplift: report.hiddenAssertionUplift,
        taskSuccessUplift: report.taskSuccessUplift,
        paired: report.paired,
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
  let timeoutMilliseconds = 300_000;
  let outputPath = path.join(repositoryRoot, "bench", "data", "code-quality-eval", "latest.json");
  for (let index = 0; index < args.length; index += 1) {
    const option = args[index];
    const value = args[index + 1];
    if (!value) throw new Error(`Missing value for ${option}`);
    switch (option) {
      case "--runtime":
        if (!isRuntime(value)) throw new Error(`Unsupported runtime: ${value}`);
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
      case "--output":
        outputPath = value;
        break;
      default:
        throw new Error(`Unknown option: ${option}`);
    }
    index += 1;
  }
  return {
    runtime,
    model: model ?? defaultModel(runtime),
    reasoningEffort,
    repetitions,
    timeoutMilliseconds,
    outputPath,
  };
}

function positiveInteger(value: string, option: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${option} must be a positive integer`);
  }
  return parsed;
}

function isRuntime(value: string): value is CodeQualityRunConfig["runtime"] {
  return value === "codex" || value === "claude";
}

function defaultModel(runtime: CodeQualityRunConfig["runtime"]): string {
  return runtime === "claude" ? "claude-opus-5" : "gpt-5.6-terra";
}

function isReasoningEffort(value: string): value is CodeQualityRunConfig["reasoningEffort"] {
  return ["low", "medium", "high", "xhigh"].includes(value);
}

async function verifyBuiltArtifacts(repositoryRoot: string): Promise<void> {
  const required = [
    path.join(repositoryRoot, "packages", "local", "dist", "cli.js"),
    path.join(repositoryRoot, "plugins", "kontext-brain", "server.mjs"),
  ];
  try {
    await Promise.all(required.map((artifact) => access(artifact)));
  } catch {
    throw new Error("Kontext artifacts are missing. Run `pnpm build` before this evaluation.");
  }
}

function helpText(): string {
  return `Usage: pnpm --filter @kontext-brain/bench code-quality [options]

Runs paired code-generation tasks with and without Kontext Brain on the
selected subscription runtime.
API-key environment variables are removed so Codex uses its authenticated subscription.

Options:
  --runtime <name>        codex or claude (default: codex)
  --model <name>          model id (default: gpt-5.6-terra, or claude-opus-5)
  --reasoning <effort>    low, medium, high, or xhigh (default: medium)
  --repetitions <count>   Paired repetitions per scenario (default: 1)
  --timeout-ms <ms>       Timeout for each Codex execution (default: 300000)
  --output <path>         JSON output path (default: bench/data/code-quality-eval/latest.json)
  -h, --help              Show this help
`;
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
});
