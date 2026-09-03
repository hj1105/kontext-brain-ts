#!/usr/bin/env node
import { existsSync } from "node:fs";
import { access, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { CodeQualityArm, CodeQualityRunConfig } from "../contracts.js";
import type { RealOssRunConfig } from "./contracts.js";
import { flaskBlueprintNameTask } from "./manifest.js";
import { renderRealOssMarkdown } from "./report.js";
import { runRealOssEvaluation } from "./runner.js";

interface CliOptions extends RealOssRunConfig {
  readonly outputPath: string;
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
  await verifyBuiltArtifacts(repositoryRoot);
  const report = await runRealOssEvaluation({
    repositoryRoot,
    task: flaskBlueprintNameTask,
    config: options,
    dependencies: {
      onProgress: (message) => process.stderr.write(`${message}\n`),
    },
  });
  const outputPath = path.resolve(options.outputPath);
  const markdownPath = `${outputPath.replace(/\.json$/i, "")}.md`;
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  await writeFile(markdownPath, renderRealOssMarkdown(report), "utf8");
  process.stdout.write(
    `${JSON.stringify(
      {
        outputPath,
        markdownPath,
        evidenceStrength: report.evidenceStrength,
        task: report.task,
        summaries: report.summaries,
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
  let sourceRepositoryPath: string | undefined;
  let cacheDirectory = path.join(tmpdir(), "kontext-real-oss-source-cache");
  let outputPath = path.join(
    repositoryRoot,
    "bench",
    "data",
    "code-quality-eval",
    "real-oss-latest.json",
  );

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
        if (!isReasoningEffort(value)) {
          throw new Error(`Unsupported reasoning effort: ${value}`);
        }
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
      case "--source":
        sourceRepositoryPath = value;
        break;
      case "--cache-dir":
        cacheDirectory = value;
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
    model: model ?? (runtime === "claude" ? "claude-opus-5" : "gpt-5.6-terra"),
    reasoningEffort,
    repetitions,
    timeoutMilliseconds,
    arms,
    cacheDirectory: path.resolve(cacheDirectory),
    ...(sourceRepositoryPath ? { sourceRepositoryPath: path.resolve(sourceRepositoryPath) } : {}),
    outputPath,
  };
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

async function verifyBuiltArtifacts(repositoryRoot: string): Promise<void> {
  const required = [
    path.join(repositoryRoot, "packages", "code", "dist", "index.js"),
    path.join(repositoryRoot, "packages", "local", "dist", "cli.js"),
    path.join(repositoryRoot, "plugins", "kontext-brain", "server.mjs"),
  ];
  try {
    await Promise.all(required.map((artifact) => access(artifact)));
  } catch {
    throw new Error("Kontext artifacts are missing. Run `corepack pnpm build` first.");
  }
}

function helpText(): string {
  return `Usage: node --import tsx bench/src/code-quality-eval/real-oss/cli.ts [options]

Runs a paired benchmark against the real pallets/flask repository at the
SWE-bench Verified base commit for pallets__flask-5014. The source fix is never
shown to agents; the upstream regression test patch is applied only by grading.

Options:
  --runtime <name>        codex or claude (default: codex)
  --model <name>          model id (default: gpt-5.6-terra, or claude-opus-5)
  --reasoning <effort>    low, medium, high, or xhigh (default: medium)
  --arms <list>           baseline,rag,kontext subset (default: all three)
  --repetitions <count>   Paired repetitions (default: 1)
  --timeout-ms <ms>       Per-arm timeout (default: 600000)
  --source <path>         Existing local pallets/flask clone containing the base commit
  --cache-dir <path>      Clone cache used when --source is omitted
  --output <path>         JSON output path
  -h, --help              Show this help
`;
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
});
