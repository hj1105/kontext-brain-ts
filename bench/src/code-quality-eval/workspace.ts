import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { promisify } from "node:util";
import type { CodeQualityScenario, HiddenEvaluationResult } from "./contracts.js";

const execFileAsync = promisify(execFile);

export interface ScenarioWorkspace {
  readonly workspacePath: string;
  readonly baseRevision: string;
}

export interface WorkspaceCommandResult {
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
}

export async function createScenarioWorkspace(
  scenario: CodeQualityScenario,
): Promise<ScenarioWorkspace> {
  const workspacePath = await mkdtemp(
    path.join(tmpdir(), `kontext-code-eval-${scenario.scenarioId}-`),
  );
  await mkdir(path.join(workspacePath, "src"), { recursive: true });
  await mkdir(path.join(workspacePath, "test"), { recursive: true });
  await writeFile(
    path.join(workspacePath, "package.json"),
    `${JSON.stringify(
      {
        name: `code-quality-${scenario.scenarioId}`,
        version: "1.0.0",
        private: true,
        type: "module",
        scripts: { test: "node --test" },
      },
      null,
      2,
    )}\n`,
  );
  await writeFile(path.join(workspacePath, scenario.sourceFile), scenario.initialSource);
  await writeFile(path.join(workspacePath, "test", "public.test.js"), scenario.publicTestSource);
  await writeFile(
    path.join(workspacePath, "TASK.md"),
    `# Task\n\n${scenario.publicPrompt}\n\nOnly edit \`${scenario.sourceFile}\`.\n`,
  );
  await git(workspacePath, ["init", "-q"]);
  await git(workspacePath, ["add", "."]);
  await git(workspacePath, [
    "-c",
    "user.name=Kontext Code Eval",
    "-c",
    "user.email=code-eval@example.invalid",
    "commit",
    "-qm",
    "fixture baseline",
  ]);
  return {
    workspacePath,
    baseRevision: (await git(workspacePath, ["rev-parse", "HEAD"])).stdout.trim(),
  };
}

export async function evaluateWorkspace(
  scenario: CodeQualityScenario,
  workspacePath: string,
): Promise<{
  readonly publicTestsPassed: boolean;
  readonly hidden: HiddenEvaluationResult;
  readonly source: string;
  readonly changedPaths: readonly string[];
  readonly patch: string;
}> {
  const publicTests = await runWorkspaceCommand(workspacePath, "npm", [
    "test",
    "--",
    "--test-reporter=spec",
  ]);
  const sourcePath = path.join(workspacePath, scenario.sourceFile);
  const source = await readFile(sourcePath, "utf8").catch(() => "");
  let hidden: HiddenEvaluationResult;
  try {
    const moduleUrl = `${pathToFileURL(sourcePath).href}?eval=${Date.now()}-${Math.random()}`;
    const loaded = (await import(moduleUrl)) as Readonly<Record<string, unknown>>;
    hidden = await scenario.evaluateHidden(loaded);
  } catch (error) {
    hidden = {
      assertions: [
        {
          assertionId: "module-load",
          passed: false,
          diagnostic: error instanceof Error ? error.message : String(error),
        },
      ],
    };
  }
  const changedPaths = (await git(workspacePath, ["status", "--short"])).stdout
    .split("\n")
    .map((line) => line.slice(3).trim())
    .filter(Boolean)
    .sort();
  const patch = (await git(workspacePath, ["diff", "--binary", "HEAD", "--"])).stdout;
  return {
    publicTestsPassed: publicTests.exitCode === 0,
    hidden,
    source,
    changedPaths,
    patch,
  };
}

export async function runWorkspaceCommand(
  cwd: string,
  command: string,
  args: readonly string[],
  environment: NodeJS.ProcessEnv = process.env,
): Promise<WorkspaceCommandResult> {
  try {
    const result = await execFileAsync(command, args, {
      cwd,
      env: environment,
      encoding: "utf8",
      maxBuffer: 10 * 1024 * 1024,
    });
    return { exitCode: 0, stdout: result.stdout, stderr: result.stderr };
  } catch (error) {
    if (isExecError(error)) {
      return {
        exitCode: typeof error.code === "number" ? error.code : 1,
        stdout: error.stdout ?? "",
        stderr: error.stderr ?? error.message,
      };
    }
    throw error;
  }
}

async function git(cwd: string, args: readonly string[]): Promise<WorkspaceCommandResult> {
  return runWorkspaceCommand(cwd, "git", args);
}

function isExecError(error: unknown): error is Error & {
  readonly code?: number | string;
  readonly stdout?: string;
  readonly stderr?: string;
} {
  return error instanceof Error;
}
