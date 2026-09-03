import { createHash } from "node:crypto";
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { runWorkspaceCommand } from "../workspace.js";
import type { RealOssGrade, RealOssTask, RealOssWorkspace } from "./contracts.js";

export async function resolveRealOssSource(input: {
  readonly task: RealOssTask;
  readonly cacheDirectory: string;
  readonly sourceRepositoryPath?: string;
}): Promise<string> {
  if (input.sourceRepositoryPath) {
    const source = path.resolve(input.sourceRepositoryPath);
    await validateRealOssSource(input.task, source);
    return source;
  }

  await mkdir(input.cacheDirectory, { recursive: true });
  const source = path.join(input.cacheDirectory, input.task.instanceId);
  if (!(await pathExists(path.join(source, ".git")))) {
    const clone = await runWorkspaceCommand(input.cacheDirectory, "git", [
      "clone",
      "--no-checkout",
      input.task.repositoryUrl,
      source,
    ]);
    requireSuccess(clone, `clone ${input.task.repository}`);
  }
  if (!(await commitExists(source, input.task.baseCommit))) {
    const fetch = await runWorkspaceCommand(source, "git", [
      "fetch",
      "origin",
      input.task.baseCommit,
    ]);
    requireSuccess(fetch, `fetch ${input.task.baseCommit}`);
  }
  await validateRealOssSource(input.task, source);
  return source;
}

export async function validateRealOssSource(task: RealOssTask, source: string): Promise<void> {
  const repository = await runWorkspaceCommand(source, "git", ["rev-parse", "--git-dir"]);
  requireSuccess(repository, `open source repository ${source}`);
  if (!(await commitExists(source, task.baseCommit))) {
    throw new Error(`Source repository does not contain base commit ${task.baseCommit}`);
  }
  for (const expected of task.sourceIntegrity) {
    const file = await runWorkspaceCommand(source, "git", [
      "show",
      `${task.baseCommit}:${expected.relativePath}`,
    ]);
    requireSuccess(file, `read ${expected.relativePath} at ${task.baseCommit}`);
    const actual = sha256(file.stdout);
    if (actual !== expected.sha256) {
      throw new Error(
        `Source integrity mismatch for ${expected.relativePath}: ${actual} != ${expected.sha256}`,
      );
    }
  }
  const hiddenDigest = sha256(task.hiddenTest.patch);
  if (hiddenDigest !== task.hiddenTest.patchSha256) {
    throw new Error(
      `Hidden test patch integrity mismatch: ${hiddenDigest} != ${task.hiddenTest.patchSha256}`,
    );
  }
}

/**
 * Fetches only the pinned base commit into a fresh repository. This preserves
 * the authentic upstream revision while ensuring the agent cannot inspect the
 * later fix through branches, tags, remotes, reflogs, or local object history.
 */
export async function createRealOssWorkspace(input: {
  readonly task: RealOssTask;
  readonly sourceRepositoryPath: string;
  readonly installDependencies?: boolean;
}): Promise<RealOssWorkspace> {
  const workspacePath = await mkdtemp(
    path.join(tmpdir(), `kontext-real-oss-${input.task.instanceId}-`),
  );
  try {
    requireSuccess(
      await runWorkspaceCommand(workspacePath, "git", ["init", "-q"]),
      "initialize isolated repository",
    );
    requireSuccess(
      await runWorkspaceCommand(workspacePath, "git", [
        "fetch",
        "--quiet",
        "--depth=1",
        input.sourceRepositoryPath,
        input.task.baseCommit,
      ]),
      `fetch isolated base ${input.task.baseCommit}`,
    );
    requireSuccess(
      await runWorkspaceCommand(workspacePath, "git", [
        "checkout",
        "--detach",
        "--quiet",
        "FETCH_HEAD",
      ]),
      "checkout isolated base",
    );
    // FETCH_HEAD records the local cache path used for the one-commit fetch.
    // Removing it closes the last breadcrumb an agent could use to discover the
    // cache repository, whose history legitimately contains the later fix.
    await rm(path.join(workspacePath, ".git", "FETCH_HEAD"), { force: true });
    const revision = await runWorkspaceCommand(workspacePath, "git", ["rev-parse", "HEAD"]);
    requireSuccess(revision, "read isolated base revision");
    const baseRevision = revision.stdout.trim();
    if (baseRevision !== input.task.baseCommit) {
      throw new Error(`Isolated revision mismatch: ${baseRevision} != ${input.task.baseCommit}`);
    }
    if (input.installDependencies !== false) {
      await installRealOssDependencies(input.task, workspacePath);
    }
    return { workspacePath, baseRevision };
  } catch (error) {
    await rm(workspacePath, { recursive: true, force: true });
    throw error;
  }
}

async function installRealOssDependencies(task: RealOssTask, workspacePath: string): Promise<void> {
  requireSuccess(
    await runWorkspaceCommand(workspacePath, "uv", [
      "venv",
      "--python",
      task.environment.pythonVersion,
      ".venv",
    ]),
    `create Python ${task.environment.pythonVersion} environment`,
  );
  requireSuccess(
    await runWorkspaceCommand(
      workspacePath,
      "uv",
      ["pip", "install", "--python", ".venv/bin/python", "-e", ".", ...task.environment.packages],
      { ...process.env, UV_LINK_MODE: "copy" },
    ),
    "install pinned real-OSS dependencies",
  );
  const status = await changedFiles(workspacePath);
  if (status.length > 0) {
    throw new Error(`Dependency setup modified tracked benchmark files: ${status.join(", ")}`);
  }
}

export async function gradeRealOssWorkspace(
  task: RealOssTask,
  workspace: RealOssWorkspace,
): Promise<RealOssGrade> {
  const changed = await changedFiles(workspace.workspacePath);
  const candidatePatch = await runWorkspaceCommand(workspace.workspacePath, "git", [
    "diff",
    "--binary",
    "HEAD",
    "--",
  ]);
  requireSuccess(candidatePatch, "capture candidate patch");
  const publicTests = await runWorkspaceCommand(
    workspace.workspacePath,
    task.publicTest.command,
    task.publicTest.args,
  );
  const patchDirectory = await mkdtemp(path.join(tmpdir(), "kontext-real-oss-hidden-"));
  const patchPath = path.join(patchDirectory, "hidden-tests.diff");
  await writeFile(patchPath, task.hiddenTest.patch, { encoding: "utf8", mode: 0o600 });
  let hiddenPatchApplied = false;
  let failToPassPassed = 0;
  let passToPassPassed = 0;
  const hiddenFailures: string[] = [];
  try {
    const check = await runWorkspaceCommand(workspace.workspacePath, "git", [
      "apply",
      "--check",
      patchPath,
    ]);
    if (check.exitCode !== 0) {
      hiddenFailures.push(`hidden test patch did not apply: ${diagnostic(check)}`);
    } else {
      const apply = await runWorkspaceCommand(workspace.workspacePath, "git", ["apply", patchPath]);
      if (apply.exitCode !== 0) {
        hiddenFailures.push(`hidden test patch failed: ${diagnostic(apply)}`);
      } else {
        hiddenPatchApplied = true;
        const failToPass = await runHiddenTestSelection(
          task,
          workspace.workspacePath,
          task.hiddenTest.failToPass,
        );
        const passToPass = await runHiddenTestSelection(
          task,
          workspace.workspacePath,
          task.hiddenTest.passToPass,
        );
        failToPassPassed = failToPass.passed;
        passToPassPassed = passToPass.passed;
        if (!failToPass.complete) hiddenFailures.push(`FAIL_TO_PASS: ${failToPass.diagnostic}`);
        if (!passToPass.complete) hiddenFailures.push(`PASS_TO_PASS: ${passToPass.diagnostic}`);
      }
    }
  } finally {
    await rm(patchDirectory, { recursive: true, force: true });
  }

  return {
    publicTestsPassed: publicTests.exitCode === 0,
    targetChanged: task.targets.some((target) => changed.includes(target.relativePath)),
    allowedPathsOnly: changed.every((file) => task.allowedPaths.includes(file)),
    changedFiles: changed,
    failToPassPassed,
    failToPassTotal: task.hiddenTest.failToPass.length,
    passToPassPassed,
    passToPassTotal: task.hiddenTest.passToPass.length,
    hiddenPatchApplied,
    hiddenFailures,
    patch: candidatePatch.stdout,
  };
}

async function runHiddenTestSelection(
  task: RealOssTask,
  workspacePath: string,
  selectors: readonly string[],
): Promise<{ readonly passed: number; readonly complete: boolean; readonly diagnostic: string }> {
  if (selectors.length === 0) return { passed: 0, complete: true, diagnostic: "" };
  const runner = task.hiddenTest.runner;
  const result = await runWorkspaceCommand(workspacePath, runner.command, [
    ...runner.args,
    ...selectors,
  ]);
  if (result.exitCode === 0) {
    return { passed: selectors.length, complete: true, diagnostic: "" };
  }
  const output = `${result.stdout}\n${result.stderr}`.trim();
  const passed =
    runner.kind === "pytest-selectors"
      ? parsePytestPassed(output)
      : parseDjangoPassed(output, selectors.length);
  return {
    passed,
    complete: false,
    diagnostic: output.slice(-4_000),
  };
}

function parsePytestPassed(output: string): number {
  const passed = [...output.matchAll(/(?:^|\s)(\d+) passed(?:,|\s|$)/g)].at(-1)?.[1];
  return passed ? Number(passed) : 0;
}

export function parseDjangoPassed(output: string, requested: number): number {
  const ran = [...output.matchAll(/Ran (\d+) tests?/g)].at(-1)?.[1];
  if (!ran) return 0;
  const summary = output.match(/FAILED \(([^)]*)\)/)?.[1] ?? "";
  const unsuccessful = ["failures", "errors", "skipped", "unexpected successes"]
    .map((label) => Number(summary.match(new RegExp(`(?:^|, )${label}=(\\d+)`))?.[1] ?? 0))
    .reduce((total, count) => total + count, 0);
  return Math.max(0, Math.min(requested, Number(ran) - unsuccessful));
}

async function changedFiles(workspacePath: string): Promise<readonly string[]> {
  const tracked = await runWorkspaceCommand(workspacePath, "git", [
    "diff",
    "--name-only",
    "HEAD",
    "--",
  ]);
  requireSuccess(tracked, "read changed tracked files");
  const untracked = await runWorkspaceCommand(workspacePath, "git", [
    "ls-files",
    "--others",
    "--exclude-standard",
  ]);
  requireSuccess(untracked, "read untracked files");
  return [...new Set([...lines(tracked.stdout), ...lines(untracked.stdout)])].sort();
}

async function commitExists(repositoryPath: string, commit: string): Promise<boolean> {
  const result = await runWorkspaceCommand(repositoryPath, "git", [
    "cat-file",
    "-e",
    `${commit}^{commit}`,
  ]);
  return result.exitCode === 0;
}

function lines(value: string): readonly string[] {
  return value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function diagnostic(result: { readonly stdout: string; readonly stderr: string }): string {
  return `${result.stderr}\n${result.stdout}`.trim().slice(-4_000);
}

function requireSuccess(
  result: { readonly exitCode: number; readonly stdout: string; readonly stderr: string },
  operation: string,
): void {
  if (result.exitCode !== 0) throw new Error(`Cannot ${operation}: ${diagnostic(result)}`);
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

async function pathExists(value: string): Promise<boolean> {
  try {
    await access(value);
    return true;
  } catch {
    return false;
  }
}

export async function readWorkspaceFile(
  workspacePath: string,
  relativePath: string,
): Promise<string> {
  return readFile(path.join(workspacePath, relativePath), "utf8");
}
