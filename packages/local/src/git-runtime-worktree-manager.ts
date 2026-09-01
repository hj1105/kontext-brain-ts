import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { access, mkdir, realpath } from "node:fs/promises";
import path from "node:path";
import {
  type RuntimeWorktree,
  type RuntimeWorktreePort,
  deterministicWorktreeId,
} from "@kontext-brain/orchestrator";

export class GitRuntimeWorktreeManager implements RuntimeWorktreePort {
  constructor(
    private readonly repositoryPath: string,
    private readonly worktreeRoot: string,
  ) {}

  async prepare(input: Parameters<RuntimeWorktreePort["prepare"]>[0]): Promise<RuntimeWorktree> {
    const repositoryPath = await realpath(path.resolve(this.repositoryPath));
    const root = path.resolve(this.worktreeRoot);
    await mkdir(root, { recursive: true, mode: 0o700 });
    const repositoryRoot = (await runGit(repositoryPath, ["rev-parse", "--show-toplevel"])).trim();
    if ((await realpath(repositoryRoot)) !== repositoryPath) {
      throw new Error("Runtime worktree manager requires the Git repository root");
    }
    const suffix = createHash("sha256")
      .update(JSON.stringify([input.taskId, input.workItem.workItemId]))
      .digest("hex")
      .slice(0, 12);
    const branchName = `codex/kontext-${slug(input.taskId)}-${slug(
      input.workItem.workItemId,
    )}-${suffix}`;
    const workspacePath = path.join(root, suffix);
    if (!isWithin(root, workspacePath)) throw new Error("Runtime worktree path escapes its root");
    if (await exists(path.join(workspacePath, ".git"))) {
      const currentBranch = (
        await runGit(workspacePath, ["rev-parse", "--abbrev-ref", "HEAD"])
      ).trim();
      if (currentBranch !== branchName) {
        throw new Error(`Existing worktree uses unexpected branch ${currentBranch}`);
      }
      return runtimeWorktree(input, workspacePath, branchName);
    }
    const branchExists =
      (
        await runGit(
          repositoryPath,
          ["show-ref", "--verify", "--quiet", `refs/heads/${branchName}`],
          true,
        )
      ).exitCode === 0;
    if (branchExists) {
      await runGit(repositoryPath, ["worktree", "add", "--", workspacePath, branchName]);
    } else {
      await runGit(repositoryPath, [
        "worktree",
        "add",
        "-b",
        branchName,
        "--",
        workspacePath,
        input.baseRevision,
      ]);
    }
    return runtimeWorktree(input, workspacePath, branchName);
  }
}

function runtimeWorktree(
  input: Parameters<RuntimeWorktreePort["prepare"]>[0],
  workspacePath: string,
  branchName: string,
): RuntimeWorktree {
  return {
    worktreeId: deterministicWorktreeId(input.taskId, input.workItem.workItemId),
    workspacePath,
    branchName,
    baseRevision: input.baseRevision,
  };
}

async function exists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

function runGit(cwd: string, args: readonly string[], allowFailure?: false): Promise<string>;
function runGit(
  cwd: string,
  args: readonly string[],
  allowFailure: true,
): Promise<{ readonly exitCode: number; readonly stdout: string }>;
function runGit(
  cwd: string,
  args: readonly string[],
  allowFailure = false,
): Promise<string | { readonly exitCode: number; readonly stdout: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn("git", args, { cwd, shell: false, stdio: ["ignore", "pipe", "pipe"] });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
    child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
    child.once("error", reject);
    child.once("close", (code) => {
      const output = Buffer.concat(stdout).toString("utf8");
      if (code === 0) {
        resolve(allowFailure ? { exitCode: 0, stdout: output } : output);
      } else if (allowFailure) {
        resolve({ exitCode: code ?? 1, stdout: output });
      } else {
        reject(
          new Error(Buffer.concat(stderr).toString("utf8").trim() || `git exited with ${code}`),
        );
      }
    });
  });
}

function slug(value: string): string {
  const normalized = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 24);
  return normalized || "work";
}

function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative !== "" && relative !== ".." && !relative.startsWith(`..${path.sep}`);
}
