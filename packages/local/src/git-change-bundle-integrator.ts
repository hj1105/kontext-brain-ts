import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { access, mkdir, realpath } from "node:fs/promises";
import path from "node:path";
import type {
  ChangeBundleIntegrationPatch,
  ChangeBundleIntegrationPort,
  ChangeBundleIntegrationResult,
  ChangeBundleIntegrationWorkspace,
} from "@kontext-brain/orchestrator";

const maxPatchBytes = 64 * 1024 * 1024;

export class GitChangeBundleIntegrator implements ChangeBundleIntegrationPort {
  constructor(
    private readonly repositoryPath: string,
    private readonly worktreeRoot: string,
  ) {}

  async prepare(input: {
    readonly taskId: string;
    readonly scheduleJobId: string;
    readonly baseRevision: string;
  }): Promise<ChangeBundleIntegrationWorkspace> {
    const repositoryPath = await realpath(path.resolve(this.repositoryPath));
    const repositoryRoot = (await git(repositoryPath, ["rev-parse", "--show-toplevel"])).trim();
    if ((await realpath(repositoryRoot)) !== repositoryPath) {
      throw new Error("Change Bundle integration requires the Git repository root");
    }
    const root = path.resolve(this.worktreeRoot);
    await mkdir(root, { recursive: true, mode: 0o700 });
    const suffix = createHash("sha256")
      .update(JSON.stringify([input.taskId, input.scheduleJobId]))
      .digest("hex")
      .slice(0, 12);
    const branchName = `codex/kontext-integrate-${slug(input.taskId)}-${suffix}`;
    const workspacePath = path.join(root, suffix);
    if (!isWithin(root, workspacePath))
      throw new Error("Integration worktree path escapes its root");
    if (await exists(path.join(workspacePath, ".git"))) {
      const currentBranch = (
        await git(workspacePath, ["rev-parse", "--abbrev-ref", "HEAD"])
      ).trim();
      if (currentBranch !== branchName) {
        throw new Error(`Existing integration worktree uses unexpected branch ${currentBranch}`);
      }
      return { workspacePath, branchName, baseRevision: input.baseRevision };
    }
    const branchExists =
      (
        await gitResult(repositoryPath, [
          "show-ref",
          "--verify",
          "--quiet",
          `refs/heads/${branchName}`,
        ])
      ).exitCode === 0;
    if (branchExists) {
      await git(repositoryPath, ["worktree", "add", "--", workspacePath, branchName]);
    } else {
      await git(repositoryPath, [
        "worktree",
        "add",
        "-b",
        branchName,
        "--",
        workspacePath,
        input.baseRevision,
      ]);
    }
    return { workspacePath, branchName, baseRevision: input.baseRevision };
  }

  async apply(
    workspace: ChangeBundleIntegrationWorkspace,
    patches: readonly ChangeBundleIntegrationPatch[],
  ): Promise<ChangeBundleIntegrationResult> {
    if ((await git(workspace.workspacePath, ["status", "--porcelain", "-z"])).length > 0) {
      throw new Error("Integration worktree must be clean before applying Change Bundles");
    }
    if (
      (
        await gitResult(workspace.workspacePath, [
          "merge-base",
          "--is-ancestor",
          workspace.baseRevision,
          "HEAD",
        ])
      ).exitCode !== 0
    ) {
      throw new Error("Integration worktree no longer descends from its immutable base revision");
    }
    const existingSubjects = (
      await git(workspace.workspacePath, [
        "log",
        "--reverse",
        "--format=%s",
        `${workspace.baseRevision}..HEAD`,
      ])
    )
      .split("\n")
      .filter(Boolean);
    if (
      existingSubjects.length > patches.length ||
      existingSubjects.some(
        (subject, index) => subject !== `Integrate ${patches[index]?.workItemId ?? ""}`,
      )
    ) {
      throw new Error("Integration worktree contains commits outside the declared Change Bundles");
    }
    const commonDirectory = await canonicalGitCommonDirectory(workspace.workspacePath);
    const appliedBundleIds = patches
      .slice(0, existingSubjects.length)
      .map((patch) => patch.bundleId);
    for (const patch of patches.slice(existingSubjects.length)) {
      if ((await canonicalGitCommonDirectory(patch.sourceWorkspacePath)) !== commonDirectory) {
        throw new Error(`Change Bundle ${patch.bundleId} source belongs to another repository`);
      }
      const patchBytes = await createPatch(patch);
      if (patchBytes.length === 0) {
        throw new Error(`Change Bundle ${patch.bundleId} produced an empty Git patch`);
      }
      await git(
        workspace.workspacePath,
        ["apply", "--index", "--3way", "--binary", "-"],
        patchBytes,
      );
      const stagedPaths = splitZero(
        await git(workspace.workspacePath, ["diff", "--cached", "--name-only", "-z"]),
      );
      if (!sameStrings(stagedPaths.map(canonicalPath), patch.changedPaths.map(canonicalPath))) {
        throw new Error(`Applied paths do not match Change Bundle ${patch.bundleId}`);
      }
      await git(workspace.workspacePath, [
        "-c",
        "user.name=Kontext Brain",
        "-c",
        "user.email=kontext-brain@invalid.local",
        "commit",
        "--no-gpg-sign",
        "-m",
        `Integrate ${patch.workItemId}`,
      ]);
      appliedBundleIds.push(patch.bundleId);
    }
    return {
      ...workspace,
      gitCommit: (await git(workspace.workspacePath, ["rev-parse", "HEAD"])).trim(),
      appliedBundleIds,
    };
  }

  async diff(workspace: ChangeBundleIntegrationWorkspace): Promise<string> {
    return git(workspace.workspacePath, [
      "diff",
      "--binary",
      "--full-index",
      `${workspace.baseRevision}..HEAD`,
      "--",
    ]);
  }
}

async function createPatch(patch: ChangeBundleIntegrationPatch): Promise<Buffer> {
  const paths = patch.changedPaths.map(canonicalPath);
  if (paths.length === 0 || paths.some((value) => !isSafeRelativePath(value))) {
    throw new Error(`Change Bundle ${patch.bundleId} has unsafe or empty changed paths`);
  }
  const tracked = await gitBuffer(patch.sourceWorkspacePath, [
    "diff",
    "--binary",
    "--full-index",
    patch.baseRevision,
    "--",
    ...paths,
  ]);
  const untracked = splitZero(
    await git(patch.sourceWorkspacePath, [
      "ls-files",
      "--others",
      "--exclude-standard",
      "-z",
      "--",
      ...paths,
    ]),
  );
  const additions: Buffer[] = [];
  for (const relativePath of untracked) {
    const result = await gitResult(
      patch.sourceWorkspacePath,
      ["diff", "--no-index", "--binary", "--full-index", "--", "/dev/null", relativePath],
      undefined,
    );
    if (result.exitCode !== 1 || result.stdout.length === 0) {
      throw new Error(`Cannot create untracked patch for ${relativePath}`);
    }
    additions.push(result.stdout);
  }
  const combined = Buffer.concat([tracked, ...additions]);
  if (combined.length > maxPatchBytes) {
    throw new Error(`Change Bundle ${patch.bundleId} patch exceeds ${maxPatchBytes} bytes`);
  }
  return combined;
}

async function canonicalGitCommonDirectory(workspacePath: string): Promise<string> {
  const raw = (await git(workspacePath, ["rev-parse", "--git-common-dir"])).trim();
  return realpath(path.resolve(workspacePath, raw));
}

async function git(cwd: string, args: readonly string[], stdin?: Buffer): Promise<string> {
  const result = await gitResult(cwd, args, stdin);
  if (result.exitCode !== 0) {
    throw new Error(
      result.stderr.trim() || `git ${args[0] ?? "command"} exited ${result.exitCode}`,
    );
  }
  return result.stdout.toString("utf8");
}

async function gitBuffer(cwd: string, args: readonly string[]): Promise<Buffer> {
  const result = await gitResult(cwd, args);
  if (result.exitCode !== 0) {
    throw new Error(
      result.stderr.trim() || `git ${args[0] ?? "command"} exited ${result.exitCode}`,
    );
  }
  return result.stdout;
}

function gitResult(
  cwd: string,
  args: readonly string[],
  stdin?: Buffer,
): Promise<{ readonly exitCode: number; readonly stdout: Buffer; readonly stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn("git", args, { cwd, shell: false, stdio: ["pipe", "pipe", "pipe"] });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    let bytes = 0;
    const capture = (target: Buffer[], chunk: Buffer): void => {
      bytes += chunk.byteLength;
      if (bytes > maxPatchBytes) {
        child.kill("SIGKILL");
        reject(new Error(`Git integration output exceeds ${maxPatchBytes} bytes`));
        return;
      }
      target.push(chunk);
    };
    child.stdout.on("data", (chunk: Buffer) => capture(stdout, chunk));
    child.stderr.on("data", (chunk: Buffer) => capture(stderr, chunk));
    child.once("error", reject);
    child.once("close", (code) =>
      resolve({
        exitCode: code ?? 1,
        stdout: Buffer.concat(stdout),
        stderr: Buffer.concat(stderr).toString("utf8"),
      }),
    );
    if (stdin) child.stdin.end(stdin);
    else child.stdin.end();
  });
}

async function exists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

function splitZero(value: string): string[] {
  return value.split("\u0000").filter(Boolean);
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  const normalize = (values: readonly string[]) => Array.from(new Set(values)).sort();
  return JSON.stringify(normalize(left)) === JSON.stringify(normalize(right));
}

function canonicalPath(value: string): string {
  return value.replaceAll("\\", "/").replace(/^\.\//, "");
}

function isSafeRelativePath(value: string): boolean {
  return value.length > 0 && !path.isAbsolute(value) && value !== ".." && !value.startsWith("../");
}

function slug(value: string): string {
  const normalized = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 24);
  return normalized || "task";
}

function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative !== "" && relative !== ".." && !relative.startsWith(`..${path.sep}`);
}
