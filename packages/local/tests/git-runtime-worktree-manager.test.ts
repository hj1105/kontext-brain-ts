import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { GitRuntimeWorktreeManager } from "../src/index.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("GitRuntimeWorktreeManager", () => {
  it("creates one deterministic isolated branch and reopens it idempotently", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "kontext-runtime-worktree-"));
    temporaryDirectories.push(root);
    const repositoryPath = path.join(root, "repository");
    await mkdir(repositoryPath);
    await writeFile(path.join(repositoryPath, "README.md"), "baseline\n");
    initializeGit(repositoryPath);
    const baseRevision = git(repositoryPath, ["rev-parse", "HEAD"]).trim();
    const manager = new GitRuntimeWorktreeManager(repositoryPath, path.join(root, "worktrees"));
    const input = {
      taskId: "task:worktree",
      workItem: {
        workItemId: "work-item:handler",
        taskId: "task:worktree",
        plannedSymbolIds: ["symbol:handler"],
        dependsOn: [],
        allowedPaths: ["src/handler.ts"],
        requiredVerifiers: [],
        capabilityId: "capability:handler",
      },
      baseRevision,
    };

    const first = await manager.prepare(input);
    const reopened = await manager.prepare(input);

    expect(reopened).toEqual(first);
    expect(first.workspacePath).not.toBe(repositoryPath);
    expect(first.branchName).toMatch(/^codex\/kontext-/);
    expect(git(first.workspacePath, ["rev-parse", "HEAD"]).trim()).toBe(baseRevision);
  });
});

function initializeGit(repositoryPath: string): void {
  git(repositoryPath, ["init", "-q"]);
  git(repositoryPath, ["add", "."]);
  git(repositoryPath, [
    "-c",
    "user.name=Kontext Test",
    "-c",
    "user.email=kontext@example.invalid",
    "commit",
    "-qm",
    "baseline",
  ]);
}

function git(cwd: string, args: readonly string[]): string {
  return execFileSync("git", args, { cwd, encoding: "utf8" });
}
