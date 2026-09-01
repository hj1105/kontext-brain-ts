import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { GitChangeBundleIntegrator } from "../src/index.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("GitChangeBundleIntegrator", () => {
  it("applies isolated tracked and untracked bundle patches in declared order", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "kontext-integration-"));
    temporaryDirectories.push(root);
    const repositoryPath = path.join(root, "repository");
    await mkdir(repositoryPath);
    await writeFile(path.join(repositoryPath, "shared.ts"), "export const first = 1;\n");
    initializeGit(repositoryPath);
    const baseRevision = git(repositoryPath, ["rev-parse", "HEAD"]).trim();
    const firstSource = path.join(root, "source-a");
    const secondSource = path.join(root, "source-b");
    git(repositoryPath, ["worktree", "add", "--detach", firstSource, baseRevision]);
    git(repositoryPath, ["worktree", "add", "--detach", secondSource, baseRevision]);
    await writeFile(path.join(firstSource, "shared.ts"), "export const first = 2;\n");
    await writeFile(path.join(secondSource, "second.ts"), "export const second = 2;\n");

    const integrator = new GitChangeBundleIntegrator(repositoryPath, path.join(root, "integrated"));
    const workspace = await integrator.prepare({
      taskId: "task:integration",
      scheduleJobId: "schedule:one",
      baseRevision,
    });
    const result = await integrator.apply(workspace, [
      {
        bundleId: "bundle:a",
        workItemId: "work:a",
        sourceWorkspacePath: firstSource,
        baseRevision,
        changedPaths: ["shared.ts"],
      },
      {
        bundleId: "bundle:b",
        workItemId: "work:b",
        sourceWorkspacePath: secondSource,
        baseRevision,
        changedPaths: ["second.ts"],
      },
    ]);
    const reopened = await integrator.apply(workspace, [
      {
        bundleId: "bundle:a",
        workItemId: "work:a",
        sourceWorkspacePath: firstSource,
        baseRevision,
        changedPaths: ["shared.ts"],
      },
      {
        bundleId: "bundle:b",
        workItemId: "work:b",
        sourceWorkspacePath: secondSource,
        baseRevision,
        changedPaths: ["second.ts"],
      },
    ]);

    expect(result.appliedBundleIds).toEqual(["bundle:a", "bundle:b"]);
    expect(reopened).toEqual(result);
    expect(
      git(workspace.workspacePath, ["rev-list", "--count", `${baseRevision}..HEAD`]).trim(),
    ).toBe("2");
    expect(await integrator.diff(workspace)).toContain("second.ts");
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
