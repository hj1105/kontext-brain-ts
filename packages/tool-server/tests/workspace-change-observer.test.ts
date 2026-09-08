import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  captureWorkspaceSnapshot,
  changedPathsBetween,
  observeWorkspacePatch,
} from "../src/index.js";

const temporaryDirectories: string[] = [];
let workspacePath = "";

beforeEach(async () => {
  const root = await mkdtemp(path.join(tmpdir(), "kontext-workspace-observer-"));
  temporaryDirectories.push(root);
  workspacePath = path.join(root, "workspace");
  await mkdir(path.join(workspacePath, "src"), { recursive: true });
  await writeFile(path.join(workspacePath, "src", "handler.ts"), "export const handler = 1;\n");
  await writeFile(path.join(workspacePath, "src", "outside.ts"), "export const outside = 1;\n");
  execFileSync("git", ["init", "-q"], { cwd: workspacePath });
  execFileSync("git", ["add", "."], { cwd: workspacePath });
  execFileSync(
    "git",
    [
      "-c",
      "user.name=Kondex Test",
      "-c",
      "user.email=kondex@example.invalid",
      "commit",
      "-qm",
      "baseline",
    ],
    { cwd: workspacePath },
  );
});

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("workspace change observation", () => {
  it("limits a clean snapshot to authorized paths without exposing Git metadata", async () => {
    const snapshot = await captureWorkspaceSnapshot(workspacePath, ["src/handler.ts"]);

    expect(snapshot.files).toEqual([
      expect.objectContaining({ path: "src/handler.ts", kind: "file" }),
    ]);
  });

  it("includes dirty tracked and untracked paths outside authorization", async () => {
    const before = await captureWorkspaceSnapshot(workspacePath, ["src/handler.ts"]);
    await writeFile(path.join(workspacePath, "src", "outside.ts"), "export const outside = 2;\n");
    await writeFile(path.join(workspacePath, "src", "untracked.ts"), "export const fresh = 1;\n");
    const after = await captureWorkspaceSnapshot(workspacePath, ["src/handler.ts"]);

    expect(after.files.map((file) => file.path)).toEqual([
      "src/handler.ts",
      "src/outside.ts",
      "src/untracked.ts",
    ]);
    expect(changedPathsBetween(before, after)).toEqual(["src/outside.ts", "src/untracked.ts"]);
  });

  it("changes the revision but not user paths when only Git HEAD advances", async () => {
    const before = await captureWorkspaceSnapshot(workspacePath, ["src/handler.ts"]);
    execFileSync(
      "git",
      [
        "-c",
        "user.name=Kondex Test",
        "-c",
        "user.email=kondex@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "advance head",
      ],
      { cwd: workspacePath },
    );
    const after = await captureWorkspaceSnapshot(workspacePath, ["src/handler.ts"]);

    expect(after.revision).not.toBe(before.revision);
    expect(observeWorkspacePatch(before, after).changedPaths).toEqual([]);
  });

  it("observes creation of an authorized path that began missing", async () => {
    const target = "src/new-handler.ts";
    const before = await captureWorkspaceSnapshot(workspacePath, [target]);
    await writeFile(path.join(workspacePath, target), "export const newHandler = 1;\n");
    const after = await captureWorkspaceSnapshot(workspacePath, [target]);

    expect(before.files).toEqual([
      { path: target, kind: "missing", contentDigest: "sha256:missing" },
    ]);
    expect(changedPathsBetween(before, after)).toEqual([target]);
  });

  it("captures staged and untracked paths before the first Git commit", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "kondex-unborn-workspace-"));
    temporaryDirectories.push(root);
    const unbornWorkspace = path.join(root, "workspace");
    await mkdir(path.join(unbornWorkspace, "src"), { recursive: true });
    await writeFile(path.join(unbornWorkspace, "src", "staged.ts"), "export const staged = 1;\n");
    await writeFile(
      path.join(unbornWorkspace, "src", "untracked.ts"),
      "export const untracked = 1;\n",
    );
    execFileSync("git", ["init", "-q"], { cwd: unbornWorkspace });
    execFileSync("git", ["add", "src/staged.ts"], { cwd: unbornWorkspace });

    const snapshot = await captureWorkspaceSnapshot(unbornWorkspace);

    expect(snapshot.files.map((file) => file.path)).toEqual(["src/staged.ts", "src/untracked.ts"]);
  });
});
