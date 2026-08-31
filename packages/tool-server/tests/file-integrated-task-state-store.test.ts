import { mkdtemp, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileIntegratedTaskStateStore } from "../src/index.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("FileIntegratedTaskStateStore", () => {
  it("persists one digest-checked private integration result per schedule", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "kontext-integrated-state-"));
    temporaryDirectories.push(directory);
    const store = new FileIntegratedTaskStateStore(directory);
    const state = {
      taskId: "task:integration",
      scheduleJobId: "schedule:one",
      repositoryPath: "/repository",
      workspacePath: "/worktree",
      baseRevision: "commit:base",
      gitCommit: "commit:integrated",
      resultRevision: "workspace:integrated",
      contextDigest: "context:current",
      changeBundleIds: ["bundle:one"],
      workItemIds: ["work:one"],
      changedPaths: ["src/one.ts"],
      changedSymbolIds: ["symbol:one"],
      authorProviders: ["codex" as const],
      createdAt: "2026-08-31T02:00:00.000Z",
    };

    await expect(store.put(state)).resolves.toEqual(state);
    await expect(store.get(state.taskId)).resolves.toEqual(state);
    expect((await stat(store.filePath(state.taskId))).mode & 0o777).toBe(0o600);
  });
});
