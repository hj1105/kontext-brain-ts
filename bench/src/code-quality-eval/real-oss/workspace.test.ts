import { createHash } from "node:crypto";
import { access, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { runWorkspaceCommand } from "../workspace.js";
import type { RealOssTask } from "./contracts.js";
import {
  createRealOssWorkspace,
  parseDjangoPassed,
  readWorkspaceFile,
  validateRealOssSource,
} from "./workspace.js";

const cleanup = new Set<string>();

afterEach(async () => {
  await Promise.all([...cleanup].map((entry) => rm(entry, { recursive: true, force: true })));
  cleanup.clear();
});

describe("real OSS workspace", () => {
  it("keeps the authentic base commit but exposes no remote or later history", async () => {
    const fixture = await sourceFixture();
    const workspace = await createRealOssWorkspace({
      task: fixture.task,
      sourceRepositoryPath: fixture.source,
      installDependencies: false,
    });
    cleanup.add(workspace.workspacePath);

    expect(workspace.baseRevision).toBe(fixture.task.baseCommit);
    expect(await readWorkspaceFile(workspace.workspacePath, "src/example.py")).toBe("VALUE = 1\n");
    const remotes = await runWorkspaceCommand(workspace.workspacePath, "git", ["remote"]);
    expect(remotes.stdout.trim()).toBe("");
    await expect(
      access(path.join(workspace.workspacePath, ".git", "FETCH_HEAD")),
    ).rejects.toThrow();
    const commits = await runWorkspaceCommand(workspace.workspacePath, "git", [
      "rev-list",
      "--all",
      "--count",
    ]);
    expect(commits.stdout.trim()).toBe("1");
  });

  it("rejects source bytes that do not match the pinned provenance", async () => {
    const fixture = await sourceFixture();
    const invalid: RealOssTask = {
      ...fixture.task,
      sourceIntegrity: [{ relativePath: "src/example.py", sha256: "0".repeat(64) }],
    };
    await expect(validateRealOssSource(invalid, fixture.source)).rejects.toThrow(
      "Source integrity mismatch",
    );
  });

  it("counts partial Django test success from unittest summaries", () => {
    expect(parseDjangoPassed("Ran 4 tests in 0.1s\nFAILED (failures=2)", 4)).toBe(2);
    expect(parseDjangoPassed("Ran 4 tests in 0.1s\nFAILED (errors=1, skipped=1)", 4)).toBe(2);
    expect(
      parseDjangoPassed("Ran 4 tests in 0.1s\nFAILED (errors=1, expected failures=2)", 4),
    ).toBe(3);
  });
});

async function sourceFixture(): Promise<{ readonly source: string; readonly task: RealOssTask }> {
  const source = await mkdtemp(path.join(tmpdir(), "kontext-real-oss-source-test-"));
  cleanup.add(source);
  await mkdir(path.join(source, "src"));
  await writeFile(path.join(source, "src", "example.py"), "VALUE = 1\n");
  await runWorkspaceCommand(source, "git", ["init", "-q"]);
  await runWorkspaceCommand(source, "git", ["add", "."]);
  await runWorkspaceCommand(source, "git", [
    "-c",
    "user.name=Real OSS Test",
    "-c",
    "user.email=real-oss@example.invalid",
    "commit",
    "-qm",
    "base",
  ]);
  const revision = await runWorkspaceCommand(source, "git", ["rev-parse", "HEAD"]);
  const hiddenPatch = "diff --git a/test.py b/test.py\n";
  return {
    source,
    task: {
      instanceId: "example__repo-1",
      taskId: "task:example",
      codebaseId: "codebase:example",
      repository: "example/repo",
      repositoryUrl: "https://github.com/example/repo.git",
      license: "MIT",
      baseCommit: revision.stdout.trim(),
      upstreamIssueUrl: "https://github.com/example/repo/issues/1",
      upstreamPullRequestUrl: "https://github.com/example/repo/pull/2",
      publicPrompt: "Change the value.",
      acceptanceStatement: "The value changes.",
      nonGoals: ["Editing tests"],
      risk: "low",
      codeRoots: ["src"],
      allowedPaths: ["src/example.py"],
      targets: [
        {
          workItemId: "work-item:example",
          plannedSymbolId: "planned-symbol:example",
          relativePath: "src/example.py",
          qualifiedName: "change_value",
          symbolKind: "function",
          binding: "planned",
          responsibility: "Change the value.",
          ontologyNodeIds: ["domain:example"],
          capabilityId: "capability:example",
        },
      ],
      sourceIntegrity: [
        {
          relativePath: "src/example.py",
          sha256: createHash("sha256").update("VALUE = 1\n").digest("hex"),
        },
      ],
      environment: { pythonVersion: "3.11", packages: [] },
      publicTest: { command: "python3", args: ["-c", "pass"] },
      hiddenTest: {
        patch: hiddenPatch,
        patchSha256: createHash("sha256").update(hiddenPatch).digest("hex"),
        runner: { kind: "pytest-selectors", command: "python3", args: ["-m", "pytest", "-q"] },
        failToPass: [],
        passToPass: [],
      },
      sourceDocuments: [],
      normativeRecords: [],
      limitations: [],
    },
  };
}
