import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  captureWorkspaceCodeSymbols,
  captureWorkspaceSnapshot,
  changedWorkspaceCodeSymbolIds,
  observeWorkspacePatch,
} from "../src/index.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("workspace Code Symbol observation", () => {
  it("derives stable semantic changes from the actual worktree", async () => {
    const workspace = await createWorkspace();
    const target = "src/handler.ts";
    const before = await captureWorkspaceCodeSymbols(workspace, [target]);
    await writeFile(
      path.join(workspace, target),
      "export function handler(value: number) { return value + 2; }\n",
    );
    const after = await captureWorkspaceCodeSymbols(workspace, [target]);

    expect(after.codebaseId).toBe(before.codebaseId);
    expect(changedWorkspaceCodeSymbolIds(before, after)).toEqual([
      expect.stringMatching(/^code-symbol:/),
    ]);
    expect(after.symbols.find((symbol) => symbol.behaviorBearing)?.identity).toMatchObject({
      relativePath: target,
      qualifiedName: "handler",
      kind: "function",
    });
  });

  it("ignores formatting-only edits when the provider reports stable symbol content", async () => {
    const workspace = await createWorkspace();
    const target = "src/handler.ts";
    const before = await captureWorkspaceCodeSymbols(workspace, [target]);
    await writeFile(
      path.join(workspace, target),
      "export function handler(value: number) {\n  return value + 1;\n}\n",
    );
    const after = await captureWorkspaceCodeSymbols(workspace, [target]);

    expect(changedWorkspaceCodeSymbolIds(before, after)).toEqual([]);
  });

  it("digests the sidecar-observed before and after file states", async () => {
    const workspace = await createWorkspace();
    const target = "src/handler.ts";
    const before = await captureWorkspaceSnapshot(workspace, [target]);
    await writeFile(path.join(workspace, target), "export const handler = () => 2;\n");
    const after = await captureWorkspaceSnapshot(workspace, [target]);

    expect(observeWorkspacePatch(before, after)).toEqual({
      patchDigest: expect.stringMatching(/^sha256:[a-f0-9]{64}$/),
      changedPaths: [target],
      beforeRevision: before.revision,
      afterRevision: after.revision,
    });
  });
});

async function createWorkspace(): Promise<string> {
  const root = await mkdtemp(path.join(tmpdir(), "kontext-code-observer-"));
  temporaryDirectories.push(root);
  const workspace = path.join(root, "workspace");
  await mkdir(path.join(workspace, "src"), { recursive: true });
  await writeFile(
    path.join(workspace, "src", "handler.ts"),
    "export function handler(value: number) { return value + 1; }\n",
  );
  execFileSync("git", ["init", "-q"], { cwd: workspace });
  execFileSync("git", ["add", "."], { cwd: workspace });
  execFileSync(
    "git",
    [
      "-c",
      "user.name=Kontext Test",
      "-c",
      "user.email=test@example.invalid",
      "commit",
      "-qm",
      "base",
    ],
    { cwd: workspace },
  );
  return workspace;
}
