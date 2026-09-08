import { execFileSync } from "node:child_process";
import {
  chmod,
  mkdir,
  mkdtemp,
  readFile,
  readlink,
  realpath,
  rm,
  stat,
  symlink,
  truncate,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { GitRuntimeWorktreeManager } from "@kontext-brain/local";
import { afterEach, describe, expect, it } from "vitest";
import {
  prepareCodingWorkspaceSeed,
  verifyCodingWorkspaceSeed,
} from "../src/local-coding-workspace-seed.js";
import { inspectTaskWorkspace } from "../src/local-task-preparation.js";
import { captureWorkspaceCodeSymbols } from "../src/workspace-code-symbol-observer.js";

const directories: string[] = [];
afterEach(async () => {
  for (const directory of directories.splice(0))
    await rm(directory, { recursive: true, force: true });
});
function git(cwd: string, args: string[]) {
  return execFileSync("git", ["-c", "core.hooksPath=/dev/null", ...args], {
    cwd,
    encoding: "utf8",
  }).trim();
}
async function fixture(kind: "dirty" | "unborn" | "folder") {
  const root = await mkdtemp(path.join(tmpdir(), "kontext-coding-seed-"));
  directories.push(root);
  const workspacePath = path.join(root, "source with spaces");
  const dataDirectory = path.join(root, "private");
  await mkdir(workspacePath);
  await writeFile(path.join(workspacePath, "logic.ts"), "export const value = 1;\r\n");
  await writeFile(path.join(workspacePath, ".gitignore"), "ignored/\n");
  if (kind !== "folder") git(workspacePath, ["init", "-q"]);
  if (kind === "dirty") {
    git(workspacePath, ["add", "."]);
    git(workspacePath, [
      "-c",
      "user.name=Test",
      "-c",
      "user.email=test@invalid.local",
      "commit",
      "--no-gpg-sign",
      "-qm",
      "base",
    ]);
    await writeFile(path.join(workspacePath, "logic.ts"), "export const value = 2;\r\n");
    git(workspacePath, ["add", "logic.ts"]);
    await writeFile(path.join(workspacePath, "logic.ts"), "export const value = 3;\r\n");
  }
  await mkdir(path.join(workspacePath, "ignored"));
  await writeFile(path.join(workspacePath, "ignored", "private.txt"), "not coding input");
  await writeFile(path.join(workspacePath, "new file.ts"), "export const added = true;\n");
  return { root, workspacePath, dataDirectory };
}

describe("private coding workspace seed", () => {
  it.each(["dirty", "unborn", "folder"] as const)(
    "captures %s input without changing its files or Git state",
    async (kind) => {
      const f = await fixture(kind);
      const status = kind === "folder" ? null : git(f.workspacePath, ["status", "--porcelain=v1"]);
      const index = kind === "dirty" ? git(f.workspacePath, ["show", ":logic.ts"]) : null;
      const seed = await prepareCodingWorkspaceSeed(f.dataDirectory, f.workspacePath);
      const repeated = await prepareCodingWorkspaceSeed(f.dataDirectory, f.workspacePath);
      expect(repeated).toEqual(seed);
      const originalSymbols = await captureWorkspaceCodeSymbols(await realpath(f.workspacePath), [
        "logic.ts",
      ]);
      const seedSymbols = await captureWorkspaceCodeSymbols(seed.repositoryPath, ["logic.ts"]);
      expect(seedSymbols.codebaseId).toBe(originalSymbols.codebaseId);
      expect(seedSymbols.symbols.map((symbol) => symbol.symbolId)).toEqual(
        originalSymbols.symbols.map((symbol) => symbol.symbolId),
      );
      expect(git(seed.repositoryPath, ["remote"])).toBe("");
      expect(seed.workspacePath).toBe(await realpath(f.workspacePath));
      expect(seed.repositoryPath).not.toBe(f.workspacePath);
      expect(await readFile(path.join(seed.repositoryPath, "logic.ts"))).toEqual(
        await readFile(path.join(f.workspacePath, "logic.ts")),
      );
      expect(git(seed.repositoryPath, ["status", "--porcelain=v1"])).toBe("");
      expect(git(seed.repositoryPath, ["ls-tree", "-r", "--name-only", "HEAD"])).not.toContain(
        "ignored/",
      );
      expect(git(seed.repositoryPath, ["ls-tree", "-r", "--name-only", "HEAD"])).toContain(
        "new file.ts",
      );
      if (kind !== "folder")
        expect(git(f.workspacePath, ["status", "--porcelain=v1"])).toBe(status);
      else
        await expect(readFile(path.join(f.workspacePath, ".git", "HEAD"))).rejects.toMatchObject({
          code: "ENOENT",
        });
      if (kind === "dirty") expect(git(f.workspacePath, ["show", ":logic.ts"])).toBe(index);
      await expect(verifyCodingWorkspaceSeed(f.dataDirectory, seed)).resolves.toEqual(
        seed.repositoryPath,
      );
      await writeFile(path.join(f.workspacePath, "new file.ts"), "export const added = false;\n");
      const changed = await prepareCodingWorkspaceSeed(f.dataDirectory, f.workspacePath);
      expect(changed.codeRevision).not.toBe(seed.codeRevision);
      expect(
        (await captureWorkspaceCodeSymbols(changed.repositoryPath, ["logic.ts"])).codebaseId,
      ).toBe(originalSymbols.codebaseId);
      expect(await readFile(path.join(seed.repositoryPath, "new file.ts"), "utf8")).toContain(
        "true",
      );
    },
  );

  it.skipIf(process.platform === "win32")(
    "does not follow a source symlink into files outside the selected workspace",
    async () => {
      const f = await fixture("folder");
      await writeFile(path.join(f.root, "private.txt"), "do not copy");
      await symlink("../private.txt", path.join(f.workspacePath, "escape.txt"));
      await expect(prepareCodingWorkspaceSeed(f.dataDirectory, f.workspacePath)).rejects.toThrow(
        /symlink/i,
      );
    },
  );

  it("preserves raw bytes, file names and executable modes through the actual worker worktree", async () => {
    const f = await fixture("folder");
    await writeFile(path.join(f.workspacePath, ".gitattributes"), "*.ts text eol=lf\n");
    const binary = Buffer.from([0, 255, 13, 10, 128]);
    await writeFile(path.join(f.workspacePath, "binary.dat"), binary);
    await writeFile(path.join(f.workspacePath, "run.sh"), "#!/bin/sh\necho fixture\n");
    await chmod(path.join(f.workspacePath, "run.sh"), 0o755);
    const seed = await prepareCodingWorkspaceSeed(f.dataDirectory, f.workspacePath);
    const worktree = await new GitRuntimeWorktreeManager(
      seed.repositoryPath,
      path.join(f.root, "workers"),
    ).prepare({
      taskId: "task:seed",
      baseRevision: seed.codeRevision,
      workItem: {
        taskId: "task:seed",
        workItemId: "work:seed",
        plannedSymbolIds: ["symbol:seed"],
        dependsOn: [],
        allowedPaths: ["logic.ts"],
        requiredVerifiers: [],
        capabilityId: "capability:seed",
      },
    });
    expect(await readFile(path.join(worktree.workspacePath, "logic.ts"))).toEqual(
      await readFile(path.join(f.workspacePath, "logic.ts")),
    );
    expect(await readFile(path.join(worktree.workspacePath, "binary.dat"))).toEqual(binary);
    expect(await readFile(path.join(worktree.workspacePath, "new file.ts"), "utf8")).toContain(
      "added",
    );
    if (process.platform !== "win32")
      expect((await stat(path.join(worktree.workspacePath, "run.sh"))).mode & 0o111).not.toBe(0);
  });

  it.skipIf(process.platform === "win32")(
    "preserves internal relative symlinks and rejects added ignored files",
    async () => {
      const f = await fixture("folder");
      await symlink("logic.ts", path.join(f.workspacePath, "alias.ts"));
      const seed = await prepareCodingWorkspaceSeed(f.dataDirectory, f.workspacePath);
      expect(await readlink(path.join(seed.repositoryPath, "alias.ts"))).toBe("logic.ts");
      await mkdir(path.join(seed.repositoryPath, "ignored"));
      await writeFile(path.join(seed.repositoryPath, "ignored", "injected.ts"), "unreviewed");
      await expect(verifyCodingWorkspaceSeed(f.dataDirectory, seed)).rejects.toThrow(
        /seed.*changed/i,
      );
    },
  );

  it("rejects oversized files before copying their contents", async () => {
    const f = await fixture("folder");
    const oversized = path.join(f.workspacePath, "oversized.bin");
    await writeFile(oversized, "");
    await truncate(oversized, 64 * 1024 * 1024 + 1);
    await expect(prepareCodingWorkspaceSeed(f.dataDirectory, f.workspacePath)).rejects.toThrow(
      /byte limit/,
    );
  });

  it("refuses a modified published seed before worker admission", async () => {
    const f = await fixture("folder");
    const seed = await prepareCodingWorkspaceSeed(f.dataDirectory, f.workspacePath);
    await writeFile(path.join(seed.repositoryPath, "logic.ts"), "replaced");
    await expect(verifyCodingWorkspaceSeed(f.dataDirectory, seed)).rejects.toThrow(
      /seed.*changed/i,
    );
  });

  it("keeps completion workspace inspection read-only instead of creating a replacement baseline", async () => {
    const f = await fixture("folder");
    await expect(
      inspectTaskWorkspace(f.dataDirectory, { workspacePath: f.workspacePath }),
    ).rejects.toThrow(/clean Git commit/);
    await expect(stat(f.dataDirectory)).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("binds the Codebase identity to the seed commit and refuses a changed override", async () => {
    const f = await fixture("dirty");
    git(f.workspacePath, ["remote", "add", "origin", "https://example.invalid/seed-fixture.git"]);
    const seed = await prepareCodingWorkspaceSeed(f.dataDirectory, f.workspacePath);
    const original = await captureWorkspaceCodeSymbols(await realpath(f.workspacePath), [
      "logic.ts",
    ]);
    expect((await captureWorkspaceCodeSymbols(seed.repositoryPath, ["logic.ts"])).codebaseId).toBe(
      original.codebaseId,
    );
    git(seed.repositoryPath, ["config", "kontext.seedCodebaseId", `codebase:${"0".repeat(64)}`]);
    await expect(verifyCodingWorkspaceSeed(f.dataDirectory, seed)).rejects.toThrow(
      /Codebase identity changed/,
    );
  });
});
