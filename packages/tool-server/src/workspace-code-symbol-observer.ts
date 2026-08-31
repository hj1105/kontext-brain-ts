import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile, realpath } from "node:fs/promises";
import path from "node:path";
import { type CodeSymbolIdentity, TypeScriptCodeProvider } from "@kontext-brain/code";
import {
  type WorkspaceObservationSnapshot,
  captureWorkspaceSnapshot,
} from "./workspace-change-observer.js";

export interface WorkspaceCodeSymbolState {
  readonly symbolId: string;
  readonly identity: CodeSymbolIdentity;
  readonly behaviorBearing: boolean;
  readonly contentHash: string;
}

export interface WorkspaceCodeSymbolSnapshot {
  readonly codebaseId: string;
  readonly workspaceRevision: string;
  readonly symbols: readonly WorkspaceCodeSymbolState[];
}

const maxSourceFiles = 10_000;
const maxSourceBytes = 32 * 1024 * 1024;
const codeExtension = /\.(?:[cm]?[jt]sx?)$/i;

export async function captureWorkspaceCodeSymbols(
  workspacePath: string,
  targetPaths: readonly string[],
  observedWorkspace?: WorkspaceObservationSnapshot,
): Promise<WorkspaceCodeSymbolSnapshot> {
  const requestedWorkspace = path.resolve(workspacePath);
  const workspace = await realpath(requestedWorkspace).catch(() => requestedWorkspace);
  const workspaceSnapshot =
    observedWorkspace ?? (await captureWorkspaceSnapshot(workspace, targetPaths));
  if (path.resolve(workspaceSnapshot.workspacePath) !== workspace) {
    throw new Error("Code Symbol observation does not match the observed workspace");
  }
  const sourcePaths = workspaceSnapshot.files
    .filter((file) => file.kind === "file" && codeExtension.test(file.path))
    .map((file) => file.path);
  if (sourcePaths.length > maxSourceFiles) {
    throw new Error(`Code observation exceeds ${maxSourceFiles} source files`);
  }

  let sourceBytes = 0;
  const files = await Promise.all(
    sourcePaths.map(async (relativePath) => {
      const content = await readFile(path.join(workspace, relativePath), "utf8");
      sourceBytes += Buffer.byteLength(content);
      if (sourceBytes > maxSourceBytes) {
        throw new Error(`Code observation exceeds ${maxSourceBytes} source bytes`);
      }
      return { path: relativePath, content };
    }),
  );
  const sourcePathSet = new Set(sourcePaths);
  const targets = uniqueSorted(
    targetPaths
      .map((targetPath) => canonicalRelativePath(requestedWorkspace, workspace, targetPath))
      .filter((targetPath) => codeExtension.test(targetPath) && sourcePathSet.has(targetPath)),
  );
  const codebaseId = await resolveCodebaseId(workspace);
  const provider = new TypeScriptCodeProvider();
  const symbols = targets
    .flatMap((targetPath) =>
      provider.analyze({ codebaseId, targetPath, files }).symbols.map((symbol) => ({
        symbolId: symbol.symbolId,
        identity: symbol.identity,
        behaviorBearing: symbol.behaviorBearing,
        contentHash: symbol.contentHash,
      })),
    )
    .sort(compareSymbolState);
  return {
    codebaseId,
    workspaceRevision: workspaceSnapshot.revision,
    symbols,
  };
}

export function changedWorkspaceCodeSymbolIds(
  before: WorkspaceCodeSymbolSnapshot,
  after: WorkspaceCodeSymbolSnapshot,
): readonly string[] {
  if (before.codebaseId !== after.codebaseId) {
    throw new Error("Code Symbol comparison requires the same Codebase");
  }
  const previous = new Map(before.symbols.map((symbol) => [symbol.symbolId, symbol] as const));
  const current = new Map(after.symbols.map((symbol) => [symbol.symbolId, symbol] as const));
  return uniqueSorted([...previous.keys(), ...current.keys()]).filter((symbolId) => {
    const left = previous.get(symbolId);
    const right = current.get(symbolId);
    return (
      Boolean(left?.behaviorBearing || right?.behaviorBearing) &&
      (!left || !right || left.contentHash !== right.contentHash)
    );
  });
}

async function resolveCodebaseId(workspacePath: string): Promise<string> {
  const remote = (
    await runGit(workspacePath, ["config", "--get", "remote.origin.url"]).catch(() => "")
  ).trim();
  const commonDirectory = await runGit(workspacePath, ["rev-parse", "--git-common-dir"]).catch(
    () => workspacePath,
  );
  const identity = remote || path.resolve(workspacePath, commonDirectory.trim());
  return `codebase:${createHash("sha256").update(identity).digest("hex")}`;
}

function runGit(workspacePath: string, args: readonly string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn("git", args, {
      cwd: workspacePath,
      shell: false,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    let bytes = 0;
    const capture = (target: Buffer[], chunk: Buffer): void => {
      bytes += chunk.byteLength;
      if (bytes > 1024 * 1024) {
        child.kill("SIGKILL");
        reject(new Error("Git Codebase identity output is too large"));
        return;
      }
      target.push(chunk);
    };
    child.stdout.on("data", (chunk: Buffer) => capture(stdout, chunk));
    child.stderr.on("data", (chunk: Buffer) => capture(stderr, chunk));
    child.once("error", reject);
    child.once("close", (code) => {
      if (code === 0) resolve(Buffer.concat(stdout).toString("utf8"));
      else reject(new Error(Buffer.concat(stderr).toString("utf8") || `git exited ${code}`));
    });
  });
}

function canonicalRelativePath(
  requestedWorkspacePath: string,
  canonicalWorkspacePath: string,
  value: string,
): string {
  if (!path.isAbsolute(value)) return canonicalPath(value);
  const requestedRelative = canonicalPath(path.relative(requestedWorkspacePath, value));
  const relative = isSafeRelativePath(requestedRelative)
    ? requestedRelative
    : path.relative(canonicalWorkspacePath, value);
  return relative.replaceAll("\\", "/").replace(/^\.\//, "");
}

function canonicalPath(value: string): string {
  return value.replaceAll("\\", "/").replace(/^\.\//, "");
}

function isSafeRelativePath(value: string): boolean {
  return value.length > 0 && value !== ".." && !value.startsWith("../") && !path.isAbsolute(value);
}

function compareSymbolState(
  left: WorkspaceCodeSymbolState,
  right: WorkspaceCodeSymbolState,
): number {
  return (
    left.identity.relativePath.localeCompare(right.identity.relativePath) ||
    left.identity.qualifiedName.localeCompare(right.identity.qualifiedName) ||
    left.symbolId.localeCompare(right.symbolId)
  );
}

function uniqueSorted(values: readonly string[]): readonly string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}
