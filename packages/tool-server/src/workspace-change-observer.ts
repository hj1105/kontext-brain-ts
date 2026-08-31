import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, readlink, realpath } from "node:fs/promises";
import path from "node:path";

export interface WorkspaceFileState {
  readonly path: string;
  readonly kind: "file" | "symlink" | "missing";
  readonly contentDigest: string;
}

export interface WorkspaceObservationSnapshot {
  readonly workspacePath: string;
  readonly revision: string;
  readonly files: readonly WorkspaceFileState[];
}

export interface WorkspacePatchObservation {
  readonly patchDigest: string;
  readonly changedPaths: readonly string[];
  readonly beforeRevision: string;
  readonly afterRevision: string;
}

const maxGitOutputBytes = 64 * 1024 * 1024;
const maxObservedFiles = 50_000;

export async function captureWorkspaceSnapshot(
  workspacePath: string,
  fallbackPaths: readonly string[] = [],
): Promise<WorkspaceObservationSnapshot> {
  const normalizedWorkspace = path.resolve(workspacePath);
  const canonicalWorkspace = await realpath(normalizedWorkspace).catch(() => normalizedWorkspace);
  const gitFiles = await listGitFiles(canonicalWorkspace).catch(() => undefined);
  const relativePaths =
    gitFiles ?? normalizeFallbackPaths(normalizedWorkspace, canonicalWorkspace, fallbackPaths);
  if (relativePaths.length > maxObservedFiles) {
    throw new Error(`Workspace observation exceeds ${maxObservedFiles} files`);
  }
  const files = (
    await Promise.all(
      relativePaths.map((relativePath) => fileState(canonicalWorkspace, relativePath)),
    )
  ).sort((left, right) => left.path.localeCompare(right.path));
  const revision = `workspace-revision:${sha256(stableJson(files))}`;
  return { workspacePath: canonicalWorkspace, revision, files };
}

export function changedPathsBetween(
  before: WorkspaceObservationSnapshot,
  after: WorkspaceObservationSnapshot,
): readonly string[] {
  const previous = new Map(before.files.map((file) => [file.path, file] as const));
  const current = new Map(after.files.map((file) => [file.path, file] as const));
  return Array.from(new Set([...previous.keys(), ...current.keys()]))
    .filter((filePath) => stableJson(previous.get(filePath)) !== stableJson(current.get(filePath)))
    .sort((left, right) => left.localeCompare(right));
}

export function observeWorkspacePatch(
  before: WorkspaceObservationSnapshot,
  after: WorkspaceObservationSnapshot,
): WorkspacePatchObservation {
  if (before.workspacePath !== after.workspacePath) {
    throw new Error("Workspace patch observation requires the same workspace");
  }
  const previous = new Map(before.files.map((file) => [file.path, file] as const));
  const current = new Map(after.files.map((file) => [file.path, file] as const));
  const changedPaths = changedPathsBetween(before, after);
  const changes = changedPaths.map((changedPath) => ({
    path: changedPath,
    before: previous.get(changedPath) ?? null,
    after: current.get(changedPath) ?? null,
  }));
  return {
    patchDigest: `sha256:${sha256(stableJson(changes))}`,
    changedPaths,
    beforeRevision: before.revision,
    afterRevision: after.revision,
  };
}

async function listGitFiles(workspacePath: string): Promise<readonly string[]> {
  const output = await run(
    "git",
    ["ls-files", "-c", "-o", "--exclude-standard", "-z"],
    workspacePath,
  );
  return Array.from(
    new Set(
      output
        .toString("utf8")
        .split("\u0000")
        .filter(Boolean)
        .map(canonicalRelativePath)
        .filter((filePath) => isSafeRelativePath(filePath)),
    ),
  ).sort((left, right) => left.localeCompare(right));
}

function normalizeFallbackPaths(
  requestedWorkspacePath: string,
  canonicalWorkspacePath: string,
  fallbackPaths: readonly string[],
): readonly string[] {
  return Array.from(
    new Set(
      fallbackPaths
        .map((filePath) => {
          if (!path.isAbsolute(filePath)) return filePath;
          const requestedRelative = path.relative(requestedWorkspacePath, filePath);
          return isSafeRelativePath(canonicalRelativePath(requestedRelative))
            ? requestedRelative
            : path.relative(canonicalWorkspacePath, filePath);
        })
        .map(canonicalRelativePath)
        .filter((filePath) => isSafeRelativePath(filePath)),
    ),
  ).sort((left, right) => left.localeCompare(right));
}

async function fileState(workspacePath: string, relativePath: string): Promise<WorkspaceFileState> {
  const absolutePath = path.resolve(workspacePath, relativePath);
  if (!isWithin(workspacePath, absolutePath)) {
    throw new Error(`Observed path escapes workspace: ${relativePath}`);
  }
  try {
    const metadata = await lstat(absolutePath);
    if (metadata.isSymbolicLink()) {
      return {
        path: relativePath,
        kind: "symlink",
        contentDigest: `sha256:${sha256(await readlink(absolutePath))}`,
      };
    }
    if (!metadata.isFile()) {
      return { path: relativePath, kind: "missing", contentDigest: "sha256:non-file" };
    }
    return {
      path: relativePath,
      kind: "file",
      contentDigest: `sha256:${await hashFile(absolutePath)}`,
    };
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return { path: relativePath, kind: "missing", contentDigest: "sha256:missing" };
    }
    throw error;
  }
}

function hashFile(filePath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    const stream = createReadStream(filePath);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.once("error", reject);
    stream.once("end", () => resolve(hash.digest("hex")));
  });
}

function run(command: string, args: readonly string[], cwd: string): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, shell: false, stdio: ["ignore", "pipe", "pipe"] });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    let bytes = 0;
    const capture = (target: Buffer[], chunk: Buffer): void => {
      bytes += chunk.byteLength;
      if (bytes > maxGitOutputBytes) {
        child.kill("SIGKILL");
        reject(new Error("Git workspace observation output is too large"));
        return;
      }
      target.push(chunk);
    };
    child.stdout.on("data", (chunk: Buffer) => capture(stdout, chunk));
    child.stderr.on("data", (chunk: Buffer) => capture(stderr, chunk));
    child.once("error", reject);
    child.once("close", (code) => {
      if (code === 0) resolve(Buffer.concat(stdout));
      else reject(new Error(Buffer.concat(stderr).toString("utf8") || `git exited ${code}`));
    });
  });
}

function canonicalRelativePath(value: string): string {
  return value.replaceAll("\\", "/").replace(/^\.\//, "");
}

function isSafeRelativePath(value: string): boolean {
  return value.length > 0 && !path.isAbsolute(value) && value !== ".." && !value.startsWith("../");
}

function isWithin(workspacePath: string, candidatePath: string): boolean {
  const relative = path.relative(workspacePath, candidatePath);
  return relative !== ".." && !relative.startsWith(`..${path.sep}`);
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function stableJson(value: unknown): string {
  return JSON.stringify(stableValue(value));
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, nested]) => [key, stableValue(nested)]),
    );
  }
  return value;
}

function isNodeError(value: unknown): value is NodeJS.ErrnoException {
  return value instanceof Error && "code" in value;
}
