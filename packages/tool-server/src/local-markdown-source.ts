import { createHash } from "node:crypto";
import { open, realpath, stat } from "node:fs/promises";
import path from "node:path";
import {
  type Principal,
  RegexHeaderChunkingStrategy,
  type ResourceSnapshot,
} from "@kontext-brain/core";

const MAX_BYTES = 512 * 1024;
const sections = new RegexHeaderChunkingStrategy(/(?=(?:^|\n)##\s)/, /(?:^|\n)##\s*([^\n]+)/, 1);
export async function resolveLocalMarkdownSource(workspacePath: string, relativePath: string) {
  const segments = relativePath.replaceAll("\\", "/").split("/");
  if (
    !path.isAbsolute(workspacePath) ||
    path.isAbsolute(relativePath) ||
    path.win32.isAbsolute(relativePath) ||
    segments.some((part) => !part || part === "." || part === "..") ||
    !/\.(md|markdown)$/i.test(relativePath)
  )
    throw new Error("Select a workspace-relative Markdown file");
  const root = await realpath(workspacePath);
  return {
    root,
    segments,
    source: {
      connectorId: "local-markdown",
      externalId: `${hash(root)}:${segments.join("/")}`,
      type: "markdown",
    },
  };
}
export async function captureLocalMarkdownSource(
  principal: Principal,
  workspacePath: string,
  relativePath: string,
): Promise<ResourceSnapshot> {
  const { root, segments, source } = await resolveLocalMarkdownSource(workspacePath, relativePath);
  const selected = path.join(root, ...segments);
  const canonical = await realpath(selected);
  const relative = path.relative(root, canonical);
  if (
    !relative ||
    relative === ".." ||
    relative.startsWith(`..${path.sep}`) ||
    path.isAbsolute(relative)
  )
    throw new Error("Markdown source must stay inside the selected workspace");
  const before = await stat(canonical);
  if (!before.isFile() || before.size > MAX_BYTES)
    throw new Error("Markdown source must be a regular file no larger than 512 KiB");
  const handle = await open(canonical, "r");
  let bytes: Buffer;
  try {
    const opened = await handle.stat();
    if (!sameFile(before, opened)) throw new Error("Markdown source changed while opening");
    const buffer = Buffer.alloc(MAX_BYTES + 1);
    let length = 0;
    while (length < buffer.length) {
      const result = await handle.read(buffer, length, buffer.length - length, length);
      if (result.bytesRead === 0) break;
      length += result.bytesRead;
    }
    if (length > MAX_BYTES) throw new Error("Markdown source exceeds 512 KiB");
    const after = await handle.stat();
    if (
      !sameFile(before, after) ||
      !sameFile(after, await stat(canonical)) ||
      (await realpath(selected)) !== canonical
    )
      throw new Error("Markdown source changed during capture");
    bytes = buffer.subarray(0, length);
  } finally {
    await handle.close();
  }
  const body = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  const parts = sections.split(body);
  if (parts.length > 256) throw new Error("Markdown source has too many sections");
  const occurrences = new Map<string, number>();
  const chunks = parts.map((part, position) => {
    const title = part.title;
    const ordinal = occurrences.get(title) ?? 0;
    occurrences.set(title, ordinal + 1);
    return {
      id: `section:${hash(title)}:${ordinal}`,
      position,
      text: part.fullText,
      contentHash: hash(part.fullText),
    };
  });
  return {
    organizationId: principal.organizationId,
    source,
    title: segments.join("/"),
    body,
    contentHash: hash(bytes),
    acl: { subjectIds: [principal.subjectId] },
    chunks,
    entities: [],
    facts: [],
    ontologyNodeIds: [],
  };
}
function hash(value: string | Buffer): string {
  return `sha256:${createHash("sha256").update(value).digest("hex")}`;
}
function sameFile(
  left: Awaited<ReturnType<typeof stat>>,
  right: Awaited<ReturnType<typeof stat>>,
): boolean {
  return (
    left.dev === right.dev &&
    left.ino === right.ino &&
    left.size === right.size &&
    left.mtimeMs === right.mtimeMs &&
    left.ctimeMs === right.ctimeMs
  );
}
