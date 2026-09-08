import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";
import type { MCPConnector, MCPData, MCPResource } from "./mcp-connector.js";

/**
 * Exposes Markdown already in a repository as an MCP source, so a codebase that
 * keeps its decisions in `docs/` can build an ontology without standing up a
 * server first. It is a connector rather than a one-off document source so the
 * existing collect/classify/index path applies to it unchanged.
 */

export interface LocalMarkdownOptions {
  /** Directories to walk, relative to the root. Defaults to the root itself. */
  readonly include?: readonly string[];
  readonly extensions?: readonly string[];
  readonly maxFiles?: number;
  /** Directory names skipped anywhere in the tree. */
  readonly exclude?: readonly string[];
}

const DEFAULT_EXTENSIONS = [".md", ".markdown"] as const;
const DEFAULT_EXCLUDE = [
  "node_modules",
  ".git",
  "dist",
  "build",
  "out",
  "coverage",
  ".next",
  ".turbo",
  "vendor",
] as const;
const DEFAULT_MAX_FILES = 2000;
/** Enough for a title, summary and the opening decision; full text is read on fetch. */
const DESCRIPTION_CHARS = 400;

function walk(
  directory: string,
  extensions: readonly string[],
  exclude: readonly string[],
  limit: number,
  found: string[],
): void {
  if (found.length >= limit) return;
  let entries: string[];
  try {
    entries = readdirSync(directory);
  } catch {
    return;
  }
  for (const entry of entries.sort()) {
    if (found.length >= limit) return;
    if (exclude.includes(entry)) continue;
    const path = join(directory, entry);
    let stats: ReturnType<typeof statSync>;
    try {
      stats = statSync(path);
    } catch {
      continue;
    }
    if (stats.isDirectory()) {
      walk(path, extensions, exclude, limit, found);
    } else if (extensions.some((extension) => entry.toLowerCase().endsWith(extension))) {
      found.push(path);
    }
  }
}

/** The first heading names the document; a bare filename describes the file, not the decision. */
function titleOf(text: string, fallback: string): string {
  for (const line of text.split(/\r?\n/).slice(0, 60)) {
    const heading = /^#{1,3}\s+(.+?)\s*$/.exec(line);
    if (heading?.[1]) return heading[1];
  }
  return fallback;
}

function summaryOf(text: string): string {
  const body = text
    .split(/\r?\n/)
    .filter((line) => !/^\s*#/.test(line) && line.trim() !== "")
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
  return body.slice(0, DESCRIPTION_CHARS);
}

export class LocalMarkdownConnector implements MCPConnector {
  constructor(
    public readonly name: string,
    private readonly root: string,
    private readonly options: LocalMarkdownOptions = {},
  ) {}

  private files(): string[] {
    const extensions = this.options.extensions ?? DEFAULT_EXTENSIONS;
    const exclude = this.options.exclude ?? DEFAULT_EXCLUDE;
    const limit = this.options.maxFiles ?? DEFAULT_MAX_FILES;
    const roots = (this.options.include ?? [""]).map((part) =>
      part ? join(this.root, part) : this.root,
    );
    const found: string[] = [];
    for (const directory of roots) {
      walk(directory, extensions, exclude, limit, found);
    }
    return found;
  }

  private idOf(path: string): string {
    return relative(this.root, path).split(sep).join("/");
  }

  async listResources(): Promise<MCPResource[]> {
    const resources: MCPResource[] = [];
    for (const path of this.files()) {
      const id = this.idOf(path);
      let text: string;
      try {
        text = readFileSync(path, "utf8");
      } catch {
        continue;
      }
      resources.push({
        id,
        name: titleOf(text, id),
        description: summaryOf(text),
        mimeType: "text/markdown",
      });
    }
    return resources;
  }

  async fetchResource(resourceId: string): Promise<MCPData> {
    // Why: ids are repository-relative by construction; refuse anything that walks
    // out of the root rather than reading an arbitrary file off the machine.
    const path = join(this.root, resourceId);
    const inside = relative(this.root, path);
    if (inside.startsWith("..") || inside === "") {
      throw new Error(`local markdown '${this.name}': resource outside root: ${resourceId}`);
    }
    return {
      resourceId,
      content: readFileSync(path, "utf8"),
      metadata: { source: this.name, path: resourceId },
      fetchedAt: new Date(),
    };
  }

  async search(query: string): Promise<MCPData[]> {
    const needle = query.trim().toLowerCase();
    if (needle === "") return [];
    const hits: MCPData[] = [];
    for (const path of this.files()) {
      let text: string;
      try {
        text = readFileSync(path, "utf8");
      } catch {
        continue;
      }
      if (!text.toLowerCase().includes(needle)) continue;
      const id = this.idOf(path);
      hits.push({
        resourceId: id,
        content: text,
        metadata: { source: this.name, path: id },
        fetchedAt: new Date(),
      });
    }
    return hits;
  }
}
