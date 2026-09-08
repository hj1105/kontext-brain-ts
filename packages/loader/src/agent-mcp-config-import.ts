import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { MCPConfigDto } from "./kontext-config.js";

/**
 * Imports MCP server definitions a user already maintains for Claude Code or
 * Codex, so connecting an ontology source does not mean re-authoring transport
 * details that exist elsewhere on the machine.
 */

export type AgentConfigKind = "claude" | "codex";

export interface DiscoveredMCPServer {
  readonly config: MCPConfigDto;
  /** Which agent's configuration the definition came from. */
  readonly origin: AgentConfigKind;
  /** File the definition was read from, for reporting. */
  readonly sourcePath: string;
  /** Project path the definition is scoped to, when it is not user-global. */
  readonly scope: string | null;
}

export interface DiscoverOptions {
  readonly homeDirectory?: string;
  /** Project directory whose local `.mcp.json` and per-project entries apply. */
  readonly projectDirectory?: string;
  readonly include?: readonly AgentConfigKind[];
}

/**
 * Layer names the ontology already has adapters for. Matching is by whole word
 * so an unrelated server such as `github-actions-linter` is not typed as a PR
 * source, which would hand its documents to the wrong layer adapter.
 */
const LAYER_PATTERNS: ReadonlyArray<readonly [RegExp, string]> = [
  [/(^|[^a-z])notion([^a-z]|$)/i, "notion"],
  [/(^|[^a-z])jira([^a-z]|$)/i, "jira"],
  [/(^|[^a-z])slack([^a-z]|$)/i, "slack"],
  [/(^|[^a-z])github[-_]?pr([^a-z]|$)/i, "github_pr"],
];

export function inferLayerType(name: string): string | undefined {
  for (const [pattern, type] of LAYER_PATTERNS) {
    if (pattern.test(name)) return type;
  }
  return undefined;
}

function readJsonFile(path: string): unknown {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return undefined;
  }
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function asStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const out = value.filter((item): item is string => typeof item === "string");
  return out.length === value.length ? out : undefined;
}

function asStringRecord(value: unknown): Record<string, string> | undefined {
  const record = asRecord(value);
  if (!record) return undefined;
  const out: Record<string, string> = {};
  for (const [key, item] of Object.entries(record)) {
    if (typeof item === "string") out[key] = item;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

/** Shape shared by Claude's `mcpServers` entries and Codex's `[mcp_servers.*]`. */
function toConfig(name: string, entry: Record<string, unknown>): MCPConfigDto | undefined {
  const command = typeof entry.command === "string" ? entry.command : undefined;
  const url = typeof entry.url === "string" ? entry.url : undefined;
  if (!command && !url) return undefined;
  const args = asStringArray(entry.args);
  const env = asStringRecord(entry.env);
  const type = inferLayerType(name);
  return {
    name,
    ...(command ? { command } : {}),
    ...(url ? { url } : {}),
    ...(args && args.length > 0 ? { args } : {}),
    ...(env ? { env } : {}),
    ...(type ? { type } : {}),
    transport: command ? "stdio" : "sse",
  };
}

function collectServers(
  servers: unknown,
  origin: AgentConfigKind,
  sourcePath: string,
  scope: string | null,
  into: DiscoveredMCPServer[],
): void {
  const record = asRecord(servers);
  if (!record) return;
  for (const [name, raw] of Object.entries(record)) {
    const entry = asRecord(raw);
    if (!entry) continue;
    const config = toConfig(name, entry);
    if (config) into.push({ config, origin, sourcePath, scope });
  }
}

function discoverClaude(
  home: string,
  projectDirectory: string | undefined,
  into: DiscoveredMCPServer[],
): void {
  const userConfigPath = join(home, ".claude.json");
  const userConfig = asRecord(readJsonFile(userConfigPath));
  if (userConfig) {
    collectServers(userConfig.mcpServers, "claude", userConfigPath, null, into);
    const projects = asRecord(userConfig.projects);
    if (projects) {
      for (const [path, raw] of Object.entries(projects)) {
        // Why: a per-project block only applies to that project; importing every
        // project's servers would connect sources the current ontology never sees.
        if (projectDirectory && path !== projectDirectory) continue;
        const project = asRecord(raw);
        if (project) collectServers(project.mcpServers, "claude", userConfigPath, path, into);
      }
    }
  }
  if (projectDirectory) {
    const localPath = join(projectDirectory, ".mcp.json");
    const local = asRecord(readJsonFile(localPath));
    if (local) collectServers(local.mcpServers, "claude", localPath, projectDirectory, into);
  }
}

/**
 * Minimal reader for the `[mcp_servers.*]` tables Codex writes. Only the keys
 * an MCP client needs are read; anything else in the file is ignored rather
 * than parsed, so this does not pretend to be a general TOML parser.
 */
export function parseCodexMcpServers(toml: string): Record<string, Record<string, unknown>> {
  const servers: Record<string, Record<string, unknown>> = {};
  let current: { name: string; env: boolean } | null = null;
  for (const rawLine of toml.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (line === "" || line.startsWith("#")) continue;
    const table = /^\[([^\]]+)\]$/.exec(line);
    if (table) {
      const path = table[1] ?? "";
      const envMatch = /^mcp_servers\.(.+)\.env$/.exec(path);
      const serverMatch = /^mcp_servers\.([^.]+)$/.exec(path);
      if (envMatch?.[1]) current = { name: envMatch[1], env: true };
      else if (serverMatch?.[1]) current = { name: serverMatch[1], env: false };
      else current = null;
      if (current) servers[current.name] ??= {};
      continue;
    }
    if (!current) continue;
    const pair = /^([A-Za-z0-9_-]+)\s*=\s*(.+)$/.exec(line);
    if (!pair) continue;
    const key = pair[1] as string;
    const value = parseTomlValue(pair[2] as string);
    if (value === undefined) continue;
    const target = servers[current.name] as Record<string, unknown>;
    if (current.env) {
      const env = asRecord(target.env) ?? {};
      env[key] = value;
      target.env = env;
    } else {
      target[key] = value;
    }
  }
  return servers;
}

function parseTomlValue(raw: string): string | string[] | number | undefined {
  const text = raw.replace(/\s+#.*$/, "").trim();
  if (text.startsWith("[") && text.endsWith("]")) {
    const inner = text.slice(1, -1).trim();
    if (inner === "") return [];
    const items = inner.split(",").map((item) => item.trim());
    const parsed = items.map((item) => parseTomlValue(item));
    return parsed.every((item): item is string => typeof item === "string") ? parsed : undefined;
  }
  if (
    (text.startsWith('"') && text.endsWith('"')) ||
    (text.startsWith("'") && text.endsWith("'"))
  ) {
    return text.slice(1, -1);
  }
  const numeric = Number(text);
  return Number.isFinite(numeric) ? numeric : undefined;
}

function discoverCodex(home: string, into: DiscoveredMCPServer[]): void {
  const path = join(home, ".codex", "config.toml");
  let toml: string;
  try {
    toml = readFileSync(path, "utf8");
  } catch {
    return;
  }
  collectServers(parseCodexMcpServers(toml), "codex", path, null, into);
}

export function discoverAgentMCPServers(
  options: DiscoverOptions = {},
): readonly DiscoveredMCPServer[] {
  const home = options.homeDirectory ?? homedir();
  const include = options.include ?? (["claude", "codex"] as const);
  const found: DiscoveredMCPServer[] = [];
  if (include.includes("claude")) discoverClaude(home, options.projectDirectory, found);
  if (include.includes("codex")) discoverCodex(home, found);
  return dedupeByName(found);
}

/** Keeps the first definition of each name; later duplicates are reported, not merged. */
function dedupeByName(found: readonly DiscoveredMCPServer[]): DiscoveredMCPServer[] {
  const seen = new Set<string>();
  const out: DiscoveredMCPServer[] = [];
  for (const item of found) {
    if (seen.has(item.config.name)) continue;
    seen.add(item.config.name);
    out.push(item);
  }
  return out;
}

export interface MergeResult {
  readonly merged: readonly MCPConfigDto[];
  readonly added: readonly string[];
  readonly kept: readonly string[];
}

/**
 * Adds discovered servers that the config does not already name. An existing
 * entry is never rewritten: the user's own transport or type choice outranks
 * whatever another agent's configuration happens to say today.
 */
export function mergeMCPConfigs(
  existing: readonly MCPConfigDto[],
  discovered: readonly MCPConfigDto[],
): MergeResult {
  const byName = new Map(existing.map((entry) => [entry.name, entry]));
  const added: string[] = [];
  const kept: string[] = [];
  const merged = [...existing];
  for (const candidate of discovered) {
    if (byName.has(candidate.name)) {
      kept.push(candidate.name);
      continue;
    }
    merged.push(candidate);
    added.push(candidate.name);
  }
  return { merged, added, kept };
}
