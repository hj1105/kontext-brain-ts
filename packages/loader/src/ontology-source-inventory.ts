import type { MCPConnector } from "@kontext-brain/mcp";
import {
  type AgentConfigKind,
  discoverAgentMCPServers,
  mergeMCPConfigs,
} from "./agent-mcp-config-import.js";
import {
  type KontextConfigDocument,
  readMCPEntries,
  withMCPEntries,
} from "./kontext-config-file.js";
import type { MCPConfigDto } from "./kontext-config.js";
import { createSourceConnector } from "./ontology-source-connectors.js";

/**
 * Structured results for the ontology source commands. A surface that can only
 * read a text dump cannot show per-source state, so every command answers with
 * data and leaves rendering to the caller.
 */

/** Layer adapters the ontology can apply to a source's documents. */
export const ONTOLOGY_SOURCE_TYPES = ["notion", "jira", "github_pr", "slack"] as const;
export type OntologySourceType = (typeof ONTOLOGY_SOURCE_TYPES)[number];

export type OntologySourceTransport = "stdio" | "sse" | "local" | "git";

export interface OntologySourceSummary {
  readonly name: string;
  readonly transport: OntologySourceTransport;
  readonly type: string | null;
  /** The one address field that applies to this transport. */
  readonly target: string;
}

const KNOWN_TRANSPORTS = new Set<string>(["stdio", "sse", "local", "git"]);

/**
 * A config file can name a transport this build does not know — an `http` entry
 * copied from another agent, say. Reporting the closest known transport keeps one
 * odd row from failing the whole listing and hiding every other source.
 */
export function transportOf(entry: MCPConfigDto): OntologySourceTransport {
  const declared = entry.transport;
  if (declared && KNOWN_TRANSPORTS.has(declared)) {
    return declared as OntologySourceTransport;
  }
  return entry.path ? "local" : entry.command ? "stdio" : "sse";
}

function targetOf(entry: MCPConfigDto): string {
  const transport = transportOf(entry);
  if (transport === "local") return typeof entry.path === "string" ? entry.path : "";
  if (transport === "git") {
    const url = typeof entry.url === "string" ? entry.url : "";
    return entry.ref ? `${url} (${entry.ref})` : url;
  }
  if (transport === "stdio") {
    // Why: `args` written as a scalar would make the spread throw and kill the command.
    const args = Array.isArray(entry.args) ? entry.args : [];
    return [typeof entry.command === "string" ? entry.command : "", ...args].join(" ").trim();
  }
  return typeof entry.url === "string" ? entry.url : "";
}

export function summarizeSources(
  document: KontextConfigDocument,
): readonly OntologySourceSummary[] {
  return readMCPEntries(document).map((entry) => ({
    name: entry.name,
    transport: transportOf(entry),
    type: typeof entry.type === "string" ? entry.type : null,
    target: targetOf(entry),
  }));
}

export interface DiscoveredSourceSummary extends OntologySourceSummary {
  readonly origin: AgentConfigKind | "local-markdown";
  readonly scope: string | null;
  /** True when the config already names it, so importing would change nothing. */
  readonly alreadyPresent: boolean;
}

export interface ImportOutcome {
  readonly discovered: readonly DiscoveredSourceSummary[];
  readonly document: KontextConfigDocument;
  readonly added: readonly string[];
}

export function importAgentSources(
  document: KontextConfigDocument,
  options: {
    readonly from: readonly AgentConfigKind[];
    readonly projectDirectory?: string;
    readonly markdownRoot?: string;
  },
): ImportOutcome {
  const existing = readMCPEntries(document);
  const existingNames = new Set(existing.map((entry) => entry.name));
  const found = discoverAgentMCPServers({
    include: options.from,
    ...(options.projectDirectory ? { projectDirectory: options.projectDirectory } : {}),
  });

  const candidates: MCPConfigDto[] = found.map((item) => item.config);
  const origins = new Map<
    string,
    { origin: DiscoveredSourceSummary["origin"]; scope: string | null }
  >(found.map((item) => [item.config.name, { origin: item.origin, scope: item.scope }]));
  if (options.markdownRoot) {
    const markdown: MCPConfigDto = {
      name: "local-markdown",
      transport: "local",
      path: options.markdownRoot,
    };
    candidates.push(markdown);
    origins.set(markdown.name, { origin: "local-markdown", scope: null });
  }

  const { merged, added } = mergeMCPConfigs(existing, candidates);
  const discovered = candidates.map((config) => {
    const origin = origins.get(config.name);
    return {
      name: config.name,
      transport: transportOf(config),
      type: config.type ?? null,
      target: targetOf(config),
      origin: origin?.origin ?? "local-markdown",
      scope: origin?.scope ?? null,
      alreadyPresent: existingNames.has(config.name),
    };
  });
  return { discovered, document: withMCPEntries(document, merged), added };
}

export interface AddSourceRequest {
  readonly name: string;
  readonly transport: OntologySourceTransport;
  readonly command?: string;
  readonly args?: readonly string[];
  readonly url?: string;
  /** git: branch or tag to read. */
  readonly ref?: string;
  readonly path?: string;
  readonly include?: readonly string[];
  readonly type?: string;
  readonly env?: Readonly<Record<string, string>>;
}

export class OntologySourceError extends Error {
  override readonly name = "OntologySourceError";
}

/**
 * Adds one source the caller described directly, so a provider that is not
 * registered with another agent can still be connected here.
 */
export function addSource(
  document: KontextConfigDocument,
  request: AddSourceRequest,
): KontextConfigDocument {
  const name = request.name.trim();
  if (name === "") throw new OntologySourceError("A source needs a name.");
  const existing = readMCPEntries(document);
  if (existing.some((entry) => entry.name === name)) {
    // Why: silently replacing would discard transport or credentials the user set
    // by hand, and the caller cannot tell that happened from a success result.
    throw new OntologySourceError(`A source named '${name}' already exists.`);
  }
  const entry: MCPConfigDto = { name, transport: request.transport };
  if (request.transport === "stdio") {
    if (!request.command?.trim()) {
      throw new OntologySourceError("A stdio source needs the command that starts it.");
    }
    entry.command = request.command.trim();
    if (request.args && request.args.length > 0) entry.args = [...request.args];
  } else if (request.transport === "sse") {
    if (!request.url?.trim()) throw new OntologySourceError("An SSE source needs its URL.");
    entry.url = request.url.trim();
  } else if (request.transport === "git") {
    if (!request.url?.trim()) {
      throw new OntologySourceError("A git source needs the repository URL to clone.");
    }
    entry.url = request.url.trim();
    if (request.ref?.trim()) entry.ref = request.ref.trim();
    if (request.include && request.include.length > 0) entry.include = [...request.include];
  } else {
    if (!request.path?.trim()) {
      throw new OntologySourceError("A local source needs the directory to read.");
    }
    entry.path = request.path.trim();
    if (request.include && request.include.length > 0) entry.include = [...request.include];
  }
  const type = request.type?.trim();
  if (type) {
    // Why: an unrecognised type is not rejected downstream — the layer factory falls
    // back to Notion — so a typo would silently read the source with the wrong adapter.
    if (!(ONTOLOGY_SOURCE_TYPES as readonly string[]).includes(type)) {
      throw new OntologySourceError(
        `Unknown layer '${type}'. Use one of: ${ONTOLOGY_SOURCE_TYPES.join(", ")}.`,
      );
    }
    entry.type = type;
  }
  if (request.env && Object.keys(request.env).length > 0) entry.env = { ...request.env };
  return withMCPEntries(document, [...existing, entry]);
}

export interface SourceCheckResult {
  readonly name: string;
  readonly ok: boolean;
  readonly resourceCount: number | null;
  readonly error: string | null;
}

/**
 * Closes a probe connector. A stdio source spawns its MCP server on the first
 * request, so a check that only lists resources leaves one server process per
 * source running for the life of the caller.
 */
async function release(connector: unknown): Promise<void> {
  // Why: a source whose entry is incomplete throws before a connector exists, and a
  // throw from the finally would discard every other source's result.
  if (connector === null || connector === undefined) return;
  const close = (connector as { close?: () => Promise<void> }).close;
  if (typeof close !== "function") return;
  try {
    await close.call(connector);
  } catch {
    // A probe that already failed to connect has nothing to close.
  }
}

/**
 * A probe waits on the MCP client's own request timeout, which defaults to a
 * minute. Several dead sources would then outlast the caller's patience and the
 * whole check would be reported as one timeout instead of naming the bad source.
 */
export const SOURCE_CHECK_TIMEOUT_MS = 15_000;

function withDeadline<T>(work: Promise<T>, timeoutMs: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`No answer within ${Math.round(timeoutMs / 1000)}s.`)),
      timeoutMs,
    );
    work.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error: unknown) => {
        clearTimeout(timer);
        reject(error instanceof Error ? error : new Error(String(error)));
      },
    );
  });
}

export async function checkSources(
  document: KontextConfigDocument,
  timeoutMs: number = SOURCE_CHECK_TIMEOUT_MS,
): Promise<readonly SourceCheckResult[]> {
  const results: SourceCheckResult[] = [];
  for (const entry of readMCPEntries(document)) {
    let connector: unknown;
    try {
      connector = createSourceConnector(entry);
      const resources = await withDeadline((connector as MCPConnector).listResources(), timeoutMs);
      results.push({ name: entry.name, ok: true, resourceCount: resources.length, error: null });
    } catch (error) {
      // Why: one unreachable source must not hide the state of the others.
      results.push({
        name: entry.name,
        ok: false,
        resourceCount: null,
        error: error instanceof Error ? error.message : String(error),
      });
    } finally {
      await release(connector);
    }
  }
  return results;
}
