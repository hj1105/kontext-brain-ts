import path, { resolve } from "node:path";
import {
  FileResourceContentStore,
  type KnowledgeSearchResult,
  LocalKnowledgeSearch,
  type OntologyBuildProgressSink,
  SqliteKnowledgeGraphRepository,
} from "@kontext-brain/core";
import { stringify as stringifyYaml } from "yaml";
import type { AgentConfigKind } from "./agent-mcp-config-import.js";
import {
  GitHubListingError,
  type GitHubRepositoryListing,
  listGitHubRepositories,
} from "./github-repository-listing.js";
import type { KontextAgent } from "./kontext-agent.js";
import {
  readConfigDocument,
  readMCPEntries,
  toOntologyYamlNodes,
  withDefaultLlm,
  withOntology,
  writeConfigDocument,
} from "./kontext-config-file.js";
import { KontextLoader } from "./kontext-loader.js";
import { loadLocalKnowledgePrincipal } from "./local-knowledge-principal.js";
import {
  type LocalKnowledgeRuntime,
  createLocalKnowledgeRuntime,
  resolveKontextDataDirectory,
} from "./local-knowledge-runtime.js";
import { OntologyBuildProgressWriter } from "./ontology-build-progress-file.js";
import { renderResult } from "./ontology-cli-render.js";
import {
  OntologySourceError,
  addSource,
  checkSources,
  importAgentSources,
  summarizeSources,
} from "./ontology-source-inventory.js";

const DEFAULT_CONFIG = "kontext.yaml";

export interface OntologyCliOptions {
  readonly config: string;
  readonly from: readonly AgentConfigKind[];
  readonly project: string | undefined;
  readonly markdown: string | undefined;
  readonly targetNodes: number | undefined;
  /** github-repos: organization or user, as a name or a github.com URL. */
  readonly owner: string | undefined;
  /** setup/query: sidecar data directory holding the knowledge graph. */
  readonly dataDirectory: string | undefined;
  /** query: the question to search the knowledge graph with. */
  readonly question: string | undefined;
  readonly limit: number | undefined;
  /** query: restrict hits to resources on these ontology nodes. */
  readonly nodes: readonly string[];
  readonly write: boolean;
  readonly json: boolean;
  /** Set when --from named something that is not a supported agent. */
  readonly fromError: string | undefined;
  readonly source: {
    readonly name: string | undefined;
    readonly transport: "stdio" | "sse" | "local" | "git" | undefined;
    readonly command: string | undefined;
    readonly args: readonly string[] | undefined;
    readonly url: string | undefined;
    readonly ref: string | undefined;
    readonly path: string | undefined;
    readonly include: readonly string[] | undefined;
    readonly type: string | undefined;
    readonly env: Record<string, string> | undefined;
    readonly code: boolean;
  };
}

function splitList(value: string): string[] {
  return value
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part !== "");
}

export function parseOntologyCliOptions(argv: readonly string[]): OntologyCliOptions {
  let config = DEFAULT_CONFIG;
  let from: AgentConfigKind[] = ["claude", "codex"];
  let project: string | undefined;
  let markdown: string | undefined;
  let targetNodes: number | undefined;
  let owner: string | undefined;
  let dataDirectory: string | undefined;
  let question: string | undefined;
  let limit: number | undefined;
  const nodes: string[] = [];
  let code = false;
  let write = false;
  let json = false;
  let fromError: string | undefined;
  let name: string | undefined;
  let transport: "stdio" | "sse" | "local" | "git" | undefined;
  let command: string | undefined;
  let args: string[] | undefined;
  let url: string | undefined;
  let path: string | undefined;
  let include: string[] | undefined;
  let type: string | undefined;
  let ref: string | undefined;
  let env: Record<string, string> | undefined;

  let flagError: string | undefined;
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    const value = argv[index + 1];
    // Why: without this, `--path --write` stores "--write" as the path and silently
    // drops the write, reporting success while saving nothing.
    const take = (): string | undefined => {
      if (value === undefined || value.startsWith("--")) {
        flagError ??= `${flag} needs a value.`;
        return undefined;
      }
      index += 1;
      return value;
    };
    switch (flag) {
      case "--config":
        config = take() ?? config;
        break;
      case "--from": {
        const raw = take();
        if (raw) {
          const requested = splitList(raw);
          if (requested.length === 0) {
            fromError = "--from needs at least one of: claude, codex.";
          }
          const unknown = requested.filter((part) => part !== "claude" && part !== "codex");
          // Why: dropping an unrecognised name silently would scan nothing and report
          // "no MCP servers found", which reads as an empty machine rather than a typo.
          if (unknown.length > 0) {
            fromError = `Unknown --from value(s): ${unknown.join(", ")}. Use claude and/or codex.`;
          }
          from = requested.filter(
            (part): part is AgentConfigKind => part === "claude" || part === "codex",
          );
        }
        break;
      }
      case "--project": {
        const raw = take();
        if (raw) project = resolve(raw);
        break;
      }
      case "--markdown": {
        const raw = take();
        if (raw) markdown = resolve(raw);
        break;
      }
      case "--target-nodes": {
        const raw = take();
        if (raw) {
          // Why: NaN would reach autoSetup and fail deep inside the builder.
          if (!/^\d+$/.test(raw)) {
            flagError ??= `--target-nodes must be a whole number, got '${raw}'.`;
          } else {
            targetNodes = Number(raw);
          }
        }
        break;
      }
      case "--name":
        name = take();
        break;
      case "--owner":
        owner = take();
        break;
      case "--data-dir":
        dataDirectory = take();
        break;
      case "--question":
        question = take();
        break;
      case "--node": {
        const raw = take();
        if (raw) nodes.push(raw);
        break;
      }
      case "--limit": {
        const raw = take();
        if (raw) {
          if (!/^\d+$/.test(raw)) flagError ??= `--limit must be a whole number, got '${raw}'.`;
          else limit = Number(raw);
        }
        break;
      }
      case "--code":
        code = true;
        break;
      case "--transport": {
        const raw = take();
        if (raw === "stdio" || raw === "sse" || raw === "local" || raw === "git") transport = raw;
        break;
      }
      case "--ref":
        ref = take();
        break;
      case "--env": {
        // Why: a stdio server that needs a token does not start without it, and the
        // config already carries `env`; this is the only way to set it from a surface.
        const raw = take();
        const separator = raw?.indexOf("=") ?? -1;
        if (raw === undefined || separator <= 0) {
          flagError ??= `--env needs KEY=VALUE, got '${raw ?? ""}'.`;
          break;
        }
        env = { ...(env ?? {}), [raw.slice(0, separator)]: raw.slice(separator + 1) };
        break;
      }
      case "--command":
        command = take();
        break;
      case "--arg": {
        // Why: a comma-joined list loses any argument that itself contains a comma,
        // and drops empty ones. Repeating the flag keeps argv exactly as given.
        const raw = take();
        if (raw !== undefined) args = [...(args ?? []), raw];
        break;
      }
      case "--args": {
        const raw = take();
        if (raw) args = splitList(raw);
        break;
      }
      case "--url":
        url = take();
        break;
      case "--path": {
        const raw = take();
        if (raw) path = resolve(raw);
        break;
      }
      case "--include-dir": {
        const raw = take();
        if (raw !== undefined) include = [...(include ?? []), raw];
        break;
      }
      case "--include": {
        const raw = take();
        if (raw) include = splitList(raw);
        break;
      }
      case "--type":
        type = take();
        break;
      case "--write":
        write = true;
        break;
      case "--json":
        json = true;
        break;
      default:
        break;
    }
  }
  return {
    config,
    from,
    project,
    markdown,
    targetNodes,
    write,
    json,
    fromError: fromError ?? flagError,
    owner,
    dataDirectory: resolveKontextDataDirectory(dataDirectory),
    question,
    limit,
    nodes,
    source: { name, transport, command, args, url, ref, path, include, type, env, code },
  };
}

export function ontologyCliUsage(): string {
  return `Usage: kontext-ontology <command> [options]

Commands:
  list         Show the sources the config already names
  import-mcp   Add MCP servers already configured for Claude Code or Codex
  add          Add one source directly, for a provider no other agent knows
  check        Connect every configured source and report what it exposes
  setup        Build the ontology from the connected sources and save it
  github-repos List an organization's repositories to pick sources from
  query        Search the knowledge graph a build wrote (needs --data-dir)
  nodes        List ontology nodes with the documents filed under each (--data-dir)

Options:
  --config <path>        Config file (default: ${DEFAULT_CONFIG})
  --json                 Emit the result as JSON instead of text
  --write                Save changes; without it nothing is written

  --from claude,codex    Which agent configurations to import from (import-mcp)
  --project <dir>        Project whose per-project and .mcp.json servers apply
  --markdown <dir>       Also add the directory's Markdown as a source

  --name <name>          Source name (add)
  --transport stdio|sse|local|git
  --command <command>    stdio: the command that starts the server
  --arg <value>          stdio: one argument; repeat for each (comma-safe)
  --args a,b             stdio: arguments, comma separated (loses embedded commas)
  --url <url>            sse: the server URL; git: the repository to clone
  --ref <name>           git: branch or tag to read (default: the remote default)
  --code                 local/git: read source files too, not only Markdown
  --env KEY=VALUE        stdio: environment for the server; repeat for each
  --path <dir>           local: directory whose Markdown is read
  --include-dir <dir>    local: one subdirectory; repeat for each
  --include a,b          local: subdirectories, comma separated
  --type notion|jira|github_pr|slack

  --target-nodes <n>     Ontology node-count override (setup)
  --owner <org|url>      github-repos: the organization or user to list (uses gh)
  --data-dir <dir>       setup: write documents into this sidecar's knowledge graph
                         (default: $KONTEXT_PLUGIN_DATA; without it only the node
                         schema is saved). query: the graph to search.
  --question <text>      query: what to look for
  --limit <n>            query: hits to return (default 10)
  --node <id>            query: only resources on this ontology node; repeat for each

Run import-mcp or add, then check, then setup.

LLM providers: claude and openai bill per token against an API key. Use
provider: codex to drive your logged-in Codex CLI instead, which a ChatGPT
subscription already covers. ollama runs locally.
`;
}

/** One ontology node with the documents the knowledge graph currently files under it. */
export interface OntologyNodeMembers {
  readonly id: string;
  readonly description: string;
  readonly parentId: string | null;
  /** Null when no data directory was given, so membership could not be read. */
  readonly resourceCount: number | null;
  readonly samples: readonly { title: string; connectorId: string; externalId: string }[];
}

export interface OntologyCliDeps {
  /** Overrides how the agent is built, so setup can run without a paid model. */
  readonly loadAgent?: (
    configPath: string,
    yaml: string,
    knowledge: LocalKnowledgeRuntime | undefined,
    buildProgress: OntologyBuildProgressSink | undefined,
  ) => Promise<KontextAgent>;
  /** Overrides the GitHub lookup, so tests need neither gh nor the network. */
  readonly listRepositories?: (owner: string) => Promise<GitHubRepositoryListing>;
}

export type OntologyCliResult =
  | { command: "list"; ok: true; sources: ReturnType<typeof summarizeSources> }
  | {
      command: "import-mcp";
      ok: true;
      discovered: ReturnType<typeof importAgentSources>["discovered"];
      added: readonly string[];
      written: boolean;
    }
  | { command: "add"; ok: true; name: string; written: boolean }
  | ({ command: "github-repos"; ok: true } & GitHubRepositoryListing)
  | ({ command: "query"; ok: true; dataDirectory: string } & KnowledgeSearchResult)
  | {
      command: "nodes";
      ok: true;
      nodes: readonly OntologyNodeMembers[];
      knowledgeStore: string | null;
    }
  | {
      command: "check";
      ok: boolean;
      sources: Awaited<ReturnType<typeof checkSources>>;
    }
  | {
      command: "setup";
      ok: true;
      nodesCreated: number;
      nodesReused: number;
      documentsClassified: number;
      documentsUnmapped: number;
      nodeIds: readonly string[];
      written: boolean;
      /** The knowledge graph the documents were written into; null when none was given. */
      knowledgeStore: string | null;
      codeFilesSynced: number;
    }
  | { command: string; ok: false; error: string };

async function execute(
  command: string,
  options: OntologyCliOptions,
  deps: OntologyCliDeps,
): Promise<OntologyCliResult> {
  if (options.fromError !== undefined) {
    return { command, ok: false, error: options.fromError };
  }
  if (command === "query") {
    if (!options.dataDirectory) {
      return { command, ok: false, error: "query needs --data-dir (or KONTEXT_PLUGIN_DATA)." };
    }
    if (!options.question) return { command, ok: false, error: "query needs --question." };
    const principal = await loadLocalKnowledgePrincipal(options.dataDirectory);
    const search = new LocalKnowledgeSearch(
      await SqliteKnowledgeGraphRepository.open(options.dataDirectory),
      new FileResourceContentStore(path.join(options.dataDirectory, "knowledge-content")),
    );
    const result = await search.search({
      question: options.question,
      principal,
      ...(options.limit === undefined ? {} : { limit: options.limit }),
      ...(options.nodes.length > 0 ? { ontologyNodeIds: options.nodes } : {}),
    });
    return { command, ok: true, dataDirectory: options.dataDirectory, ...result };
  }
  if (command === "nodes") {
    const document = readConfigDocument(options.config);
    const raw = Array.isArray(document.data.ontology) ? document.data.ontology : [];
    const nodes: OntologyNodeMembers[] = [];
    let graph: SqliteKnowledgeGraphRepository | undefined;
    let organizationId: string | undefined;
    if (options.dataDirectory) {
      organizationId = (await loadLocalKnowledgePrincipal(options.dataDirectory)).organizationId;
      graph = await SqliteKnowledgeGraphRepository.open(options.dataDirectory);
    }
    for (const entry of raw) {
      if (typeof entry !== "object" || entry === null) continue;
      const node = entry as { id?: unknown; description?: unknown; parentId?: unknown };
      if (typeof node.id !== "string") continue;
      const members =
        graph && organizationId
          ? (await graph.listResourcesByOntologyNode(organizationId, node.id)).filter(
              (resource) => resource.status === "active",
            )
          : [];
      nodes.push({
        id: node.id,
        description: typeof node.description === "string" ? node.description : "",
        parentId: typeof node.parentId === "string" ? node.parentId : null,
        resourceCount: graph ? members.length : null,
        samples: members.slice(0, 8).map((resource) => ({
          title: resource.title,
          connectorId: resource.source.connectorId,
          externalId: resource.source.externalId,
        })),
      });
    }
    return { command, ok: true, nodes, knowledgeStore: options.dataDirectory ?? null };
  }
  if (command === "github-repos") {
    // Why: listing needs no config; a workspace without kontext.yaml can still pick sources.
    if (!options.owner) return { command, ok: false, error: "github-repos needs --owner." };
    try {
      const listing = await (deps.listRepositories ?? listGitHubRepositories)(options.owner);
      return { command, ok: true, ...listing };
    } catch (error) {
      if (error instanceof GitHubListingError) return { command, ok: false, error: error.message };
      throw error;
    }
  }
  const document = readConfigDocument(options.config);

  if (command === "list") {
    return { command, ok: true, sources: summarizeSources(document) };
  }

  if (command === "import-mcp") {
    const outcome = importAgentSources(document, {
      from: options.from,
      ...(options.project ? { projectDirectory: options.project } : {}),
      ...(options.markdown ? { markdownRoot: options.markdown } : {}),
    });
    if (options.write && outcome.added.length > 0)
      writeConfigDocument(withDefaultLlm(outcome.document));
    return {
      command,
      ok: true,
      discovered: outcome.discovered,
      added: outcome.added,
      written: options.write && outcome.added.length > 0,
    };
  }

  if (command === "add") {
    const { name, transport } = options.source;
    if (!name || !transport) {
      return { command, ok: false, error: "add needs --name and --transport." };
    }
    const next = addSource(document, {
      name,
      transport,
      ...(options.source.command ? { command: options.source.command } : {}),
      ...(options.source.args ? { args: options.source.args } : {}),
      ...(options.source.url ? { url: options.source.url } : {}),
      ...(options.source.ref ? { ref: options.source.ref } : {}),
      ...(options.source.path ? { path: options.source.path } : {}),
      ...(options.source.include ? { include: options.source.include } : {}),
      ...(options.source.type ? { type: options.source.type } : {}),
      ...(options.source.env ? { env: options.source.env } : {}),
      ...(options.source.code ? { code: true } : {}),
    });
    if (options.write) writeConfigDocument(withDefaultLlm(next));
    return { command, ok: true, name, written: options.write };
  }

  if (command === "check") {
    if (readMCPEntries(document).length === 0) {
      return { command, ok: false, error: `No sources in ${options.config}.` };
    }
    const sources = await checkSources(document);
    return { command, ok: sources.every((source) => source.ok), sources };
  }

  if (command === "setup") {
    if (readMCPEntries(document).length === 0) {
      return { command, ok: false, error: `No sources in ${options.config}.` };
    }
    // Why: a file that only lists sources has no model yet; setup runs it with the
    // default and, when writing, records that choice so the file says what ran.
    const effective = withDefaultLlm(document);
    // Why: with a data directory the build also writes every document's content
    // into the sidecar's knowledge graph; without one only the schema survives.
    const knowledge = options.dataDirectory
      ? await createLocalKnowledgeRuntime(options.dataDirectory, readMCPEntries(effective))
      : undefined;
    // Why a file: stdout carries one JSON result at the end, so a host reading
    // progress during a multi-minute build needs somewhere else to look.
    const progress = options.dataDirectory
      ? new OntologyBuildProgressWriter(options.dataDirectory, options.config)
      : undefined;
    const loadAgent =
      deps.loadAgent ??
      ((
        _path: string,
        yaml: string,
        runtime: LocalKnowledgeRuntime | undefined,
        buildProgress: OntologyBuildProgressSink | undefined,
      ) =>
        KontextLoader.fromYaml(yaml, {
          ...(runtime ? { knowledgeRuntime: runtime } : {}),
          ...(buildProgress ? { buildProgress } : {}),
        }));
    let result: Awaited<ReturnType<KontextAgent["autoSetup"]>>;
    let agent: KontextAgent;
    try {
      agent = await loadAgent(
        options.config,
        stringifyYaml(effective.data, { lineWidth: 0 }),
        knowledge,
        progress?.sink,
      );
      result = await agent.autoSetup(options.targetNodes);
    } catch (error) {
      progress?.finish({
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
    progress?.finish({ ok: true });
    const graph = agent.ontologyGraph;
    const nodes = toOntologyYamlNodes([...graph.nodes.values()], [...graph.edges]);
    if (nodes.length === 0) {
      return { command, ok: false, error: "No ontology nodes were produced." };
    }
    if (options.write) writeConfigDocument(withOntology(effective, nodes));
    return {
      command,
      ok: true,
      nodesCreated: result.nodesCreated,
      nodesReused: result.nodesReused,
      documentsClassified: result.documentsClassified,
      documentsUnmapped: result.documentsUnmapped,
      nodeIds: nodes.map((node) => node.id),
      written: options.write,
      knowledgeStore: knowledge?.dataDirectory ?? null,
      codeFilesSynced: result.codeFilesSynced,
    };
  }

  return { command, ok: false, error: `Unknown command: ${command}` };
}

export async function runOntologyCli(
  argv: readonly string[],
  deps: OntologyCliDeps = {},
): Promise<number> {
  const [command, ...rest] = argv;
  if (!command || command === "--help" || command === "-h" || command === "help") {
    process.stdout.write(ontologyCliUsage());
    return command ? 0 : 1;
  }
  const options = parseOntologyCliOptions(rest);
  let result: OntologyCliResult;
  try {
    result = await execute(command, options, deps);
  } catch (error) {
    if (!(error instanceof OntologySourceError)) throw error;
    result = { command, ok: false, error: error.message };
  }
  process.stdout.write(renderResult(result, options));
  return result.ok ? 0 : 1;
}
