import path, { resolve } from "node:path";
import {
  type ChunkEmbeddingOutcome,
  FileResourceContentStore,
  type KnowledgeSearchResult,
  LocalKnowledgeSearch,
  type OntologyBuildProgressSink,
  SqliteKnowledgeGraphRepository,
  embedMissingChunks,
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
  readEmbeddingConfig,
  readMCPEntries,
  toOntologyYamlNodes,
  withDefaultLlm,
  withEmbedding,
  withOntology,
  writeConfigDocument,
} from "./kontext-config-file.js";
import type { MCPConfigDto } from "./kontext-config.js";
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
  type SourceInspection,
  addSource,
  checkSources,
  importAgentSources,
  inspectSource,
  setDocumentMapping,
  summarizeSources,
} from "./ontology-source-inventory.js";
import {
  type EmbeddingProvider,
  type EmbeddingSettings,
  createTextEmbedder,
  readEmbeddingSettings,
  resolveEmbeddingSettings,
  toEmbeddingConfig,
  writeEmbeddingSettings,
} from "./text-embedder-factory.js";

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
  /** embedding: how chunks are embedded for search; only the flags given are set. */
  readonly embedding: {
    readonly provider: EmbeddingProvider | undefined;
    readonly model: string | undefined;
    readonly baseUrl: string | undefined;
    readonly apiKeyEnv: string | undefined;
  };
  readonly source: {
    readonly name: string | undefined;
    readonly transport: "stdio" | "sse" | "http" | "local" | "git" | undefined;
    readonly command: string | undefined;
    readonly args: readonly string[] | undefined;
    readonly url: string | undefined;
    readonly ref: string | undefined;
    readonly path: string | undefined;
    readonly include: readonly string[] | undefined;
    readonly type: string | undefined;
    readonly env: Record<string, string> | undefined;
    readonly code: boolean;
    readonly headers: Record<string, string> | undefined;
    readonly documents: DocumentMapping | undefined;
  };
}

type DocumentMapping = NonNullable<MCPConfigDto["documents"]>;

/** Assembles a tool document mapping from flags; partial flags are a mistake worth naming. */
function documentMapping(
  flags: {
    listTool?: string;
    listArgs?: Record<string, unknown>;
    items?: string;
    id?: string;
    title?: string;
    description?: string;
    readTool?: string;
    readArg?: string;
    readArgs?: Record<string, unknown>;
    content?: string;
  },
  fail: (message: string) => void,
): DocumentMapping | undefined {
  const any = Object.values(flags).some((value) => value !== undefined);
  if (!any) return undefined;
  if (!flags.listTool || !flags.id || !flags.readTool || !flags.readArg) {
    fail("A document mapping needs --list-tool, --id, --read-tool and --read-arg.");
    return undefined;
  }
  return {
    list: {
      tool: flags.listTool,
      ...(flags.listArgs ? { arguments: flags.listArgs } : {}),
      ...(flags.items ? { items: flags.items } : {}),
      id: flags.id,
      ...(flags.title ? { title: flags.title } : {}),
      ...(flags.description ? { description: flags.description } : {}),
    },
    read: {
      tool: flags.readTool,
      idArgument: flags.readArg,
      ...(flags.readArgs ? { arguments: flags.readArgs } : {}),
      ...(flags.content ? { content: flags.content } : {}),
    },
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
  let headers: Record<string, string> | undefined;
  let provider: EmbeddingProvider | undefined;
  let embeddingModel: string | undefined;
  let baseUrl: string | undefined;
  let apiKeyEnv: string | undefined;
  const mapping: {
    listTool?: string;
    listArgs?: Record<string, unknown>;
    items?: string;
    id?: string;
    title?: string;
    description?: string;
    readTool?: string;
    readArg?: string;
    readArgs?: Record<string, unknown>;
    content?: string;
  } = {};
  let write = false;
  let json = false;
  let fromError: string | undefined;
  let name: string | undefined;
  let transport: "stdio" | "sse" | "http" | "local" | "git" | undefined;
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
      case "--provider": {
        const raw = take();
        if (raw === "builtin" || raw === "ollama" || raw === "openai" || raw === "none") {
          provider = raw;
        } else if (raw) {
          flagError ??= `--provider must be builtin, ollama, openai or none, got '${raw}'.`;
        }
        break;
      }
      case "--model":
        embeddingModel = take();
        break;
      case "--base-url":
        baseUrl = take();
        break;
      case "--api-key-env":
        apiKeyEnv = take();
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
      case "--header": {
        const raw = take();
        if (raw) {
          const separator = raw.indexOf("=");
          if (separator <= 0) flagError ??= `--header needs KEY=VALUE, got '${raw}'.`;
          else headers = { ...headers, [raw.slice(0, separator)]: raw.slice(separator + 1) };
        }
        break;
      }
      case "--list-tool":
        mapping.listTool = take();
        break;
      case "--list-args": {
        const raw = take();
        if (raw) {
          try {
            mapping.listArgs = JSON.parse(raw) as Record<string, unknown>;
          } catch {
            flagError ??= "--list-args must be a JSON object.";
          }
        }
        break;
      }
      case "--items":
        mapping.items = take();
        break;
      case "--id":
        mapping.id = take();
        break;
      case "--title":
        mapping.title = take();
        break;
      case "--description":
        mapping.description = take();
        break;
      case "--read-tool":
        mapping.readTool = take();
        break;
      case "--read-arg":
        mapping.readArg = take();
        break;
      case "--read-args": {
        const raw = take();
        if (raw) {
          try {
            mapping.readArgs = JSON.parse(raw) as Record<string, unknown>;
          } catch {
            flagError ??= "--read-args must be a JSON object.";
          }
        }
        break;
      }
      case "--content":
        mapping.content = take();
        break;
      case "--transport": {
        const raw = take();
        if (
          raw === "stdio" ||
          raw === "sse" ||
          raw === "http" ||
          raw === "local" ||
          raw === "git"
        ) {
          transport = raw;
        }
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
    embedding: { provider, model: embeddingModel, baseUrl, apiKeyEnv },
    source: {
      name,
      transport,
      command,
      args,
      url,
      ref,
      path,
      include,
      type,
      env,
      code,
      headers,
      documents: documentMapping(mapping, (message) => {
        flagError ??= message;
      }),
    },
  };
}

export function ontologyCliUsage(): string {
  return `Usage: kontext-ontology <command> [options]

Commands:
  list         Show the sources the config already names
  import-mcp   Add MCP servers already configured for Claude Code or Codex
  add          Add one source directly, for a provider no other agent knows
  inspect      Show what one MCP source exposes: tools and resources (--name)
  map          Set how a tool server lists and reads documents (--name + mapping flags)
  check        Connect every configured source and report what it exposes
  setup        Build the ontology from the connected sources and save it
  github-repos List an organization's repositories to pick sources from
  query        Search the knowledge graph a build wrote (needs --data-dir)
  embedding    Choose how chunks are embedded for search: --provider builtin|ollama|openai|none
  embed        Embed chunks that have no vector yet, without rebuilding (--data-dir)
  nodes        List ontology nodes with the documents filed under each (--data-dir)

Options:
  --config <path>        Config file (default: ${DEFAULT_CONFIG})
  --json                 Emit the result as JSON instead of text
  --write                Save changes; without it nothing is written

  --from claude,codex    Which agent configurations to import from (import-mcp)
  --project <dir>        Project whose per-project and .mcp.json servers apply
  --markdown <dir>       Also add the directory's Markdown as a source

  --name <name>          Source name (add)
  --transport stdio|sse|http|local|git
  --command <command>    stdio: the command that starts the server
  --arg <value>          stdio: one argument; repeat for each (comma-safe)
  --args a,b             stdio: arguments, comma separated (loses embedded commas)
  --url <url>            sse: the server URL; git: the repository to clone
  --ref <name>           git: branch or tag to read (default: the remote default)
  --code                 local/git: read source files too, not only Markdown
  --env KEY=VALUE        stdio: environment for the server; repeat for each
  --header KEY=VALUE     sse/http: request header; a \${NAME} value is read from the
                         environment at run time; repeat for each
  --list-tool <name>     tool servers: the tool that lists documents
  --list-args <json>     fixed arguments for the listing tool
  --items <path>         path to the array of items in its result (default: the result)
  --id <path>            path to a document id inside an item (required with --list-tool)
  --title <path>         path to a title inside an item
  --description <path>   path to a description inside an item
  --read-tool <name>     the tool that reads one document (required with --list-tool)
  --read-arg <name>      the argument that receives the document id (required)
  --read-args <json>     fixed arguments for the reading tool
  --content <path>       path to the text in its result (default: the tool's text)
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
  --provider <name>      embedding: builtin (in-process model, default), ollama, openai, none
  --model <id>           embedding: model for the provider (hub id, Ollama name or API model)
  --base-url <url>       embedding: server address for ollama/openai
  --api-key-env <NAME>   embedding: environment variable holding the openai key

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
  /** Overrides how the embedder is built, so tests never download a model. */
  readonly createEmbedder?: typeof createTextEmbedder;
}

export type OntologyCliResult =
  | {
      command: "list";
      ok: true;
      sources: ReturnType<typeof summarizeSources>;
      /** How search embeds chunks for this workspace, defaults filled in. */
      embedding: EmbeddingSettings;
    }
  | {
      command: "import-mcp";
      ok: true;
      discovered: ReturnType<typeof importAgentSources>["discovered"];
      added: readonly string[];
      written: boolean;
    }
  | { command: "add"; ok: true; name: string; written: boolean }
  | ({ command: "inspect"; ok: true } & SourceInspection)
  | { command: "map"; ok: true; name: string; written: boolean }
  | ({ command: "github-repos"; ok: true } & GitHubRepositoryListing)
  | ({ command: "query"; ok: true; dataDirectory: string } & KnowledgeSearchResult)
  | { command: "embedding"; ok: true; embedding: EmbeddingSettings; written: boolean }
  | ({
      command: "embed";
      ok: true;
      dataDirectory: string;
      embedding: EmbeddingSettings;
    } & ChunkEmbeddingOutcome)
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
      /** Chunks given a vector after the sync; 0 without a data directory or with provider none. */
      chunksEmbedded: number;
      embedding: EmbeddingSettings;
      /** Set when embedding was configured but failed; the build itself still succeeded. */
      embeddingError: string | null;
    }
  | { command: string; ok: false; error: string };

/** Query and embed work from the settings a build recorded beside the graph, else the file's. */
function embeddingSettingsFor(
  document: { readonly data: Record<string, unknown> } | undefined,
  dataDirectory: string,
): EmbeddingSettings {
  return document
    ? resolveEmbeddingSettings(readEmbeddingConfig(document as never))
    : (readEmbeddingSettings(dataDirectory) ?? resolveEmbeddingSettings(undefined));
}

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
    // Why the recorded settings: the question must be embedded in the space the build wrote.
    const settings = embeddingSettingsFor(undefined, options.dataDirectory);
    let embedder: ReturnType<typeof createTextEmbedder> = null;
    let embeddingError: string | undefined;
    try {
      embedder = (deps.createEmbedder ?? createTextEmbedder)(settings, {
        dataDirectory: options.dataDirectory,
      });
    } catch (error) {
      embeddingError = error instanceof Error ? error.message : String(error);
    }
    const search = new LocalKnowledgeSearch(
      await SqliteKnowledgeGraphRepository.open(options.dataDirectory),
      new FileResourceContentStore(path.join(options.dataDirectory, "knowledge-content")),
      embedder,
    );
    const result = await search.search({
      question: options.question,
      principal,
      ...(options.limit === undefined ? {} : { limit: options.limit }),
      ...(options.nodes.length > 0 ? { ontologyNodeIds: options.nodes } : {}),
    });
    return {
      command,
      ok: true,
      dataDirectory: options.dataDirectory,
      ...result,
      ...(embeddingError && !result.embeddingError ? { embeddingError } : {}),
    };
  }
  if (command === "embed") {
    if (!options.dataDirectory) {
      return { command, ok: false, error: "embed needs --data-dir (or KONTEXT_PLUGIN_DATA)." };
    }
    const document = readConfigDocument(options.config);
    const settings = resolveEmbeddingSettings(readEmbeddingConfig(document));
    writeEmbeddingSettings(options.dataDirectory, settings);
    const embedder = (deps.createEmbedder ?? createTextEmbedder)(settings, {
      dataDirectory: options.dataDirectory,
    });
    if (!embedder) {
      return { command, ok: false, error: "embedding.provider is none; nothing to embed." };
    }
    const principal = await loadLocalKnowledgePrincipal(options.dataDirectory);
    const progress = new OntologyBuildProgressWriter(options.dataDirectory, options.config);
    try {
      const outcome = await embedMissingChunks(
        await SqliteKnowledgeGraphRepository.open(options.dataDirectory),
        new FileResourceContentStore(path.join(options.dataDirectory, "knowledge-content")),
        embedder,
        principal.organizationId,
        { onProgress: progress.sink },
      );
      progress.finish({ ok: true });
      return {
        command,
        ok: true,
        dataDirectory: options.dataDirectory,
        embedding: settings,
        ...outcome,
      };
    } catch (error) {
      progress.finish({ ok: false, error: error instanceof Error ? error.message : String(error) });
      throw error;
    }
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
    return {
      command,
      ok: true,
      sources: summarizeSources(document),
      embedding: resolveEmbeddingSettings(readEmbeddingConfig(document)),
    };
  }

  if (command === "embedding") {
    if (!options.embedding.provider) {
      return {
        command,
        ok: false,
        error: "embedding needs --provider builtin|ollama|openai|none.",
      };
    }
    const settings = resolveEmbeddingSettings({
      provider: options.embedding.provider,
      ...(options.embedding.model ? { model: options.embedding.model } : {}),
      ...(options.embedding.baseUrl ? { baseUrl: options.embedding.baseUrl } : {}),
      ...(options.embedding.apiKeyEnv ? { apiKeyEnv: options.embedding.apiKeyEnv } : {}),
    });
    if (options.write) {
      writeConfigDocument(withDefaultLlm(withEmbedding(document, toEmbeddingConfig(settings))));
      // Why also here: search reads the data directory, and a changed choice must reach it
      // before the next build runs.
      if (options.dataDirectory) writeEmbeddingSettings(options.dataDirectory, settings);
    }
    return { command, ok: true, embedding: settings, written: options.write };
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

  if (command === "inspect") {
    if (!options.source.name) return { command, ok: false, error: "inspect needs --name." };
    return { command, ok: true, ...(await inspectSource(document, options.source.name)) };
  }

  if (command === "map") {
    if (!options.source.name) return { command, ok: false, error: "map needs --name." };
    if (!options.source.documents) {
      return {
        command,
        ok: false,
        error: "map needs --list-tool, --id, --read-tool and --read-arg (plus optional paths).",
      };
    }
    const next = setDocumentMapping(document, options.source.name, options.source.documents);
    if (options.write) writeConfigDocument(withDefaultLlm(next));
    return { command, ok: true, name: options.source.name, written: options.write };
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
      ...(options.source.headers ? { headers: options.source.headers } : {}),
      ...(options.source.documents ? { documents: options.source.documents } : {}),
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
    // Why after the sync: vectors index chunks that now exist; a failure here leaves a
    // lexical-only graph, which is still a built ontology, so it is reported, not thrown.
    const embeddingSettings = resolveEmbeddingSettings(readEmbeddingConfig(document));
    let chunksEmbedded = 0;
    let embeddingError: string | null = null;
    if (knowledge && options.dataDirectory) {
      writeEmbeddingSettings(options.dataDirectory, embeddingSettings);
      try {
        const embedder = (deps.createEmbedder ?? createTextEmbedder)(embeddingSettings, {
          dataDirectory: options.dataDirectory,
          onDownload: (event) =>
            progress?.sink({
              phase: "embed",
              done: event.receivedBytes,
              total: event.totalBytes ?? 0,
              message: `download ${event.file}`,
            }),
        });
        if (embedder) {
          const outcome = await embedMissingChunks(
            await SqliteKnowledgeGraphRepository.open(options.dataDirectory),
            new FileResourceContentStore(path.join(options.dataDirectory, "knowledge-content")),
            embedder,
            knowledge.organizationId,
            { onProgress: progress?.sink },
          );
          chunksEmbedded = outcome.chunksEmbedded;
        }
      } catch (error) {
        embeddingError = error instanceof Error ? error.message : String(error);
      }
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
      chunksEmbedded,
      embedding: embeddingSettings,
      embeddingError,
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
