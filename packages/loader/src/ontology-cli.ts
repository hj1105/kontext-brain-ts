import { resolve } from "node:path";
import type { AgentConfigKind } from "./agent-mcp-config-import.js";
import type { KontextAgent } from "./kontext-agent.js";
import {
  readConfigDocument,
  readMCPEntries,
  toOntologyYamlNodes,
  withOntology,
  writeConfigDocument,
} from "./kontext-config-file.js";
import { KontextLoader } from "./kontext-loader.js";
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
    source: { name, transport, command, args, url, ref, path, include, type, env },
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
  --env KEY=VALUE        stdio: environment for the server; repeat for each
  --path <dir>           local: directory whose Markdown is read
  --include-dir <dir>    local: one subdirectory; repeat for each
  --include a,b          local: subdirectories, comma separated
  --type notion|jira|github_pr|slack

  --target-nodes <n>     Ontology node-count override (setup)

Run import-mcp or add, then check, then setup.

LLM providers: claude and openai bill per token against an API key. Use
provider: codex to drive your logged-in Codex CLI instead, which a ChatGPT
subscription already covers. ollama runs locally.
`;
}

export interface OntologyCliDeps {
  /** Overrides how the agent is built, so setup can run without a paid model. */
  readonly loadAgent?: (configPath: string) => Promise<KontextAgent>;
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
    if (options.write && outcome.added.length > 0) writeConfigDocument(outcome.document);
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
    });
    if (options.write) writeConfigDocument(next);
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
    const loadAgent = deps.loadAgent ?? ((path: string) => KontextLoader.fromFile(path));
    const agent = await loadAgent(options.config);
    const result = await agent.autoSetup(options.targetNodes);
    const graph = agent.ontologyGraph;
    const nodes = toOntologyYamlNodes([...graph.nodes.values()], [...graph.edges]);
    if (nodes.length === 0) {
      return { command, ok: false, error: "No ontology nodes were produced." };
    }
    if (options.write) writeConfigDocument(withOntology(document, nodes));
    return {
      command,
      ok: true,
      nodesCreated: result.nodesCreated,
      nodesReused: result.nodesReused,
      documentsClassified: result.documentsClassified,
      documentsUnmapped: result.documentsUnmapped,
      nodeIds: nodes.map((node) => node.id),
      written: options.write,
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
