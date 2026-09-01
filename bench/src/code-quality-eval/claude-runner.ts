import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  type CodexCommandRunner,
  type CodexExecutionInput,
  type CodexExecutionResult,
  codexPrompt,
  codexSubscriptionEnvironment,
  runCodexCommand,
} from "./codex-runner.js";

/**
 * The Kontext tools are the same on both hosts, so the treatment prompt is
 * shared. Only process invocation and event shape differ.
 */
export const kontextToolNames = [
  "kontext_prepare_task",
  "kontext_begin_logic",
  "kontext_authorize_write",
  "kontext_refresh_task_context",
  "kontext_check_change",
  "kontext_submit_change_bundle",
  "kontext_propose_transition",
] as const;

export async function writeClaudeMcpConfig(
  repositoryRoot: string,
  pluginDataDirectory: string,
): Promise<string> {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-claude-mcp-"));
  const configPath = path.join(directory, "mcp.json");
  await writeFile(
    configPath,
    `${JSON.stringify(
      {
        mcpServers: {
          kontext_brain: {
            command: process.execPath,
            args: [path.join(repositoryRoot, "plugins", "kontext-brain", "server.mjs")],
            env: { KONTEXT_PLUGIN_DATA: pluginDataDirectory },
          },
        },
      },
      null,
      2,
    )}\n`,
    { mode: 0o600 },
  );
  return configPath;
}

export function claudeArguments(input: {
  readonly arm: CodexExecutionInput["arm"];
  readonly workspacePath: string;
  readonly config: CodexExecutionInput["config"];
  readonly mcpConfigPath?: string;
}): readonly string[] {
  const args = [
    "-p",
    "--output-format",
    "stream-json",
    "--verbose",
    "--model",
    input.config.model,
    "--add-dir",
    input.workspacePath,
    "--max-turns",
    "40",
  ];
  if (input.arm === "kontext") {
    if (!input.mcpConfigPath) {
      throw new Error("Kontext arm requires a private MCP configuration");
    }
    args.push(
      "--mcp-config",
      input.mcpConfigPath,
      "--allowedTools",
      // Only the Kontext tools plus the edit/test primitives the task needs.
      [
        ...kontextToolNames.map((tool) => `mcp__kontext_brain__${tool}`),
        "Read",
        "Edit",
        "Write",
        "Bash",
      ].join(","),
    );
  } else {
    args.push("--allowedTools", ["Read", "Edit", "Write", "Bash"].join(","));
  }
  return args;
}

/**
 * Claude Code reports tool calls as stream-json events whose tool names carry
 * the mcp__<server>__ prefix. extractKontextTools already strips that prefix,
 * so the Codex extractor works unchanged.
 */
export class ClaudeCodeQualityRunner {
  constructor(private readonly commandRunner: CodexCommandRunner = runCodexCommand) {}

  async execute(input: CodexExecutionInput): Promise<CodexExecutionResult> {
    const mcpConfigPath =
      input.arm === "kontext" && input.pluginDataDirectory
        ? await writeClaudeMcpConfig(input.repositoryRoot, input.pluginDataDirectory)
        : undefined;
    const result = await this.commandRunner({
      command: "claude",
      args: claudeArguments({
        arm: input.arm,
        workspacePath: input.workspacePath,
        config: input.config,
        ...(mcpConfigPath ? { mcpConfigPath } : {}),
      }),
      stdin: codexPrompt(input),
      timeoutMilliseconds: input.config.timeoutMilliseconds,
      environment: codexSubscriptionEnvironment(),
    });
    const usage = extractClaudeUsage(result.stdout);
    return {
      ...result,
      ...(usage.inputTokens === undefined ? {} : { inputTokens: usage.inputTokens }),
      ...(usage.outputTokens === undefined ? {} : { outputTokens: usage.outputTokens }),
      kontextToolsObserved: extractClaudeTools(result.stdout),
    };
  }
}

/**
 * Only a tool_use content block proves the model called a tool. Claude Code
 * also streams the server's tool advertisement, whose names look identical, so
 * scanning every "name" field reported the full catalogue for a run that called
 * nothing and made contextConsulted always true.
 */
export function extractClaudeTools(stdout: string): readonly string[] {
  const tools = new Set<string>();
  for (const line of stdout.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      collectToolUse(JSON.parse(line) as unknown, tools);
    } catch {
      // Only structured stream-json events count as proof of a tool call.
    }
  }
  return [...tools].sort();
}

function collectToolUse(value: unknown, tools: Set<string>): void {
  if (!value || typeof value !== "object") return;
  if (Array.isArray(value)) {
    for (const item of value) collectToolUse(item, tools);
    return;
  }
  const record = value as Readonly<Record<string, unknown>>;
  if (record.type === "tool_use" && typeof record.name === "string") {
    const normalized = kontextToolName(record.name);
    if (normalized) tools.add(normalized);
  }
  for (const nested of Object.values(record)) collectToolUse(nested, tools);
}

function kontextToolName(rawName: string): string | undefined {
  const separatorIndex = rawName.lastIndexOf("__");
  const normalized = separatorIndex === -1 ? rawName : rawName.slice(separatorIndex + 2);
  return (kontextToolNames as readonly string[]).includes(normalized) ? normalized : undefined;
}

export function extractClaudeUsage(stdout: string): {
  readonly inputTokens?: number;
  readonly outputTokens?: number;
} {
  let last: { readonly inputTokens?: number; readonly outputTokens?: number } = {};
  for (const line of stdout.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const usage = findUsage(JSON.parse(line) as unknown);
      if (usage.inputTokens !== undefined || usage.outputTokens !== undefined) last = usage;
    } catch {
      // Claude Code may print non-JSON diagnostics alongside its event stream.
    }
  }
  return last;
}

function findUsage(value: unknown): {
  readonly inputTokens?: number;
  readonly outputTokens?: number;
} {
  if (!value || typeof value !== "object") return {};
  if (Array.isArray(value)) {
    for (const item of value) {
      const nested = findUsage(item);
      if (nested.inputTokens !== undefined || nested.outputTokens !== undefined) return nested;
    }
    return {};
  }
  const record = value as Readonly<Record<string, unknown>>;
  const usage = record.usage;
  if (usage && typeof usage === "object") {
    const fields = usage as Readonly<Record<string, unknown>>;
    const input = fields.input_tokens ?? fields.inputTokens;
    const output = fields.output_tokens ?? fields.outputTokens;
    if (typeof input === "number" || typeof output === "number") {
      return {
        ...(typeof input === "number" ? { inputTokens: input } : {}),
        ...(typeof output === "number" ? { outputTokens: output } : {}),
      };
    }
  }
  for (const nested of Object.values(record)) {
    const found = findUsage(nested);
    if (found.inputTokens !== undefined || found.outputTokens !== undefined) return found;
  }
  return {};
}
