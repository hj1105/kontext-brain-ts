import { spawn } from "node:child_process";
import path from "node:path";
import type { CodeQualityArm, CodeQualityRunConfig, CodeQualityScenario } from "./contracts.js";

const providerApiKeyEnvironmentVariables = [
  "OPENAI_API_KEY",
  "CODEX_API_KEY",
  "ANTHROPIC_API_KEY",
  "GEMINI_API_KEY",
  "GOOGLE_API_KEY",
  "AZURE_OPENAI_API_KEY",
  "NVIDIA_API_KEY",
] as const;

export interface CodexExecutionInput {
  readonly arm: CodeQualityArm;
  readonly scenario: CodeQualityScenario;
  readonly workspacePath: string;
  readonly repositoryRoot: string;
  readonly pluginDataDirectory?: string;
  readonly retrievedContext?: string;
  readonly config: CodeQualityRunConfig;
}

export interface CodexExecutionResult {
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
  readonly durationMilliseconds: number;
  readonly inputTokens?: number;
  readonly outputTokens?: number;
  readonly kontextToolsObserved: readonly string[];
}

export type CodexRuntimeArgumentsInput = Pick<
  CodexExecutionInput,
  "arm" | "workspacePath" | "repositoryRoot" | "pluginDataDirectory" | "config"
>;

export type CodexCommandRunner = (input: {
  readonly command: string;
  readonly args: readonly string[];
  readonly stdin: string;
  readonly timeoutMilliseconds: number;
  readonly environment: NodeJS.ProcessEnv;
  readonly cwd?: string;
}) => Promise<{
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
  readonly durationMilliseconds: number;
}>;

export class CodexCodeQualityRunner {
  constructor(private readonly commandRunner: CodexCommandRunner = runCodexCommand) {}

  async execute(input: CodexExecutionInput): Promise<CodexExecutionResult> {
    const result = await this.commandRunner({
      command: "codex",
      args: codexArguments(input),
      stdin: codexPrompt(input),
      timeoutMilliseconds: input.config.timeoutMilliseconds,
      environment: codexSubscriptionEnvironment(),
    });
    const usage = extractCodexUsage(result.stdout);
    return {
      ...result,
      ...(usage.inputTokens === undefined ? {} : { inputTokens: usage.inputTokens }),
      ...(usage.outputTokens === undefined ? {} : { outputTokens: usage.outputTokens }),
      kontextToolsObserved: extractKontextTools(result.stdout),
    };
  }
}

export function codexSubscriptionEnvironment(
  baseEnvironment: NodeJS.ProcessEnv = process.env,
): NodeJS.ProcessEnv {
  const environment = { ...baseEnvironment };
  for (const name of providerApiKeyEnvironmentVariables) delete environment[name];
  return environment;
}

export function codexArguments(input: CodexExecutionInput): readonly string[] {
  return codexRuntimeArguments(input);
}

/**
 * Builds the hermetic subscription-runtime invocation independently of the
 * fixture prompt. Larger benchmarks reuse this so every arm has identical
 * Codex isolation and the Kontext MCP is fail-closed when requested.
 */
export function codexRuntimeArguments(input: CodexRuntimeArgumentsInput): readonly string[] {
  const args = [
    "exec",
    "--ephemeral",
    "--ignore-user-config",
    "--ignore-rules",
    "--disable",
    "plugins",
    "--disable",
    "remote_plugin",
    "--disable",
    "plugin_sharing",
    "--disable",
    "apps",
    "--disable",
    "browser_use",
    "--disable",
    "browser_use_external",
    "--disable",
    "browser_use_full_cdp_access",
    "--disable",
    "in_app_browser",
    "--disable",
    "skill_mcp_dependency_install",
    "--skip-git-repo-check",
    "--sandbox",
    "workspace-write",
    "--json",
    "--model",
    input.config.model,
    "--config",
    `model_reasoning_effort=${JSON.stringify(input.config.reasoningEffort)}`,
    "--config",
    'approval_policy="never"',
  ];
  if (input.arm === "kontext") {
    if (!input.pluginDataDirectory) {
      throw new Error("Kontext arm requires a private plugin data directory");
    }
    const pluginDirectory = path.join(input.repositoryRoot, "plugins", "kontext-brain");
    args.push(
      "--config",
      `mcp_servers.kontext_brain.command=${JSON.stringify(process.execPath)}`,
      "--config",
      `mcp_servers.kontext_brain.args=[${JSON.stringify(path.join(pluginDirectory, "server.mjs"))}]`,
      "--config",
      `mcp_servers.kontext_brain.cwd=${JSON.stringify(pluginDirectory)}`,
      "--config",
      `mcp_servers.kontext_brain.env={KONTEXT_PLUGIN_DATA=${JSON.stringify(input.pluginDataDirectory)}}`,
      "--config",
      "mcp_servers.kontext_brain.required=true",
      "--config",
      'mcp_servers.kontext_brain.default_tools_approval_mode="approve"',
      "--config",
      'mcp_servers.kontext_brain.enabled_tools=["kontext_prepare_task","kontext_begin_logic","kontext_refresh_task_context","kontext_check_change"]',
    );
  }
  args.push("--cd", input.workspacePath, "-");
  return args;
}

export function codexPrompt(input: CodexExecutionInput): string {
  const shared = [
    "Complete this coding task autonomously.",
    input.scenario.publicPrompt,
    `Work only inside ${input.workspacePath}.`,
    `Edit only ${input.scenario.sourceFile}; do not create or change any other file.`,
    "Do not inspect files outside this workspace for implementation guidance.",
    "Run npm test before finishing. Do not ask questions.",
  ];
  if (input.arm === "baseline") return `${shared.join("\n")}\n`;

  if (input.arm === "rag") {
    return `${shared.join("\n")}

The following internal documentation was retrieved for this task. It may contain
entries that are not relevant. Use whatever applies as the authoritative current
policy, and follow its exact naming.

${input.retrievedContext ?? "No documentation was retrieved."}
`;
  }

  const contract = {
    taskId: input.scenario.taskId,
    intent: input.scenario.intent,
    acceptance: [
      {
        criterionId: `criterion:${input.scenario.scenarioId}:policy`,
        statement: "The target function implements the current approved product policy.",
        verifier: { kind: "test", ref: "workspace:test" },
      },
    ],
    nonGoals: ["Changing the public API", "Editing files outside the target source file"],
    targets: [input.scenario.plannedSymbolId],
    risk: "low",
  } as const;
  const begin = {
    taskId: input.scenario.taskId,
    workspacePath: input.workspacePath,
    logic: {
      workItemId: input.scenario.workItemId,
      plannedSymbolIds: [input.scenario.plannedSymbolId],
    },
    runtimeProvider: "codex",
    receiptTtlSeconds: 900,
    totalTokenBudget: 8_000,
    optionalEvidenceTokenBudget: 2_000,
  } as const;

  return `${shared.join("\n")}

The kontext_brain MCP server is the only source of the private current policy. Follow this workflow:
1. Call kontext_prepare_task with contract=${JSON.stringify(contract)} and a current ISO createdAt.
2. Call kontext_begin_logic with ${JSON.stringify(begin)}.
3. Continue only if editingAllowed is true. Treat returned mandatory Decisions, Domain Terms, and Invariants as authoritative; use Evidence only as provenance.
4. Implement exactly within the receipt's allowed path and use the returned canonical domain language.
5. After the edit, call kontext_check_change for the fast tier and then the targeted tier, using the exact Task ID, Work Item ID, workspace path, a current ISO observedAt, and an ISO nextAttemptAt slightly in the future.
Do not substitute your own policy when Kontext context is available.
`;
}

export function extractKontextTools(stdout: string): readonly string[] {
  return [...new Set(extractKontextToolCalls(stdout).map((call) => call.name))].sort();
}

export interface KontextToolCall {
  readonly callId: string;
  readonly name: string;
}

export function extractKontextToolCalls(stdout: string): readonly KontextToolCall[] {
  const calls = new Map<string, KontextToolCall>();
  for (const [lineIndex, line] of stdout.split(/\r?\n/).entries()) {
    if (!line.trim()) continue;
    try {
      collectToolCalls(JSON.parse(line) as unknown, calls, String(lineIndex));
    } catch {
      // Only structured Codex events count as proof that an MCP tool was called.
    }
  }
  return [...calls.values()].sort(
    (left, right) => left.callId.localeCompare(right.callId) || left.name.localeCompare(right.name),
  );
}

function collectToolCalls(
  value: unknown,
  calls: Map<string, KontextToolCall>,
  fallbackId: string,
): void {
  if (!value || typeof value !== "object") return;
  if (Array.isArray(value)) {
    for (const item of value) collectToolCalls(item, calls, fallbackId);
    return;
  }
  const record = value as Readonly<Record<string, unknown>>;
  if (record.type === "mcp_tool_call") {
    const rawName = [record.tool, record.name, record.tool_name, record.toolName].find(
      (candidate): candidate is string => typeof candidate === "string",
    );
    if (rawName) {
      const separatorIndex = rawName.lastIndexOf("__");
      const name = separatorIndex === -1 ? rawName : rawName.slice(separatorIndex + 2);
      if (/^kontext_[a-z_]+$/.test(name)) {
        const callId = typeof record.id === "string" ? record.id : fallbackId;
        calls.set(`${callId}\u0000${name}`, { callId, name });
      }
    }
  }
  for (const nested of Object.values(record)) collectToolCalls(nested, calls, fallbackId);
}

export function extractCodexUsage(stdout: string): {
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
      // Codex may emit non-JSON diagnostics next to its JSONL event stream.
    }
  }
  return last;
}

function findUsage(value: unknown): {
  readonly inputTokens?: number;
  readonly outputTokens?: number;
} {
  if (!value || typeof value !== "object") return {};
  const object = value as Readonly<Record<string, unknown>>;
  const inputTokens = numericField(object, ["input_tokens", "inputTokens"]);
  const outputTokens = numericField(object, ["output_tokens", "outputTokens"]);
  if (inputTokens !== undefined || outputTokens !== undefined) {
    return {
      ...(inputTokens === undefined ? {} : { inputTokens }),
      ...(outputTokens === undefined ? {} : { outputTokens }),
    };
  }
  for (const nested of Object.values(object)) {
    const usage = findUsage(nested);
    if (usage.inputTokens !== undefined || usage.outputTokens !== undefined) return usage;
  }
  return {};
}

function numericField(
  object: Readonly<Record<string, unknown>>,
  fields: readonly string[],
): number | undefined {
  for (const field of fields) {
    const value = object[field];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return undefined;
}

export async function runCodexCommand(input: {
  readonly command: string;
  readonly args: readonly string[];
  readonly stdin: string;
  readonly timeoutMilliseconds: number;
  readonly environment: NodeJS.ProcessEnv;
  readonly cwd?: string;
}): Promise<{
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
  readonly durationMilliseconds: number;
}> {
  const startedAt = performance.now();
  return await new Promise((resolve, reject) => {
    const child = spawn(input.command, [...input.args], {
      ...(input.cwd ? { cwd: input.cwd } : {}),
      env: input.environment,
      shell: false,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let settled = false;
    let forceKillTimer: NodeJS.Timeout | undefined;
    const settle = (operation: () => void): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timeoutTimer);
      if (forceKillTimer) clearTimeout(forceKillTimer);
      operation();
    };
    const timeoutTimer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
      forceKillTimer = setTimeout(() => child.kill("SIGKILL"), 2_000);
    }, input.timeoutMilliseconds);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });
    child.on("error", (error) => settle(() => reject(error)));
    child.on("close", (code) =>
      settle(() =>
        resolve({
          exitCode: timedOut ? 124 : (code ?? 1),
          stdout,
          stderr: timedOut
            ? `${stderr}\nTimed out after ${input.timeoutMilliseconds}ms`.trim()
            : stderr,
          durationMilliseconds: performance.now() - startedAt,
        }),
      ),
    );
    child.stdin.end(input.stdin);
  });
}
