import { randomUUID } from "node:crypto";
import {
  type AgentRuntimePort,
  type RuntimeCapabilitySnapshot,
  type RuntimePlanningInput,
  type RuntimeSession,
  type RuntimeWorkInput,
  createRuntimeCapabilitySnapshot,
} from "@kontext-brain/orchestrator";
import { type RuntimeCommandRunner, SpawnRuntimeCommandRunner } from "./process-runner.js";

/**
 * The Kontext task tool server a worker session must be able to call. Codex
 * loads MCP servers from the user's config, which an isolated runtime worktree
 * never sees; without this the worker is told to consult Kontext Brain and
 * submit a Change Bundle while holding no tool that could do either.
 */
export interface RuntimeMcpServer {
  readonly name: string;
  readonly command: string;
  readonly args: readonly string[];
  readonly env?: Readonly<Record<string, string>>;
  readonly startupTimeoutSeconds?: number;
}

export interface CodexRuntimeAdapterOptions {
  readonly cliPath?: string;
  readonly mcpServer?: RuntimeMcpServer;
  readonly environment?: Readonly<Record<string, string>>;
  readonly allowApiBilling?: boolean;
  readonly timeoutMilliseconds?: number;
  readonly runner?: RuntimeCommandRunner;
  readonly now?: () => Date;
}

export class CodexRuntimeAdapter implements AgentRuntimePort {
  readonly provider = "codex" as const;
  private readonly cliPath: string;
  private readonly mcpServer?: RuntimeMcpServer;
  private readonly environment: Readonly<Record<string, string>>;
  private readonly allowApiBilling: boolean;
  private readonly timeoutMilliseconds: number;
  private readonly runner: RuntimeCommandRunner;
  private readonly now: () => Date;
  private readonly activeExecutions = new Map<string, string>();

  constructor(options: CodexRuntimeAdapterOptions = {}) {
    this.cliPath = options.cliPath ?? "codex";
    this.mcpServer = options.mcpServer;
    this.environment = options.environment ?? stringEnvironment(process.env);
    this.allowApiBilling = options.allowApiBilling ?? false;
    this.timeoutMilliseconds = options.timeoutMilliseconds ?? 2 * 60 * 60_000;
    this.runner = options.runner ?? new SpawnRuntimeCommandRunner();
    this.now = options.now ?? (() => new Date());
  }

  async inspectCapabilities(): Promise<RuntimeCapabilitySnapshot> {
    const inspectedAt = this.now().toISOString();
    try {
      const version = await this.runner.run({
        executionId: randomUUID(),
        command: this.cliPath,
        args: ["--version"],
        cwd: process.cwd(),
        environment: this.environment,
        timeoutMilliseconds: 10_000,
      });
      if (version.exitCode !== 0) throw new Error(version.stderr || "codex --version failed");
      const status = await this.runner.run({
        executionId: randomUUID(),
        command: this.cliPath,
        args: ["login", "status"],
        cwd: process.cwd(),
        environment: this.environment,
        timeoutMilliseconds: 10_000,
      });
      const apiSelected = Boolean(this.environment.CODEX_API_KEY);
      const billingPath = apiSelected
        ? "api"
        : /chatgpt/i.test(`${status.stdout}\n${status.stderr}`)
          ? "subscription"
          : "unknown";
      const authenticated = status.exitCode === 0 && (!apiSelected || this.allowApiBilling);
      return createRuntimeCapabilitySnapshot({
        provider: this.provider,
        cliPath: this.cliPath,
        cliVersion: version.stdout.trim() || version.stderr.trim(),
        installed: true,
        authenticated,
        billingPath,
        supports: {
          structuredOutput: true,
          sessionResume: true,
          mcp: true,
          hooks: true,
          workspaceSandbox: true,
        },
        inspectedAt,
        diagnostic:
          apiSelected && !this.allowApiBilling
            ? "CODEX_API_KEY selects API billing and requires explicit consent"
            : status.exitCode === 0
              ? undefined
              : status.stderr || status.stdout,
      });
    } catch (error) {
      return createRuntimeCapabilitySnapshot({
        provider: this.provider,
        cliPath: this.cliPath,
        installed: false,
        authenticated: false,
        billingPath: "unknown",
        supports: {
          structuredOutput: false,
          sessionResume: false,
          mcp: false,
          hooks: false,
          workspaceSandbox: false,
        },
        inspectedAt,
        diagnostic: error instanceof Error ? error.message : String(error),
      });
    }
  }

  async start(input: RuntimeWorkInput): Promise<RuntimeSession> {
    this.assertBillingPath();
    const review = input.executionRole === "independent_review";
    return this.execute(
      [
        "exec",
        // Why: review judges a diff and returns JSON; it must not be able to write through the tools.
        ...(review ? [] : this.mcpServerOverrides()),
        "--json",
        "--sandbox",
        review ? "read-only" : "workspace-write",
        "--cd",
        input.workspacePath,
        "-",
      ],
      input,
    );
  }

  async plan(input: RuntimePlanningInput): Promise<RuntimeSession> {
    this.assertBillingPath();
    return this.execute(
      ["exec", "--json", "--sandbox", "read-only", "--cd", input.workspacePath, "-"],
      input,
    );
  }

  async resume(providerSessionId: string, input: RuntimeWorkInput): Promise<RuntimeSession> {
    this.assertBillingPath();
    return this.execute(
      ["exec", ...this.mcpServerOverrides(), "resume", providerSessionId, "-", "--json"],
      input,
      providerSessionId,
    );
  }

  /** `-c` overrides that add the task tool server to this one `codex exec` invocation. */
  private mcpServerOverrides(): readonly string[] {
    const server = this.mcpServer;
    if (!server) return [];
    const key = `mcp_servers.${server.name}`;
    const env = Object.entries(server.env ?? {})
      .map(([name, value]) => `${name}=${tomlString(value)}`)
      .join(",");
    return [
      "-c",
      `${key}.command=${tomlString(server.command)}`,
      "-c",
      `${key}.args=[${server.args.map(tomlString).join(",")}]`,
      ...(env ? ["-c", `${key}.env={${env}}`] : []),
      ...(server.startupTimeoutSeconds
        ? ["-c", `${key}.startup_timeout_sec=${server.startupTimeoutSeconds}`]
        : []),
    ];
  }

  async terminate(providerSessionId: string): Promise<void> {
    const executionId = this.activeExecutions.get(providerSessionId);
    if (executionId) await this.runner.terminate(executionId);
  }

  private async execute(
    args: readonly string[],
    input: RuntimeWorkInput | RuntimePlanningInput,
    knownProviderSessionId?: string,
  ): Promise<RuntimeSession> {
    const executionId = randomUUID();
    const sessionId = `runtime-session:${executionId}`;
    const startedAt = this.now().toISOString();
    const events: unknown[] = [];
    let providerSessionId = knownProviderSessionId;
    const result = await this.runner.run({
      executionId,
      command: this.cliPath,
      args,
      cwd: input.workspacePath,
      stdin: workerPrompt(input),
      environment: this.environment,
      timeoutMilliseconds: this.timeoutMilliseconds,
      signal: input.signal,
      onStdoutLine: (line) => {
        try {
          const event = JSON.parse(line) as Record<string, unknown>;
          events.push(event);
          if (event.type === "thread.started" && typeof event.thread_id === "string") {
            providerSessionId = event.thread_id;
            this.activeExecutions.set(providerSessionId, executionId);
          }
        } catch {
          events.push({ type: "unparsed", line });
        }
      },
    });
    if (providerSessionId) this.activeExecutions.delete(providerSessionId);
    const failedEvent = events.some(
      (event) => isRecord(event) && (event.type === "turn.failed" || event.type === "error"),
    );
    const output = [...events]
      .reverse()
      .map(agentMessage)
      .find((message) => message !== undefined);
    return {
      sessionId,
      provider: this.provider,
      providerSessionId,
      status: result.exitCode === 0 && !failedEvent ? "completed" : "failed",
      output,
      events,
      startedAt,
      completedAt: this.now().toISOString(),
      diagnostic:
        result.exitCode === 0 && !failedEvent
          ? undefined
          : result.stderr || `Codex exited with ${result.exitCode}`,
    };
  }

  private assertBillingPath(): void {
    if (this.environment.CODEX_API_KEY && !this.allowApiBilling) {
      throw new Error("Codex API billing is selected but has not been explicitly allowed");
    }
  }
}

/** JSON string escaping is a valid TOML basic string, which is what `-c` parses. */
function tomlString(value: string): string {
  return JSON.stringify(value);
}

function workerPrompt(input: RuntimeWorkInput | RuntimePlanningInput): string {
  if (input.executionRole === "planning") return input.prompt;
  if (input.executionRole === "independent_review") {
    return [
      input.prompt,
      "",
      "Kontext independent review contract:",
      `- Task: ${input.taskId}`,
      `- Context digest: ${input.contextDigest}`,
      `- Integrated code revision: ${input.codeRevision}`,
      "- Work read-only. Do not edit, commit, merge, or invoke implementation agents.",
      "- Judge the diff against the supplied acceptance criteria, normative revisions, Evidence, and verifier output.",
      "- Return only the requested JSON object. Do not wrap it in Markdown.",
    ].join("\n");
  }
  return [
    input.prompt,
    "",
    "Kontext execution contract:",
    `- Task: ${input.taskId}`,
    `- Logic Work Item: ${input.workItem.workItemId}`,
    `- Context digest: ${input.contextDigest}`,
    `- Base code revision: ${input.codeRevision}`,
    `- Capability: ${input.workItem.capabilityId}`,
    `- Planned behavior symbols: ${input.workItem.plannedSymbolIds.join(", ")}`,
    `- Exact allowed paths: ${input.workItem.allowedPaths.join(", ")}`,
    "- Consult Kontext Brain before implementing each behavior-bearing symbol.",
    "- Do not broaden paths, evidence, verifiers, or capability scope.",
    "- Do not merge. Finish by submitting a Change Bundle to the main orchestrator.",
    input.checkpoint
      ? `- Continue from checkpoint ${input.checkpoint.checkpointId}; do not resume its ${input.checkpoint.provider} conversation.`
      : "",
  ]
    .filter(Boolean)
    .join("\n");
}

function stringEnvironment(environment: NodeJS.ProcessEnv): Record<string, string> {
  return Object.fromEntries(
    Object.entries(environment).filter(
      (entry): entry is [string, string] => entry[1] !== undefined,
    ),
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function agentMessage(event: unknown): string | undefined {
  if (
    !isRecord(event) ||
    event.type !== "item.completed" ||
    !isRecord(event.item) ||
    event.item.type !== "agent_message" ||
    typeof event.item.text !== "string"
  ) {
    return undefined;
  }
  return event.item.text;
}
