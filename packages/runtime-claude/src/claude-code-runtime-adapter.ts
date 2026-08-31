import { randomUUID } from "node:crypto";
import {
  type AgentRuntimePort,
  type RuntimeCapabilitySnapshot,
  type RuntimeSession,
  type RuntimeWorkInput,
  createRuntimeCapabilitySnapshot,
} from "@kontext-brain/orchestrator";
import { type RuntimeCommandRunner, SpawnRuntimeCommandRunner } from "./process-runner.js";

export interface ClaudeCodeRuntimeAdapterOptions {
  readonly cliPath?: string;
  readonly pluginPath?: string;
  readonly environment?: Readonly<Record<string, string>>;
  readonly allowApiBilling?: boolean;
  readonly maxTurns?: number;
  readonly timeoutMilliseconds?: number;
  readonly runner?: RuntimeCommandRunner;
  readonly now?: () => Date;
}

export class ClaudeCodeRuntimeAdapter implements AgentRuntimePort {
  readonly provider = "claude" as const;
  private readonly cliPath: string;
  private readonly pluginPath?: string;
  private readonly environment: Readonly<Record<string, string>>;
  private readonly allowApiBilling: boolean;
  private readonly maxTurns: number;
  private readonly timeoutMilliseconds: number;
  private readonly runner: RuntimeCommandRunner;
  private readonly now: () => Date;
  private readonly activeExecutions = new Map<string, string>();

  constructor(options: ClaudeCodeRuntimeAdapterOptions = {}) {
    this.cliPath = options.cliPath ?? "claude";
    this.pluginPath = options.pluginPath;
    this.environment = options.environment ?? stringEnvironment(process.env);
    this.allowApiBilling = options.allowApiBilling ?? false;
    this.maxTurns = options.maxTurns ?? 50;
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
      if (version.exitCode !== 0) throw new Error(version.stderr || "claude --version failed");
      const status = await this.runner.run({
        executionId: randomUUID(),
        command: this.cliPath,
        args: ["auth", "status", "--json"],
        cwd: process.cwd(),
        environment: this.environment,
        timeoutMilliseconds: 10_000,
      });
      const parsed = parseJson(status.stdout);
      const apiSelected =
        Boolean(this.environment.ANTHROPIC_API_KEY) ||
        (isRecord(parsed) &&
          typeof parsed.authMethod === "string" &&
          /api.?key/i.test(parsed.authMethod));
      const loggedIn = isRecord(parsed) && parsed.loggedIn === true;
      const billingPath = apiSelected ? "api" : loggedIn ? "subscription" : "unknown";
      const authenticated =
        status.exitCode === 0 && loggedIn && (!apiSelected || this.allowApiBilling);
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
            ? "ANTHROPIC_API_KEY or API-key auth selects API billing and requires explicit consent"
            : authenticated
              ? undefined
              : status.stderr || "Claude Code is not authenticated",
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
    const providerSessionId = randomUUID();
    return this.execute(
      [
        "-p",
        "--output-format",
        "json",
        "--permission-mode",
        input.executionRole === "independent_review" ? "plan" : "acceptEdits",
        "--max-turns",
        String(this.maxTurns),
        "--session-id",
        providerSessionId,
        ...this.pluginArgs(),
      ],
      input,
      providerSessionId,
    );
  }

  async resume(providerSessionId: string, input: RuntimeWorkInput): Promise<RuntimeSession> {
    this.assertBillingPath();
    return this.execute(
      [
        "-p",
        "--output-format",
        "json",
        "--permission-mode",
        "acceptEdits",
        "--max-turns",
        String(this.maxTurns),
        "--resume",
        providerSessionId,
        ...this.pluginArgs(),
      ],
      input,
      providerSessionId,
    );
  }

  async terminate(providerSessionId: string): Promise<void> {
    const executionId = this.activeExecutions.get(providerSessionId);
    if (executionId) await this.runner.terminate(executionId);
  }

  private async execute(
    args: readonly string[],
    input: RuntimeWorkInput,
    providerSessionId: string,
  ): Promise<RuntimeSession> {
    const executionId = randomUUID();
    const startedAt = this.now().toISOString();
    this.activeExecutions.set(providerSessionId, executionId);
    try {
      const result = await this.runner.run({
        executionId,
        command: this.cliPath,
        args,
        cwd: input.workspacePath,
        stdin: workerPrompt(input),
        environment: this.environment,
        timeoutMilliseconds: this.timeoutMilliseconds,
        signal: input.signal,
      });
      const event = parseJson(result.stdout);
      const actualProviderSessionId =
        isRecord(event) && typeof event.session_id === "string"
          ? event.session_id
          : providerSessionId;
      const failed =
        result.exitCode !== 0 ||
        !isRecord(event) ||
        event.is_error === true ||
        (typeof event.subtype === "string" && event.subtype !== "success");
      return {
        sessionId: `runtime-session:${executionId}`,
        provider: this.provider,
        providerSessionId: actualProviderSessionId,
        status: failed ? "failed" : "completed",
        output: isRecord(event) && typeof event.result === "string" ? event.result : undefined,
        events: event === undefined ? [] : [event],
        startedAt,
        completedAt: this.now().toISOString(),
        diagnostic: failed
          ? result.stderr ||
            (isRecord(event) && typeof event.result === "string"
              ? event.result
              : `Claude exited with ${result.exitCode}`)
          : undefined,
      };
    } finally {
      this.activeExecutions.delete(providerSessionId);
    }
  }

  private pluginArgs(): readonly string[] {
    return this.pluginPath ? ["--plugin-dir", this.pluginPath] : [];
  }

  private assertBillingPath(): void {
    if (this.environment.ANTHROPIC_API_KEY && !this.allowApiBilling) {
      throw new Error("Claude API billing is selected but has not been explicitly allowed");
    }
  }
}

function workerPrompt(input: RuntimeWorkInput): string {
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

function parseJson(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return undefined;
  }
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
