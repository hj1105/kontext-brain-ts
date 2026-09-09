import type { LogicWorkItem } from "@kontext-brain/spec";
import { describe, expect, it } from "vitest";
import {
  CodexRuntimeAdapter,
  type RuntimeCommandInput,
  type RuntimeCommandResult,
  type RuntimeCommandRunner,
} from "../src/index.js";

describe("CodexRuntimeAdapter", () => {
  it("discovers ChatGPT subscription auth and parses official codex exec JSONL", async () => {
    const runner = new RecordingRunner([
      { exitCode: 0, stdout: "codex-cli 0.144.6\n", stderr: "" },
      { exitCode: 0, stdout: "Logged in using ChatGPT\n", stderr: "" },
      {
        exitCode: 0,
        stdout: "",
        stderr: "",
        lines: [
          JSON.stringify({ type: "thread.started", thread_id: "codex-session-1" }),
          JSON.stringify({
            type: "item.completed",
            item: { type: "agent_message", text: "Change Bundle submitted." },
          }),
          JSON.stringify({ type: "turn.completed", usage: { input_tokens: 10 } }),
        ],
      },
    ]);
    const adapter = new CodexRuntimeAdapter({
      runner,
      environment: {},
      now: sequenceClock(),
    });

    const capability = await adapter.inspectCapabilities();
    const session = await adapter.start(workInput());

    expect(capability).toEqual(
      expect.objectContaining({
        installed: true,
        authenticated: true,
        billingPath: "subscription",
      }),
    );
    expect(session).toEqual(
      expect.objectContaining({
        providerSessionId: "codex-session-1",
        status: "completed",
        output: "Change Bundle submitted.",
      }),
    );
    expect(runner.inputs[2]?.args).toEqual([
      "exec",
      "--json",
      "--sandbox",
      "workspace-write",
      "--cd",
      "/workspace",
      "-",
    ]);
    expect(runner.inputs[2]?.stdin).toContain("Consult Kontext Brain");
  });

  it("surfaces and blocks API billing unless explicitly allowed", async () => {
    const runner = new RecordingRunner([
      { exitCode: 0, stdout: "codex-cli 0.144.6\n", stderr: "" },
      { exitCode: 0, stdout: "Logged in using ChatGPT\n", stderr: "" },
    ]);
    const adapter = new CodexRuntimeAdapter({
      runner,
      environment: { CODEX_API_KEY: "not-a-real-key" },
    });

    expect(await adapter.inspectCapabilities()).toEqual(
      expect.objectContaining({
        authenticated: false,
        billingPath: "api",
        diagnostic: expect.stringContaining("explicit consent"),
      }),
    );
    await expect(adapter.start(workInput())).rejects.toThrow("not been explicitly allowed");
  });

  it("runs independent review in a read-only sandbox", async () => {
    const runner = new RecordingRunner([
      {
        exitCode: 0,
        stdout: "",
        stderr: "",
        lines: [
          JSON.stringify({ type: "thread.started", thread_id: "codex-review-1" }),
          JSON.stringify({
            type: "item.completed",
            item: {
              type: "agent_message",
              text: '{"verdict":"passed","findings":[]}',
            },
          }),
        ],
      },
    ]);
    const adapter = new CodexRuntimeAdapter({ runner, environment: {} });

    await adapter.start({ ...workInput(), executionRole: "independent_review" });

    expect(runner.inputs[0]?.args).toContain("read-only");
    expect(runner.inputs[0]?.stdin).toContain("Work read-only");
    expect(runner.inputs[0]?.stdin).not.toContain("Change Bundle to the main orchestrator");
  });
});

class RecordingRunner implements RuntimeCommandRunner {
  readonly inputs: RuntimeCommandInput[] = [];

  constructor(
    private readonly results: Array<RuntimeCommandResult & { readonly lines?: readonly string[] }>,
  ) {}

  async run(input: RuntimeCommandInput): Promise<RuntimeCommandResult> {
    this.inputs.push(input);
    const result = this.results.shift();
    if (!result) throw new Error("No mock command result");
    for (const line of result.lines ?? []) input.onStdoutLine?.(line);
    return result;
  }

  async terminate(): Promise<void> {}
}

function workInput() {
  const workItem: LogicWorkItem = {
    workItemId: "work-item:handler",
    taskId: "task:runtime",
    plannedSymbolIds: ["symbol:handler"],
    dependsOn: [],
    allowedPaths: ["src/handler.ts"],
    requiredVerifiers: [],
    capabilityId: "capability:handler",
  };
  return {
    taskId: workItem.taskId,
    workItem,
    workspacePath: "/workspace",
    prompt: "Implement the handler.",
    codeRevision: "commit:base",
    contextDigest: "context:current",
  };
}

function sequenceClock(): () => Date {
  let milliseconds = Date.parse("2026-08-29T00:00:00.000Z");
  return () => {
    const value = new Date(milliseconds);
    milliseconds += 1_000;
    return value;
  };
}

describe("CodexRuntimeAdapter worker tool server", () => {
  const mcpServer = {
    name: "kontext_brain",
    command: "C:\\Program Files\\Kondex\\Kondex.exe",
    args: ["/data/plugins/kontext-brain/server.mjs"],
    env: { KONTEXT_PLUGIN_DATA: "/data/kontext", ELECTRON_RUN_AS_NODE: "1" },
    startupTimeoutSeconds: 30,
    toolsApprovalMode: "auto" as const,
  };
  const workerRun = { exitCode: 0, stdout: "", stderr: "", lines: [] as string[] };

  it("hands an implementation session the task tool server as config overrides", async () => {
    const runner = new RecordingRunner([workerRun]);
    const adapter = new CodexRuntimeAdapter({ runner, environment: {}, mcpServer });
    await adapter.start(workInput());
    const args = runner.inputs[0]?.args ?? [];
    expect(args.slice(0, 11)).toEqual([
      "exec",
      "-c",
      'mcp_servers.kontext_brain.command="C:\\\\Program Files\\\\Kondex\\\\Kondex.exe"',
      "-c",
      'mcp_servers.kontext_brain.args=["/data/plugins/kontext-brain/server.mjs"]',
      "-c",
      'mcp_servers.kontext_brain.env={KONTEXT_PLUGIN_DATA="/data/kontext",ELECTRON_RUN_AS_NODE="1"}',
      "-c",
      "mcp_servers.kontext_brain.startup_timeout_sec=30",
      "-c",
      'mcp_servers.kontext_brain.default_tools_approval_mode="auto"',
    ]);
    expect(args.slice(-6)).toEqual([
      "--json",
      "--sandbox",
      "workspace-write",
      "--cd",
      "/workspace",
      "-",
    ]);
  });

  it("keeps the tool server away from read-only review and planning", async () => {
    const runner = new RecordingRunner([workerRun, workerRun]);
    const adapter = new CodexRuntimeAdapter({ runner, environment: {}, mcpServer });
    await adapter.start({ ...workInput(), executionRole: "independent_review" });
    await adapter.plan({
      executionRole: "planning",
      planningId: "plan-1",
      workspacePath: "/workspace",
      prompt: "plan",
      codeRevision: "rev",
      contextDigest: "sha256:digest",
    });
    for (const input of runner.inputs) {
      expect(input.args).not.toContain("-c");
      expect(input.args).toContain("read-only");
    }
  });

  it("resumes with the same server so a checkpointed worker keeps its tools", async () => {
    const runner = new RecordingRunner([workerRun]);
    const adapter = new CodexRuntimeAdapter({ runner, environment: {}, mcpServer });
    await adapter.resume("codex-session-1", workInput());
    expect(runner.inputs[0]?.args.slice(0, 2)).toEqual(["exec", "-c"]);
    expect(runner.inputs[0]?.args).toContain("resume");
  });

  it("adds nothing when no server is configured", async () => {
    const runner = new RecordingRunner([workerRun]);
    const adapter = new CodexRuntimeAdapter({ runner, environment: {} });
    await adapter.start(workInput());
    expect(runner.inputs[0]?.args).not.toContain("-c");
  });
});
