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
