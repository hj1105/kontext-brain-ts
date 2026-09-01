import type { LogicWorkItem } from "@kontext-brain/spec";
import { describe, expect, it } from "vitest";
import {
  ClaudeCodeRuntimeAdapter,
  type RuntimeCommandInput,
  type RuntimeCommandResult,
  type RuntimeCommandRunner,
} from "../src/index.js";

describe("ClaudeCodeRuntimeAdapter", () => {
  it("discovers subscription auth and parses official print-mode JSON", async () => {
    const runner = new RecordingRunner([
      { exitCode: 0, stdout: "2.1.215 (Claude Code)\n", stderr: "" },
      {
        exitCode: 0,
        stdout: JSON.stringify({
          loggedIn: true,
          authMethod: "claude.ai",
          apiProvider: "firstParty",
        }),
        stderr: "",
      },
      {
        exitCode: 0,
        stdout: JSON.stringify({
          type: "result",
          subtype: "success",
          is_error: false,
          result: "Change Bundle submitted.",
          session_id: "claude-session-1",
        }),
        stderr: "",
      },
    ]);
    const adapter = new ClaudeCodeRuntimeAdapter({
      runner,
      environment: {},
      pluginPath: "/plugin/kontext-brain",
      now: sequenceClock(),
    });

    expect(await adapter.inspectCapabilities()).toEqual(
      expect.objectContaining({
        installed: true,
        authenticated: true,
        billingPath: "subscription",
      }),
    );
    const session = await adapter.start(workInput());
    expect(session).toEqual(
      expect.objectContaining({
        providerSessionId: "claude-session-1",
        status: "completed",
        output: "Change Bundle submitted.",
      }),
    );
    expect(runner.inputs[2]?.args).toEqual(
      expect.arrayContaining([
        "-p",
        "--output-format",
        "json",
        "--permission-mode",
        "acceptEdits",
        "--plugin-dir",
        "/plugin/kontext-brain",
      ]),
    );
    expect(runner.inputs[2]?.stdin).toContain("Consult Kontext Brain");
  });

  it("surfaces and blocks API billing unless explicitly allowed", async () => {
    const runner = new RecordingRunner([
      { exitCode: 0, stdout: "2.1.215 (Claude Code)\n", stderr: "" },
      {
        exitCode: 0,
        stdout: JSON.stringify({ loggedIn: true, authMethod: "api_key" }),
        stderr: "",
      },
    ]);
    const adapter = new ClaudeCodeRuntimeAdapter({
      runner,
      environment: { ANTHROPIC_API_KEY: "not-a-real-key" },
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

  constructor(private readonly results: RuntimeCommandResult[]) {}

  async run(input: RuntimeCommandInput): Promise<RuntimeCommandResult> {
    this.inputs.push(input);
    const result = this.results.shift();
    if (!result) throw new Error("No mock command result");
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
