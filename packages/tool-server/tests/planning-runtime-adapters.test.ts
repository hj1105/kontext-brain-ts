import { ClaudeCodeRuntimeAdapter } from "@kontext-brain/runtime-claude";
import { CodexRuntimeAdapter } from "@kontext-brain/runtime-codex";
import type { RuntimeCommandInput } from "@kontext-brain/runtime-codex";
import { expect, it, vi } from "vitest";

it.each(["codex", "claude"] as const)(
  "%s plans through the existing parser and constrained CLI mode, without a worker contract",
  async (provider) => {
    const output = '{"contract":{},"logicPlans":[]}';
    const runner = {
      run: vi.fn(async (input: RuntimeCommandInput) => {
        input.onStdoutLine?.(JSON.stringify({ type: "thread.started", thread_id: "plan-session" }));
        input.onStdoutLine?.(
          JSON.stringify({ type: "item.completed", item: { type: "agent_message", text: output } }),
        );
        return {
          exitCode: 0,
          stdout: JSON.stringify({ subtype: "success", result: output }),
          stderr: "",
        };
      }),
      terminate: vi.fn(async () => undefined),
    };
    const adapter =
      provider === "codex"
        ? new CodexRuntimeAdapter({ runner, environment: {} })
        : new ClaudeCodeRuntimeAdapter({
            runner,
            environment: {},
            pluginPath: "/not-attached-to-planner",
          });
    const controller = new AbortController();
    const result = await adapter.plan({
      executionRole: "planning",
      planningId: "plan:fixture",
      workspacePath: "/fixture",
      codeRevision: "commit:fixture",
      contextDigest: "context:fixture",
      prompt: "Only propose a plan",
      signal: controller.signal,
    });
    expect(result.output).toBe(output);
    const command = runner.run.mock.calls[0][0];
    expect(command.stdin).toBe("Only propose a plan");
    expect(command.args).toContain(provider === "codex" ? "read-only" : "plan");
    expect(command.args).not.toContain("workspace-write");
    expect(command.args).not.toContain("acceptEdits");
    expect(command.args).not.toContain("--plugin-dir");
    expect(command.signal).toBe(controller.signal);
  },
);
