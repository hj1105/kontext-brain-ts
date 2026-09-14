import { describe, expect, it } from "vitest";
import {
  CodexCodeQualityRunner,
  codexArguments,
  codexPrompt,
  codexSubscriptionEnvironment,
} from "./codex-runner.js";
import { codeQualityScenarios } from "./scenarios.js";

const scenario = codeQualityScenarios[0];
if (!scenario) throw new Error("Expected a code-quality scenario");
const config = {
  runtime: "codex" as const,
  model: "test-model",
  reasoningEffort: "medium" as const,
  repetitions: 1,
  timeoutMilliseconds: 1_000,
};

describe("CodexCodeQualityRunner", () => {
  it("keeps the baseline free of Kontext configuration and private policy", () => {
    const input = {
      arm: "baseline" as const,
      scenario,
      workspacePath: "/tmp/workspace",
      repositoryRoot: "/repo",
      config,
    };
    expect(codexArguments(input).join(" ")).not.toContain("mcp_servers.kontext_brain");
    expect(codexArguments(input)).toContain('approval_policy="never"');
    const prompt = codexPrompt(input);
    expect(prompt).not.toContain("kontext_prepare_task");
    for (const rule of scenario.rules) {
      expect(prompt).not.toContain(rule.evidenceText);
      if (rule.statement) expect(prompt).not.toContain(rule.statement);
      if (rule.definition) expect(prompt).not.toContain(rule.definition);
    }
  });

  it("configures only the treatment MCP and does not leak normative content", () => {
    const input = {
      arm: "kontext" as const,
      scenario,
      workspacePath: "/tmp/workspace",
      repositoryRoot: "/repo",
      pluginDataDirectory: "/tmp/plugin-data",
      config,
    };
    expect(codexArguments(input).join(" ")).toContain("mcp_servers.kontext_brain");
    expect(codexArguments(input)).toContain("mcp_servers.kontext_brain.required=true");
    expect(codexArguments(input)).toContain(
      'mcp_servers.kontext_brain.default_tools_approval_mode="approve"',
    );
    const prompt = codexPrompt(input);
    expect(prompt).toContain("kontext_prepare_task");
    expect(prompt).toContain(scenario.taskId);
    for (const rule of scenario.rules) {
      expect(prompt).not.toContain(rule.evidenceText);
      if (rule.statement) expect(prompt).not.toContain(rule.statement);
      if (rule.definition) expect(prompt).not.toContain(rule.definition);
    }
  });

  it("removes usage-billed provider keys while retaining subscription auth state", () => {
    const environment = codexSubscriptionEnvironment({
      OPENAI_API_KEY: "remove",
      ANTHROPIC_API_KEY: "remove",
      CODEX_HOME: "/subscription-state",
      PATH: "/bin",
    });
    expect(environment.OPENAI_API_KEY).toBeUndefined();
    expect(environment.ANTHROPIC_API_KEY).toBeUndefined();
    expect(environment.CODEX_HOME).toBe("/subscription-state");
    expect(environment.PATH).toBe("/bin");
  });

  it("extracts usage and observed Kontext tool calls from Codex JSONL", async () => {
    const runner = new CodexCodeQualityRunner(async () => ({
      exitCode: 0,
      stdout: [
        JSON.stringify({
          item: {
            id: "call:prepare",
            type: "mcp_tool_call",
            tool: "mcp__kontext_brain__kontext_prepare_task",
          },
        }),
        JSON.stringify({
          item: { id: "call:begin", type: "mcp_tool_call", tool: "kontext_begin_logic" },
        }),
        JSON.stringify({ usage: { input_tokens: 120, output_tokens: 30 } }),
      ].join("\n"),
      stderr: "",
      durationMilliseconds: 25,
    }));
    const result = await runner.execute({
      arm: "kontext",
      scenario,
      workspacePath: "/tmp/workspace",
      repositoryRoot: "/repo",
      pluginDataDirectory: "/tmp/plugin-data",
      config,
    });
    expect(result.inputTokens).toBe(120);
    expect(result.outputTokens).toBe(30);
    expect(result.kontextToolsObserved).toEqual(["kontext_begin_logic", "kontext_prepare_task"]);
  });
});
