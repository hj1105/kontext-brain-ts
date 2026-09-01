import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import {
  ClaudeCodeQualityRunner,
  claudeArguments,
  extractClaudeTools,
  writeClaudeMcpConfig,
} from "./claude-runner.js";
import type { CodeQualityRunConfig } from "./contracts.js";
import { codeQualityScenarios } from "./scenarios.js";

const scenario = codeQualityScenarios[0];
if (!scenario) throw new Error("A code-quality scenario is required");

const config: CodeQualityRunConfig = {
  runtime: "claude",
  model: "claude-opus-5",
  reasoningEffort: "medium",
  repetitions: 1,
  timeoutMilliseconds: 60_000,
};

describe("ClaudeCodeQualityRunner", () => {
  it("grants the baseline arm no Kontext tools", () => {
    const args = claudeArguments({ arm: "baseline", workspacePath: "/tmp/ws", config });
    expect(args).not.toContain("--mcp-config");
    expect(args.join(" ")).not.toContain("kontext");
  });

  it("grants the treatment arm exactly the Kontext tools plus edit primitives", () => {
    const args = claudeArguments({
      arm: "kontext",
      workspacePath: "/tmp/ws",
      config,
      mcpConfigPath: "/tmp/mcp.json",
    });
    const allowed = args[args.indexOf("--allowedTools") + 1]?.split(",") ?? [];
    expect(args).toContain("--mcp-config");
    expect(allowed).toContain("mcp__kontext_brain__kontext_prepare_task");
    expect(allowed).toContain("mcp__kontext_brain__kontext_begin_logic");
    expect(allowed).toContain("mcp__kontext_brain__kontext_check_change");
    expect(allowed).toContain("Edit");
  });

  it("refuses a treatment arm without a private MCP configuration", () => {
    expect(() => claudeArguments({ arm: "kontext", workspacePath: "/tmp/ws", config })).toThrow(
      /private MCP configuration/,
    );
  });

  it("writes an MCP configuration scoped to the private data directory", async () => {
    const configPath = await writeClaudeMcpConfig("/repo", "/tmp/private-state");
    const written = JSON.parse(await readFile(configPath, "utf8")) as {
      mcpServers: Record<string, { args: string[]; env: Record<string, string> }>;
    };
    const server = written.mcpServers.kontext_brain;
    expect(server?.args?.[0]).toContain("plugins/kontext-brain/server.mjs");
    // KONTEXT_PLUGIN_DATA is the name resolvePluginDataDirectory reads, so each
    // scenario keeps its own sidecar state instead of the shared user default.
    expect(server?.env).toEqual({ KONTEXT_PLUGIN_DATA: "/tmp/private-state" });
  });

  it("observes Kontext tool calls from stream-json events", async () => {
    const runner = new ClaudeCodeQualityRunner(async () => ({
      exitCode: 0,
      stdout: [
        JSON.stringify({
          type: "assistant",
          message: {
            content: [{ type: "tool_use", name: "mcp__kontext_brain__kontext_prepare_task" }],
          },
        }),
        JSON.stringify({
          type: "assistant",
          message: { content: [{ type: "tool_use", name: "kontext_begin_logic" }] },
        }),
        JSON.stringify({ type: "result", usage: { input_tokens: 900, output_tokens: 120 } }),
      ].join("\n"),
      stderr: "",
      durationMilliseconds: 40,
    }));
    const result = await runner.execute({
      arm: "kontext",
      scenario,
      workspacePath: "/tmp/ws",
      repositoryRoot: "/repo",
      pluginDataDirectory: "/tmp/private-state",
      config,
    });
    expect(result.kontextToolsObserved).toEqual(["kontext_begin_logic", "kontext_prepare_task"]);
    expect(result.inputTokens).toBe(900);
    expect(result.outputTokens).toBe(120);
  });

  it("ignores the server tool advertisement", () => {
    // Claude Code streams the MCP server's catalogue. Counting those names made
    // a run that called nothing look like it had consulted the sidecar.
    const advertisement = JSON.stringify({
      type: "system",
      subtype: "init",
      mcp_servers: [{ name: "kontext_brain", status: "connected" }],
      tools: [
        "mcp__kontext_brain__kontext_prepare_task",
        "mcp__kontext_brain__kontext_begin_logic",
      ],
    });
    expect(extractClaudeTools(advertisement)).toEqual([]);
  });

  it("ignores a name that is not a Kontext tool", () => {
    const event = JSON.stringify({ type: "tool_use", name: "kontext_brain" });
    expect(extractClaudeTools(event)).toEqual([]);
  });

  it("keeps an unprefixed tool name intact", () => {
    expect(
      extractClaudeTools(JSON.stringify({ type: "tool_use", name: "kontext_check_change" })),
    ).toEqual(["kontext_check_change"]);
  });
});
