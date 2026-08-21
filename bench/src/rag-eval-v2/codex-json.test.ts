import { writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  CodexJsonClient,
  codexCliEnvironment,
  runCommand,
  type CommandRunner,
} from "./codex-json.js";

describe("CodexJsonClient", () => {
  it("removes provider API keys from Codex CLI child processes", () => {
    expect(codexCliEnvironment({
      PATH: "/bin",
      OPENAI_API_KEY: "openai-secret",
      CODEX_API_KEY: "codex-secret",
      ANTHROPIC_API_KEY: "anthropic-secret",
      GEMINI_API_KEY: "gemini-secret",
      GOOGLE_API_KEY: "google-secret",
      AZURE_OPENAI_API_KEY: "azure-secret",
      NVIDIA_API_KEY: "nvidia-secret",
      CODEX_ACCESS_TOKEN: "chatgpt-access-token",
    })).toEqual({
      PATH: "/bin",
      CODEX_ACCESS_TOKEN: "chatgpt-access-token",
    });
  });

  it("force-kills a command that ignores the graceful timeout signal", async () => {
    const startedAt = performance.now();
    await expect(runCommand(
      process.execPath,
      ["-e", "process.on('SIGTERM', () => {}); setInterval(() => {}, 1000)"],
      "",
      200,
    )).rejects.toThrow("timed out after 200ms");
    expect(performance.now() - startedAt).toBeLessThan(3_000);
  });

  it("pins model options and parses the common answer contract", async () => {
    let receivedArgs: readonly string[] = [];
    let receivedPrompt = "";
    let receivedEnvironment: NodeJS.ProcessEnv | undefined;
    const runner: CommandRunner = async (_command, args, stdin, _timeout, environment) => {
      receivedArgs = args;
      receivedPrompt = stdin;
      receivedEnvironment = environment;
      const outputIndex = args.indexOf("--output-last-message");
      writeFileSync(
        args[outputIndex + 1]!,
        JSON.stringify({
          answer: "The answer.",
          citations: ["e1"],
          abstained: false,
          abstention_reason: null,
        }),
      );
      return {
        exitCode: 0,
        stdout: `${JSON.stringify({ usage: { input_tokens: 12, output_tokens: 5 } })}\n`,
        stderr: "",
        durationMs: 10,
      };
    };
    const client = new CodexJsonClient(runner);
    const result = await client.answer(
      { model: "gpt-5.6-terra", reasoningEffort: "medium" },
      {
        id: "q1",
        text: "Question?",
        referenceAnswer: "gold",
        goldEvidenceIds: ["e1"],
        goldEvidenceText: ["evidence"],
        answerable: true,
        category: "Fact",
        metadata: {},
      },
      [{ id: "e1", sourceId: "s1", text: "evidence", score: 1, rank: 1, metadata: {} }],
    );
    expect(receivedArgs).toContain("gpt-5.6-terra");
    expect(receivedArgs).toContain('model_reasoning_effort="medium"');
    expect(receivedArgs).toContain("read-only");
    expect(receivedArgs).toContain("plugins");
    expect(receivedArgs).toContain("remote_plugin");
    expect(receivedArgs).toContain("browser_use");
    expect(receivedPrompt).toContain("[e1] evidence");
    expect(receivedPrompt).not.toContain("gold");
    expect(receivedEnvironment).toBeDefined();
    expect(receivedEnvironment).not.toHaveProperty("OPENAI_API_KEY");
    expect(receivedEnvironment).not.toHaveProperty("CODEX_API_KEY");
    expect(result.value).toEqual({
      answer: "The answer.",
      citations: ["e1"],
      abstained: false,
      abstentionReason: null,
    });
    expect(result.inputTokens).toBe(12);
    expect(result.outputTokens).toBe(5);
  });

  it("batches independent answer cases and restores requested query order", async () => {
    let receivedPrompt = "";
    const runner: CommandRunner = async (_command, args, stdin) => {
      receivedPrompt = stdin;
      const outputIndex = args.indexOf("--output-last-message");
      writeFileSync(args[outputIndex + 1]!, JSON.stringify({
        results: [
          {
            query_id: "q2",
            answer: "second",
            citations: ["e2"],
            abstained: false,
            abstention_reason: null,
          },
          {
            query_id: "q1",
            answer: "first",
            citations: ["e1"],
            abstained: false,
            abstention_reason: null,
          },
        ],
      }));
      return {
        exitCode: 0,
        stdout: `${JSON.stringify({ usage: { input_tokens: 20, output_tokens: 8 } })}\n`,
        stderr: "",
        durationMs: 15,
      };
    };
    const client = new CodexJsonClient(runner);
    const makeInput = (id: string, evidenceId: string) => ({
      query: {
        id,
        text: `Question ${id}?`,
        referenceAnswer: `gold ${id}`,
        goldEvidenceIds: [evidenceId],
        goldEvidenceText: ["evidence"],
        answerable: true,
        category: "Fact",
        metadata: {},
      },
      evidence: [{ id: evidenceId, sourceId: evidenceId, text: `evidence ${id}`, score: 1, rank: 1, metadata: {} }],
    });

    const result = await client.answerBatch(
      { model: "gpt-5.6-terra", reasoningEffort: "medium" },
      [makeInput("q1", "e1"), makeInput("q2", "e2")],
    );

    expect(result.value.map((item) => item.queryId)).toEqual(["q1", "q2"]);
    expect(result.value.map((item) => item.value.answer)).toEqual(["first", "second"]);
    expect(receivedPrompt).toContain('<case query_id="q1">');
    expect(receivedPrompt).toContain('<case query_id="q2">');
    expect(receivedPrompt).not.toContain("gold q1");
    expect(result.inputTokens).toBe(20);
  });
});
