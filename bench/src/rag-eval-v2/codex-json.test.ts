import { readFileSync, writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  CodexJsonClient,
  type CommandRunner,
  codexCliEnvironment,
  runCommand,
} from "./codex-json.js";

describe("CodexJsonClient", () => {
  it("removes provider API keys from Codex CLI child processes", () => {
    expect(
      codexCliEnvironment({
        PATH: "/bin",
        OPENAI_API_KEY: "openai-secret",
        CODEX_API_KEY: "codex-secret",
        ANTHROPIC_API_KEY: "anthropic-secret",
        GEMINI_API_KEY: "gemini-secret",
        GOOGLE_API_KEY: "google-secret",
        AZURE_OPENAI_API_KEY: "azure-secret",
        NVIDIA_API_KEY: "nvidia-secret",
        CODEX_ACCESS_TOKEN: "chatgpt-access-token",
      }),
    ).toEqual({
      PATH: "/bin",
      CODEX_ACCESS_TOKEN: "chatgpt-access-token",
    });
  });

  it("force-kills a command that ignores the graceful timeout signal", async () => {
    const startedAt = performance.now();
    await expect(
      runCommand(
        process.execPath,
        ["-e", "process.on('SIGTERM', () => {}); setInterval(() => {}, 1000)"],
        "",
        200,
      ),
    ).rejects.toThrow("timed out after 200ms");
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
    let receivedSchema = "";
    const runner: CommandRunner = async (_command, args, stdin) => {
      receivedPrompt = stdin;
      const schemaIndex = args.indexOf("--output-schema");
      receivedSchema = readFileSync(args[schemaIndex + 1]!, "utf8");
      const outputIndex = args.indexOf("--output-last-message");
      writeFileSync(
        args[outputIndex + 1]!,
        JSON.stringify({
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
        }),
      );
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
      evidence: [
        {
          id: evidenceId,
          sourceId: evidenceId,
          text: `evidence ${id}`,
          score: 1,
          rank: 1,
          metadata: {},
        },
      ],
    });

    const result = await client.answerBatch({ model: "gpt-5.6-terra", reasoningEffort: "medium" }, [
      makeInput("q1", "e1"),
      makeInput("q2", "e2"),
    ]);

    expect(result.value.map((item) => item.queryId)).toEqual(["q1", "q2"]);
    expect(result.value.map((item) => item.value.answer)).toEqual(["first", "second"]);
    expect(receivedPrompt).toContain('<case query_id="q1">');
    expect(receivedPrompt).toContain('<case query_id="q2">');
    expect(receivedPrompt).not.toContain("gold q1");
    expect(receivedSchema).toContain('"answer"');
    expect(receivedSchema).not.toContain('"claims"');
    expect(result.inputTokens).toBe(20);
  });

  it("adds the supported evidence-needs contract only for opted-in answer cases", async () => {
    let receivedPrompt = "";
    let receivedSchema = "";
    const runner: CommandRunner = async (_command, args, stdin) => {
      receivedPrompt = stdin;
      const schemaIndex = args.indexOf("--output-schema");
      const schemaPath = args[schemaIndex + 1];
      if (!schemaPath) throw new Error("Codex command omitted --output-schema path");
      receivedSchema = readFileSync(schemaPath, "utf8");
      const outputIndex = args.indexOf("--output-last-message");
      const outputPath = args[outputIndex + 1];
      if (!outputPath) throw new Error("Codex command omitted --output-last-message path");
      writeFileSync(
        outputPath,
        JSON.stringify({
          results: [
            {
              query_id: "q-v13",
              claims: [{ claim: "First supported fact.", citation: "e-v13" }],
              abstained: false,
              abstention_reason: null,
            },
          ],
        }),
      );
      return { exitCode: 0, stdout: "", stderr: "", durationMs: 1 };
    };
    const client = new CodexJsonClient(runner);

    const result = await client.answerBatch({ model: "gpt-5.6-terra", reasoningEffort: "medium" }, [
      {
        answerPolicy: "supported-evidence-needs",
        query: {
          id: "q-v13",
          text: "Which supported needs can be answered?",
          referenceAnswer: "private reference",
          goldEvidenceIds: ["private-gold"],
          goldEvidenceText: ["private evidence"],
          answerable: true,
          category: "test",
          metadata: {},
        },
        evidence: [
          {
            id: "e-v13",
            sourceId: "source",
            text: "Literal retrieved evidence.",
            score: 1,
            rank: 1,
            metadata: {},
          },
        ],
      },
    ]);

    expect(receivedPrompt).toContain("Internally identify the distinct evidence needs");
    expect(receivedPrompt).toContain("At most one atomic claim for each supported evidence need");
    expect(receivedPrompt).toContain("exactly one best supporting evidence ID");
    expect(receivedPrompt).toContain("no more than eight factual claims");
    expect(receivedPrompt).toContain("answer only the supported parts");
    expect(receivedPrompt).not.toContain("private reference");
    expect(receivedPrompt).not.toContain("private-gold");
    expect(receivedSchema).toContain('"maxItems":8');
    expect(receivedSchema).toContain('"enum":["e-v13"]');
    expect(result.value[0]?.value).toEqual({
      answer: "First supported fact. [e-v13]",
      citations: ["e-v13"],
      abstained: false,
      abstentionReason: null,
    });
  });

  it.each([
    {
      name: "more than eight claims",
      payload: {
        claims: Array.from({ length: 9 }, (_, index) => ({
          claim: `Claim ${index + 1}`,
          citation: `e-${index + 1}`,
        })),
        abstained: false,
        abstention_reason: null,
      },
      evidenceIds: Array.from({ length: 9 }, (_, index) => `e-${index + 1}`),
    },
    {
      name: "a citation outside the supplied evidence",
      payload: {
        claims: [{ claim: "Claim", citation: "not-retrieved" }],
        abstained: false,
        abstention_reason: null,
      },
      evidenceIds: ["e-1"],
    },
    {
      name: "normalized duplicate claims",
      payload: {
        claims: [
          { claim: "Same claim", citation: "e-1" },
          { claim: "  same   claim  ", citation: "e-2" },
        ],
        abstained: false,
        abstention_reason: null,
      },
      evidenceIds: ["e-1", "e-2"],
    },
    {
      name: "an empty claim",
      payload: {
        claims: [{ claim: "   ", citation: "e-1" }],
        abstained: false,
        abstention_reason: null,
      },
      evidenceIds: ["e-1"],
    },
    {
      name: "claims paired with abstention",
      payload: {
        claims: [{ claim: "Claim", citation: "e-1" }],
        abstained: true,
        abstention_reason: "unsupported",
      },
      evidenceIds: ["e-1"],
    },
    {
      name: "zero claims without abstention",
      payload: { claims: [], abstained: false, abstention_reason: null },
      evidenceIds: ["e-1"],
    },
  ])("rejects $name in supported evidence-needs output", async ({ payload, evidenceIds }) => {
    const runner: CommandRunner = async (_command, args) => {
      const outputIndex = args.indexOf("--output-last-message");
      const outputPath = args[outputIndex + 1];
      if (!outputPath) throw new Error("Codex command omitted --output-last-message path");
      writeFileSync(outputPath, JSON.stringify(payload));
      return { exitCode: 0, stdout: "", stderr: "", durationMs: 1 };
    };
    const client = new CodexJsonClient(runner);

    await expect(
      client.answer(
        { model: "gpt-5.6-terra", reasoningEffort: "medium" },
        {
          id: "q-v13-invalid",
          text: "Question?",
          referenceAnswer: null,
          goldEvidenceIds: [],
          goldEvidenceText: [],
          answerable: true,
          category: "test",
          metadata: {},
        },
        evidenceIds.map((id, index) => ({
          id,
          sourceId: id,
          text: `Evidence ${index + 1}`,
          score: 1,
          rank: index + 1,
          metadata: {},
        })),
        "supported-evidence-needs",
      ),
    ).rejects.toThrow();
  });

  it("allows one supplied evidence item to support multiple distinct atomic claims", async () => {
    const runner: CommandRunner = async (_command, args) => {
      const outputIndex = args.indexOf("--output-last-message");
      const outputPath = args[outputIndex + 1];
      if (!outputPath) throw new Error("Codex command omitted --output-last-message path");
      writeFileSync(
        outputPath,
        JSON.stringify({
          claims: [
            { claim: "First supported fact.", citation: "e-1" },
            { claim: "Second supported fact.", citation: "e-1" },
          ],
          abstained: false,
          abstention_reason: null,
        }),
      );
      return { exitCode: 0, stdout: "", stderr: "", durationMs: 1 };
    };
    const client = new CodexJsonClient(runner);

    const result = await client.answer(
      { model: "gpt-5.6-terra", reasoningEffort: "medium" },
      {
        id: "q-v13-shared-evidence",
        text: "Question?",
        referenceAnswer: null,
        goldEvidenceIds: [],
        goldEvidenceText: [],
        answerable: true,
        category: "test",
        metadata: {},
      },
      [{ id: "e-1", sourceId: "e-1", text: "Evidence", score: 1, rank: 1, metadata: {} }],
      "supported-evidence-needs",
    );

    expect(result.value).toEqual({
      answer: "First supported fact. [e-1]\nSecond supported fact. [e-1]",
      citations: ["e-1"],
      abstained: false,
      abstentionReason: null,
    });
  });

  it("converts a zero-claim supported-needs result into an abstention", async () => {
    const runner: CommandRunner = async (_command, args) => {
      const outputIndex = args.indexOf("--output-last-message");
      const outputPath = args[outputIndex + 1];
      if (!outputPath) throw new Error("Codex command omitted --output-last-message path");
      writeFileSync(
        outputPath,
        JSON.stringify({
          claims: [],
          abstained: true,
          abstention_reason: "No supplied evidence supports a need.",
        }),
      );
      return { exitCode: 0, stdout: "", stderr: "", durationMs: 1 };
    };
    const client = new CodexJsonClient(runner);

    const result = await client.answer(
      { model: "gpt-5.6-terra", reasoningEffort: "medium" },
      {
        id: "q-v13-abstain",
        text: "Question?",
        referenceAnswer: null,
        goldEvidenceIds: [],
        goldEvidenceText: [],
        answerable: true,
        category: "test",
        metadata: {},
      },
      [],
      "supported-evidence-needs",
    );

    expect(result.value).toEqual({
      answer: "",
      citations: [],
      abstained: true,
      abstentionReason: "No supplied evidence supports a need.",
    });
  });
});
