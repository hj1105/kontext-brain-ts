import { describe, expect, it } from "vitest";
import {
  AnthropicJsonClient,
  type AnthropicTransport,
  type AnthropicTransportRequest,
  stripUnsupportedSchemaKeywords,
} from "./anthropic-json.js";
import { CodexJsonClient } from "./codex-json.js";
import type { BenchmarkQuery, RetrievedEvidence } from "./contracts.js";
import { createJsonLlmClient } from "./llm-json-client.js";

const QUERY: BenchmarkQuery = {
  id: "q1",
  text: "Question?",
  referenceAnswer: "gold",
  goldEvidenceIds: ["e1"],
  goldEvidenceText: ["evidence"],
  answerable: true,
  category: "Fact",
  metadata: {},
};

const EVIDENCE: readonly RetrievedEvidence[] = [
  { id: "e1", sourceId: "s1", text: "evidence", score: 1, rank: 1, metadata: {} },
];

const ANSWER = {
  answer: "The answer.",
  citations: ["e1"],
  abstained: false,
  abstention_reason: null,
};

function judgePayload(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    answer_correctness: 1,
    completeness: 0.5,
    strict_faithfulness: 1,
    citation_precision: 1,
    citation_recall: 1,
    acceptable_abstention: false,
    clarity: 1,
    conciseness: 1,
    fluency: 1,
    claims: [
      { claim: "The answer.", supported: true, correct: true, citations: ["e1"], reason: "ok" },
    ],
    ...overrides,
  };
}

function transportReturning(
  payload: unknown,
  onRequest?: (request: AnthropicTransportRequest) => void,
): AnthropicTransport {
  return async (request) => {
    onRequest?.(request);
    return {
      text: JSON.stringify(payload),
      inputTokens: 12,
      outputTokens: 5,
      latencyMs: 10,
      stopReason: "end_turn",
    };
  };
}

describe("AnthropicJsonClient", () => {
  it("parses the common answer contract and reports latency and token usage", async () => {
    let receivedRequest: AnthropicTransportRequest | undefined;
    const client = new AnthropicJsonClient(
      transportReturning(ANSWER, (request) => {
        receivedRequest = request;
      }),
    );

    const result = await client.answer(
      { model: "claude-sonnet-5", reasoningEffort: "medium" },
      QUERY,
      EVIDENCE,
    );

    expect(receivedRequest?.model).toBe("claude-sonnet-5");
    expect(receivedRequest?.effort).toBe("medium");
    expect(receivedRequest?.prompt).toContain("[e1] evidence");
    expect(receivedRequest?.prompt).not.toContain("gold");
    expect(result.value).toEqual({
      answer: "The answer.",
      citations: ["e1"],
      abstained: false,
      abstentionReason: null,
    });
    expect(result.latencyMs).toBe(10);
    expect(result.inputTokens).toBe(12);
    expect(result.outputTokens).toBe(5);
  });

  it("strips range keywords from the judge schema sent over the wire", async () => {
    let receivedRequest: AnthropicTransportRequest | undefined;
    const client = new AnthropicJsonClient(
      transportReturning(judgePayload(), (request) => {
        receivedRequest = request;
      }),
    );

    const result = await client.judge(
      { model: "claude-sonnet-5", reasoningEffort: "xhigh" },
      QUERY,
      EVIDENCE,
      { answer: "The answer.", citations: ["e1"], abstained: false, abstentionReason: null },
    );

    const serializedSchema = JSON.stringify(receivedRequest?.schema);
    expect(serializedSchema).not.toContain('"minimum"');
    expect(serializedSchema).not.toContain('"maximum"');
    expect(serializedSchema).toContain('"answer_correctness"');
    expect(result.value.completeness).toBe(0.5);
  });

  it("still rejects out-of-range judge scores client-side", async () => {
    const client = new AnthropicJsonClient(
      transportReturning(judgePayload({ answer_correctness: 1.5 })),
    );

    await expect(
      client.judge({ model: "claude-sonnet-5", reasoningEffort: "xhigh" }, QUERY, EVIDENCE, {
        answer: "The answer.",
        citations: ["e1"],
        abstained: false,
        abstentionReason: null,
      }),
    ).rejects.toThrow("answer_correctness must be a number in [0, 1]");
  });

  it("throws when the response stops with a refusal", async () => {
    const client = new AnthropicJsonClient(async () => ({
      text: "",
      inputTokens: null,
      outputTokens: null,
      latencyMs: 1,
      stopReason: "refusal",
    }));

    await expect(
      client.answer({ model: "claude-sonnet-5", reasoningEffort: "medium" }, QUERY, EVIDENCE),
    ).rejects.toThrow("refusal");
  });

  it("removes only the unsupported schema keywords", () => {
    const stripped = stripUnsupportedSchemaKeywords({
      type: "object",
      additionalProperties: false,
      required: ["score", "label"],
      properties: {
        score: { type: "number", minimum: 0, maximum: 1, multipleOf: 0.05 },
        label: { type: "string", minLength: 1, maxLength: 10 },
        items: { type: "array", minItems: 1, maxItems: 3, items: { type: "string" } },
      },
    });

    expect(stripped).toEqual({
      type: "object",
      additionalProperties: false,
      required: ["score", "label"],
      properties: {
        score: { type: "number" },
        label: { type: "string" },
        items: { type: "array", items: { type: "string" } },
      },
    });
  });
});

describe("createJsonLlmClient", () => {
  it("returns the codex exec client for codex-exec execution", () => {
    expect(createJsonLlmClient("codex-exec")).toBeInstanceOf(CodexJsonClient);
  });

  it("returns the Anthropic API client for anthropic-api execution", () => {
    expect(createJsonLlmClient("anthropic-api")).toBeInstanceOf(AnthropicJsonClient);
  });
});
