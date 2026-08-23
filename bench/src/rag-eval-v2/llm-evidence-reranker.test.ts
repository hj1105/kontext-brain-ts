import { describe, expect, it } from "vitest";
import { LlmEvidenceReranker } from "./llm-evidence-reranker.js";

describe("LlmEvidenceReranker", () => {
  it("uses valid model IDs first and fills omitted positions deterministically", async () => {
    const reranker = new LlmEvidenceReranker(
      {
        completeText: async () => ({
          value: JSON.stringify({ ranked_ids: ["c", "unknown", "c"] }),
          latencyMs: 1,
          inputTokens: 1,
          outputTokens: 1,
        }),
      },
      { model: "test", reasoningEffort: "low" },
    );
    const candidates = [
      { id: "a", text: "alpha" },
      { id: "b", text: "beta" },
      { id: "c", text: "gamma" },
    ];

    await expect(reranker.rerank("question", candidates, 3)).resolves.toEqual([
      candidates[2],
      candidates[0],
      candidates[1],
    ]);
  });

  it("rejects malformed structured output", async () => {
    let attempts = 0;
    const reranker = new LlmEvidenceReranker(
      {
        completeText: async () => {
          attempts += 1;
          return {
            value: JSON.stringify({ ids: ["a"] }),
            latencyMs: 1,
            inputTokens: 1,
            outputTokens: 1,
          };
        },
      },
      { model: "test", reasoningEffort: "low" },
    );

    await expect(reranker.rerank("question", [{ id: "a", text: "alpha" }], 1)).rejects.toThrow(
      "ranked_ids",
    );
    expect(attempts).toBe(3);
  });

  it("uses the fixed coverage objective without exposing dataset metadata", async () => {
    let systemPrompt = "";
    let context = "";
    const reranker = new LlmEvidenceReranker(
      {
        completeText: async (_model, system, suppliedContext) => {
          systemPrompt = system;
          context = suppliedContext;
          return {
            value: JSON.stringify({ ranked_ids: ["a", "b"] }),
            latencyMs: 1,
            inputTokens: 1,
            outputTokens: 1,
          };
        },
      },
      { model: "test", reasoningEffort: "low" },
      { coverageAware: true },
    );

    await reranker.rerank(
      "Which event caused the later outcome?",
      [
        { id: "a", text: "event evidence" },
        { id: "b", text: "outcome evidence" },
      ],
      2,
      ["event that precedes the outcome", "later outcome evidence"],
    );

    expect(systemPrompt).toContain("collectively covers the distinct evidence needs");
    expect(systemPrompt).toContain("complementary passages");
    expect(systemPrompt).toContain("explicit coverage plan");
    expect(context).toContain("<query_derived_evidence_need index=1>");
    expect(context).toContain("later outcome evidence");
    expect(systemPrompt).not.toMatch(/dataset|reference answer|gold evidence/i);
  });
});
