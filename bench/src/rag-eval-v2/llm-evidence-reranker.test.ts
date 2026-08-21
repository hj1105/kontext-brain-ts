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
});
