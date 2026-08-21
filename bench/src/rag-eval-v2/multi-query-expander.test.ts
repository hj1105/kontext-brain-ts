import { describe, expect, it } from "vitest";
import { MultiQueryExpander } from "./multi-query-expander.js";

describe("MultiQueryExpander", () => {
  it("keeps only three distinct search perspectives and never receives corpus metadata", async () => {
    let systemPrompt = "";
    let context = "unexpected";
    const expander = new MultiQueryExpander(
      {
        completeText: async (_model, system, suppliedContext) => {
          systemPrompt = system;
          context = suppliedContext;
          return {
            value: JSON.stringify({
              queries: [
                "first bridge evidence",
                "FIRST   BRIDGE EVIDENCE",
                "second relation evidence",
                "third date evidence",
                "fourth ignored evidence",
              ],
            }),
            latencyMs: 4,
            inputTokens: 5,
            outputTokens: 6,
          };
        },
      },
      { model: "test", reasoningEffort: "low" },
    );

    await expect(expander.expand("original question")).resolves.toEqual({
      queries: ["first bridge evidence", "second relation evidence", "third date evidence"],
      latencyMs: 4,
      inputTokens: 5,
      outputTokens: 6,
      error: null,
    });
    expect(context).toBe("");
    expect(systemPrompt).toContain("Do not answer the question");
    expect(systemPrompt).not.toMatch(/reference answer|gold evidence|dataset name/i);
  });

  it("fails closed to the original query after malformed responses", async () => {
    let attempts = 0;
    const expander = new MultiQueryExpander(
      {
        completeText: async () => {
          attempts += 1;
          return {
            value: JSON.stringify({ queries: [] }),
            latencyMs: 2,
            inputTokens: null,
            outputTokens: null,
          };
        },
      },
      { model: "test", reasoningEffort: "low" },
    );

    const result = await expander.expand("original question");
    expect(result.queries).toEqual([]);
    expect(result.error).toContain("no distinct usable queries");
    expect(result.latencyMs).toBe(6);
    expect(attempts).toBe(3);
  });
});
