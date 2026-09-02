import { describe, expect, it } from "vitest";
import type { KontextToolCall } from "../codex-runner.js";
import { governedPolicy } from "./generator.js";
import { retrieveLargeScaleContext } from "./retrieval.js";
import { assessKontextConsultation, largeScalePrompt } from "./runner.js";
import type { LargeScaleLogicTarget } from "./state.js";

const targets: readonly LargeScaleLogicTarget[] = Array.from({ length: 8 }, (_, index) => ({
  workItemId: `work-item:target-${String(index + 1).padStart(2, "0")}`,
  plannedSymbolId: `planned-symbol:target-${String(index + 1).padStart(2, "0")}`,
}));

describe("large-scale benchmark runner", () => {
  it("does not leak the held-out policy through baseline or Kontext prompts", () => {
    for (const arm of ["baseline", "kontext"] as const) {
      const prompt = largeScalePrompt({
        arm,
        workspacePath: "/tmp/fixture",
        runtime: "codex",
        targets: arm === "kontext" ? targets : [],
        createdAt: "2026-09-02T00:00:00.000Z",
      });
      expect(prompt).not.toContain(governedPolicy.constantName);
      expect(prompt).not.toContain(governedPolicy.sharedModule);
      expect(prompt).not.toContain(String(governedPolicy.capMs));
      expect(prompt).not.toContain(`factor=${governedPolicy.factor}`);
      expect(prompt).not.toContain("billing subsystem");
    }
  });

  it("gives the RAG control realistic source documents rather than normative records", async () => {
    const retrieval = await retrieveLargeScaleContext(
      {
        model: "fixture",
        dimensions: 2,
        async embed(inputs) {
          return inputs.map((input) => ({
            id: input.id,
            values:
              input.id === "spec:billing-retry-recovery" || input.id === "billing"
                ? [1, 0]
                : [0, 1],
          }));
        },
        getUsage() {
          return { requests: 0, inputTokens: 0, totalTokens: 0 };
        },
      },
      1,
    );
    const prompt = largeScalePrompt({
      arm: "rag",
      workspacePath: "/tmp/fixture",
      runtime: "codex",
      targets: [],
      retrieval,
    });
    expect(prompt).toContain("source documents");
    expect(prompt).toContain("spec:billing-retry-recovery");
    expect(prompt).toContain(governedPolicy.constantName);
  });

  it("requires one real prepare and one distinct begin call per logic item", () => {
    const calls: KontextToolCall[] = [
      { callId: "prepare", name: "kontext_prepare_task" },
      ...targets.map((_, index) => ({
        callId: `begin-${index}`,
        name: "kontext_begin_logic" as const,
      })),
    ];
    expect(assessKontextConsultation(calls, targets.length)).toEqual({
      prepareCalls: 1,
      beginCalls: 8,
      complete: true,
    });
    expect(assessKontextConsultation(calls.slice(0, -1), targets.length).complete).toBe(false);
  });
});
