import { describe, expect, it } from "vitest";
import type { CodeQualityArm, CodeQualityRunResult } from "./contracts.js";
import { buildCodeQualityReport, exactTwoSidedSignTest } from "./report.js";

describe("code-quality report", () => {
  it("computes paired wins and keeps smoke evidence inconclusive", () => {
    const report = buildCodeQualityReport({
      generatedAt: "2026-08-31T00:00:00.000Z",
      config: {
        runtime: "codex",
        model: "test-model",
        reasoningEffort: "medium",
        repetitions: 1,
        timeoutMilliseconds: 1_000,
      },
      scenarios: ["scenario:a"],
      runs: [run("baseline", false), run("kontext", true)],
    });
    expect(report.hiddenAssertionUplift).toBe(1);
    expect(report.taskSuccessUplift).toBe(1);
    expect(report.paired).toMatchObject({ pairs: 1, kontextWins: 1, baselineWins: 0 });
    expect(report.evidenceStrength).toBe("smoke");
    expect(report.verdict).toBe("inconclusive");
  });

  it("computes the exact two-sided sign-test probability", () => {
    expect(exactTwoSidedSignTest(5, 0)).toBeCloseTo(0.0625);
    expect(exactTwoSidedSignTest(3, 2)).toBe(1);
  });
});

function run(arm: CodeQualityArm, passed: boolean): CodeQualityRunResult {
  return {
    runId: `scenario:a:r1:${arm}`,
    scenarioId: "scenario:a",
    repetition: 1,
    arm,
    model: "test-model",
    reasoningEffort: "medium",
    startedAt: "2026-08-31T00:00:00.000Z",
    finishedAt: "2026-08-31T00:00:01.000Z",
    durationMilliseconds: 1_000,
    runtimeExitCode: 0,
    publicTestsPassed: true,
    hiddenAssertions: [{ assertionId: "policy", passed }],
    canonicalTermsPresent: passed ? ["canonicalName"] : [],
    canonicalTermsMissing: passed ? [] : ["canonicalName"],
    changedPaths: ["src/policy.js"],
    outOfScopePaths: [],
    kontextToolsObserved: arm === "kontext" ? ["kontext_prepare_task", "kontext_begin_logic"] : [],
    contextConsulted: arm === "kontext",
    evaluationEligible: true,
    source: "",
    patch: "",
  };
}
