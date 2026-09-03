import { describe, expect, it } from "vitest";
import type { DeepSweArm, DeepSweTrialResult } from "./contracts.js";
import { compareDeepSweArms, pairedBootstrap95, summarizeDeepSweArm } from "./metrics.js";

describe("DeepSWE metrics", () => {
  it("uses task-macro pass@1 and any-of-four pass@4", () => {
    const trials = [
      trial("kontext", "one", 1, true),
      trial("kontext", "one", 2, false),
      trial("kontext", "one", 3, false),
      trial("kontext", "one", 4, false),
      trial("kontext", "two", 1, false),
      trial("kontext", "two", 2, false),
      trial("kontext", "two", 3, false),
      trial("kontext", "two", 4, false),
    ];
    const summary = summarizeDeepSweArm("kontext", trials);
    expect(summary.passAt1).toBe(0.125);
    expect(summary.passAt4).toBe(0.5);
    expect(summary.tasks).toBe(2);
  });

  it("computes paired task deltas and deterministic bootstrap intervals", () => {
    const trials = [
      trial("kontext", "one", 1, true),
      trial("kontext", "two", 1, true),
      trial("baseline", "one", 1, false),
      trial("baseline", "two", 1, true),
    ];
    const comparison = compareDeepSweArms("kontext", "baseline", trials, 1_000, 42);
    expect(comparison.passAt1Delta).toBe(0.5);
    expect(comparison.passAt4Delta).toBe(0.5);
    expect(comparison.comparableTasks).toBe(2);
    expect(comparison.passAt1ClusterBootstrap95).toEqual([0, 1]);
    expect(comparison.passAt4ClusterBootstrap95).toEqual([0, 1]);
    expect(pairedBootstrap95([0, 1], 1_000, 42)).toEqual(pairedBootstrap95([0, 1], 1_000, 42));
  });

  it("removes excluded rollouts from the denominator", () => {
    const excluded = { ...trial("rag", "one", 2, false), eligible: false };
    const summary = summarizeDeepSweArm("rag", [trial("rag", "one", 1, true), excluded]);
    expect(summary.passAt1).toBe(1);
    expect(summary.eligibleTrials).toBe(1);
    expect(summary.excludedTrials).toBe(1);
  });
});

function trial(
  arm: DeepSweArm,
  taskId: string,
  rolloutIndex: number,
  success: boolean,
): DeepSweTrialResult {
  return {
    arm,
    taskId,
    trialName: `${taskId}-${rolloutIndex}`,
    rolloutIndex,
    eligible: true,
    success,
    outputTokens: rolloutIndex * 10,
    durationMilliseconds: rolloutIndex * 100,
    context: {
      prepareCalls: 1,
      searchCalls: 0,
      beginLogicCalls: 1,
      fastCheckCalls: 1,
      targetedCheckCalls: 1,
      logicSymbols: ["src/a.ts#run"],
      fullyCheckedLogicSymbols: ["src/a.ts#run"],
      protocolComplete: true,
    },
  };
}
