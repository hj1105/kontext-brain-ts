import { describe, expect, it } from "vitest";
import type {
  DeepSweArm,
  DeepSwePreparationManifest,
  DeepSwePreparedArm,
  DeepSweTrialResult,
} from "./contracts.js";
import { runPreparedDeepSweEvaluation } from "./runner.js";

describe("DeepSWE runner", () => {
  it("rotates arm order deterministically and combines all Pier results", async () => {
    const commands: string[] = [];
    const report = await runPreparedDeepSweEvaluation({
      repositoryRoot: "/repo",
      manifest: manifest(1),
      dependencies: {
        execute: async (_workingDirectory, command) => {
          commands.push(command);
          return { exitCode: 0, stdout: "", stderr: "" };
        },
        readResults: async (arm) => [trial(arm.arm)],
      },
    });

    expect(commands).toEqual(["pier-rag", "pier-kontext", "pier-baseline"]);
    expect(report?.trials).toHaveLength(3);
    expect(report?.comparisons.map((entry) => entry.control)).toEqual(["baseline", "rag"]);
  });

  it("does not execute Pier during a dry run", async () => {
    let called = false;
    const report = await runPreparedDeepSweEvaluation({
      repositoryRoot: "/repo",
      manifest: manifest(0),
      dryRun: true,
      dependencies: {
        execute: async () => {
          called = true;
          return { exitCode: 0, stdout: "", stderr: "" };
        },
      },
    });
    expect(report).toBeUndefined();
    expect(called).toBe(false);
  });
});

function manifest(sampleSeed: number): DeepSwePreparationManifest {
  return {
    schemaVersion: 1,
    benchmark: "deepswe-kontext-ab",
    preparedAt: "2026-01-01T00:00:00.000Z",
    deepSweRevision: "deep-swe-sha",
    pierRevision: "0.3.1",
    adapterRevision: "adapter-sha",
    model: "openai/test-model",
    reasoningEffort: "medium",
    attempts: 1,
    sampleSeed,
    tasks: [
      {
        taskId: "demo",
        taskPath: "/tasks/demo",
        instructionSha256: "instruction-sha",
        taskTomlSha256: "toml-sha",
        baseCommit: "base-sha",
        language: "python",
        dockerImage: "image@sha256:digest",
      },
    ],
    arms: (["baseline", "rag", "kontext"] as const).map(preparedArm),
    corpusSha256ByTask: { demo: "corpus-sha" },
  };
}

function preparedArm(arm: DeepSweArm): DeepSwePreparedArm {
  return {
    arm,
    jobName: `job-${arm}`,
    jobConfigPath: `/run/${arm}.json`,
    contextIndexPath: `/run/context-${arm}.json`,
    expectedJobResultPath: `/jobs/${arm}/result.json`,
    command: [`pier-${arm}`, "run"],
  };
}

function trial(arm: DeepSweArm): DeepSweTrialResult {
  return {
    arm,
    taskId: "demo",
    trialName: `${arm}-1`,
    rolloutIndex: 1,
    eligible: true,
    success: arm === "kontext",
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
