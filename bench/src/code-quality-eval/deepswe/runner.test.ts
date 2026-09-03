import { describe, expect, it } from "vitest";
import type {
  DeepSweArm,
  DeepSwePreparationManifest,
  DeepSwePreparedArm,
  DeepSweTrialResult,
  DeepSweWorkerImageSnapshot,
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

  it("ensures pinned worker images before running Pier and records the immutable image id", async () => {
    const events: string[] = [];
    const input = manifest(0);
    input.workerImages.push({
      baseImage: "image@sha256:digest",
      codexVersion: "0.144.6",
      pierVersion: "0.3.1",
      recipeVersion: "pier-codex-v1",
      identitySha256: "a".repeat(64),
      tag: `kontext-brain/deepswe-codex:${"a".repeat(24)}`,
      labels: {},
    });
    const plannedImage = input.workerImages[0];
    if (!plannedImage) throw new Error("Missing planned worker image");
    const report = await runPreparedDeepSweEvaluation({
      repositoryRoot: "/repo",
      manifest: input,
      dependencies: {
        ensureWorkerImage: async ({ spec }) => {
          events.push(`image:${spec.baseImage}`);
          return {
            ...plannedImage,
            imageId: "sha256:immutable",
            reused: true,
            labels: {},
          };
        },
        execute: async (_workingDirectory, command) => {
          events.push(command);
          return { exitCode: 0, stdout: "", stderr: "" };
        },
        readResults: async (arm) => [trial(arm.arm)],
      },
    });

    expect(events[0]).toBe("image:image@sha256:digest");
    expect(report?.manifest.workerImages[0]).toEqual(
      expect.objectContaining({ imageId: "sha256:immutable", reused: true }),
    );
  });
});

function manifest(sampleSeed: number): DeepSwePreparationManifest & {
  workerImages: DeepSweWorkerImageSnapshot[];
} {
  return {
    schemaVersion: 1,
    benchmark: "deepswe-kontext-ab",
    preparedAt: "2026-01-01T00:00:00.000Z",
    deepSweRevision: "deep-swe-sha",
    pierRevision: "0.3.1",
    adapterRevision: "adapter-sha",
    runtime: "codex-subscription",
    agentVersion: "0.144.6",
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
    workerImages: [],
    corpusSha256ByTask: { demo: "corpus-sha" },
  };
}

function preparedArm(arm: DeepSweArm): DeepSwePreparedArm {
  return {
    arm,
    runtime: "codex-subscription",
    billingMode: "subscription",
    taskIds: ["demo"],
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
