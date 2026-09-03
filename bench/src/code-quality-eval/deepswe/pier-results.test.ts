import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { DeepSwePreparedArm } from "./contracts.js";
import { classifyOutcome, readPierArmResults } from "./pier-results.js";

const cleanup = new Set<string>();

afterEach(async () => {
  await Promise.all([...cleanup].map((entry) => rm(entry, { recursive: true, force: true })));
  cleanup.clear();
});

describe("Pier DeepSWE result ingestion", () => {
  it("normalizes package task names, metrics, telemetry, and patch identity", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "kontext-deepswe-results-"));
    cleanup.add(root);
    const job = path.join(root, "job");
    const trialName = "datacurve__demo__1";
    const archivedAgent = path.join(job, trialName, "artifacts", "kontext-agent");
    await mkdir(archivedAgent, { recursive: true });
    await writeFile(
      path.join(archivedAgent, "kontext-calls.jsonl"),
      `${JSON.stringify({ command: "prepare-task", arguments: {} })}\n${JSON.stringify({ command: "begin-logic", arguments: { path: "src/a.py", symbol: "A.run" } })}\n${JSON.stringify({ command: "check-change", arguments: { path: "src/a.py", symbol: "A.run", tier: "fast" } })}\n${JSON.stringify({ command: "check-change", arguments: { path: "src/a.py", symbol: "A.run", tier: "targeted" } })}\n`,
      "utf8",
    );
    await writeFile(path.join(job, trialName, "artifacts", "model.patch"), "diff --git a/a b/a\n");
    await writeFile(
      path.join(archivedAgent, "mini-swe-agent.trajectory.json"),
      JSON.stringify({
        messages: [
          { role: "user", content: "task" },
          {
            role: "assistant",
            extra: {
              response: {
                usage: {
                  input_tokens: 100,
                  output_tokens: 30,
                  input_tokens_details: { cached_tokens: 20 },
                  cost_details: { upstream_inference_cost: 0.25 },
                },
              },
            },
          },
        ],
      }),
      "utf8",
    );
    await writeFile(path.join(job, "result.json"), JSON.stringify({ n_total_trials: 1 }), "utf8");
    await writeFile(
      path.join(job, trialName, "result.json"),
      JSON.stringify({
        task_name: "datacurve/demo",
        trial_name: trialName,
        verifier_result: { rewards: { reward: 1 } },
        agent_execution: {
          started_at: "2026-01-01T00:00:00.000Z",
          finished_at: "2026-01-01T00:00:02.500Z",
        },
      }),
      "utf8",
    );
    const results = await readPierArmResults(arm(path.join(job, "result.json")));

    expect(results).toHaveLength(1);
    expect(results[0]).toMatchObject({
      arm: "kontext",
      taskId: "demo",
      eligible: true,
      success: true,
      reward: 1,
      durationMilliseconds: 2_500,
      inputTokens: 100,
      cachedTokens: 20,
      outputTokens: 30,
      costUsd: 0.25,
      agentSteps: 1,
      peakContextTokens: 100,
      context: {
        prepareCalls: 1,
        beginLogicCalls: 1,
        fastCheckCalls: 1,
        targetedCheckCalls: 1,
        logicSymbols: ["src/a.py#A.run"],
        fullyCheckedLogicSymbols: ["src/a.py#A.run"],
        protocolComplete: true,
      },
    });
    expect(results[0]?.patchSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(results[0]?.trajectoryPath).toBe(
      path.join(archivedAgent, "mini-swe-agent.trajectory.json"),
    );
    expect(results[0]?.trajectorySha256).toMatch(/^[0-9a-f]{64}$/);
  });

  it("counts agent capability failures but excludes infrastructure failures", () => {
    expect(classifyOutcome(undefined, "AgentTimeoutError", "timed out")).toEqual({
      eligible: true,
      success: false,
      capabilityFailureReason: "agent_timeout",
    });
    expect(classifyOutcome(undefined, "APIError", "provider overloaded")).toEqual({
      eligible: false,
      success: false,
      exclusionReason: "provider_error",
    });
    expect(classifyOutcome(-1, "VerifierError", "reward missing")).toEqual({
      eligible: false,
      success: false,
      exclusionReason: "verifier_error",
    });
  });

  it("refuses to turn missing child results into a silent zero-trial report", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "kontext-deepswe-missing-results-"));
    cleanup.add(root);
    const resultPath = path.join(root, "result.json");
    await writeFile(resultPath, JSON.stringify({ n_total_trials: 1 }), "utf8");

    await expect(readPierArmResults(arm(resultPath))).rejects.toThrow(
      "declares 1 trials but 0 trial results were found",
    );
  });
});

function arm(expectedJobResultPath: string): DeepSwePreparedArm {
  return {
    arm: "kontext",
    jobName: "job",
    jobConfigPath: "/tmp/pier.json",
    contextIndexPath: "/tmp/context.json",
    expectedJobResultPath,
    command: ["pier", "run"],
  };
}
