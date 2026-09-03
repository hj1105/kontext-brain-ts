import { describe, expect, it } from "vitest";
import type { RealOssRunConfig, RealOssRunResult } from "./contracts.js";
import { flaskBlueprintNameTask } from "./manifest.js";
import { buildRealOssReport, renderRealOssMarkdown } from "./report.js";

const config: RealOssRunConfig = {
  runtime: "codex",
  model: "test-model",
  reasoningEffort: "medium",
  repetitions: 1,
  timeoutMilliseconds: 1_000,
  arms: ["baseline", "kontext"],
  cacheDirectory: "/tmp/cache",
};

describe("real OSS report", () => {
  it("reports upstream test gates and ontology ingestion separately", () => {
    const report = buildRealOssReport({
      task: flaskBlueprintNameTask,
      config,
      generatedAt: "2026-09-02T00:00:00.000Z",
      runs: [run("baseline", false), run("kontext", true)],
    });
    expect(report.evidenceStrength).toBe("smoke");
    expect(report.summaries.find((entry) => entry.arm === "kontext")?.taskSuccessRate).toBe(1);
    expect(renderRealOssMarkdown(report)).toContain("Ontology ingestion");
    expect(renderRealOssMarkdown(report)).toContain("59/59");
  });
});

function run(arm: "baseline" | "kontext", withOntology: boolean): RealOssRunResult {
  return {
    runId: `real-oss:r1:${arm}`,
    instanceId: flaskBlueprintNameTask.instanceId,
    repetition: 1,
    arm,
    model: "test-model",
    reasoningEffort: "medium",
    startedAt: "2026-09-02T00:00:00.000Z",
    finishedAt: "2026-09-02T00:00:01.000Z",
    durationMilliseconds: 1_000,
    runtimeExitCode: 0,
    kontextToolCalls: [],
    expectedLogicConsultations: withOntology ? 1 : 0,
    observedLogicConsultations: withOntology ? 1 : 0,
    contextConsulted: withOntology,
    evaluationEligible: true,
    taskSuccess: true,
    ...(withOntology
      ? {
          ontology: {
            codeResources: 20,
            codeSymbols: 300,
            behaviorBearingSymbols: 200,
            provenanceResources: 4,
            normativeRecords: 3,
            targetSymbolId: "code-symbol:1",
            targetQualifiedName: "Blueprint.__init__",
            governingRecordIds: ["decision:one"],
          },
        }
      : {}),
    grade: {
      publicTestsPassed: true,
      targetChanged: true,
      allowedPathsOnly: true,
      changedFiles: ["src/flask/blueprints.py"],
      failToPassPassed: 1,
      failToPassTotal: 1,
      passToPassPassed: 59,
      passToPassTotal: 59,
      hiddenPatchApplied: true,
      hiddenFailures: [],
      patch: "diff",
    },
  };
}
