import { writeFile } from "node:fs/promises";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { runCodeQualityEvaluation } from "./harness.js";
import { codeQualityScenarios } from "./scenarios.js";

const correctSources: Readonly<Record<string, string>> = {
  "retry-policy": `const RECOVERY_WINDOW_MS = 4500;
export function computeRetryDelay(failureIndex, baseMs) {
  if (!Number.isInteger(failureIndex) || failureIndex < 0 || !Number.isInteger(baseMs) || baseMs < 0) throw new RangeError("invalid input");
  return Math.min(baseMs * 3 ** failureIndex, RECOVERY_WINDOW_MS);
}
`,
  "order-cancellation": `export function cancellationOutcome(order) {
  const hasRevocationEligibility = order.state === "confirmed" && !order.shipmentId && order.fraudHold !== true;
  return hasRevocationEligibility ? "revocable" : "locked";
}
`,
  "service-credit-allocation": `export function allocateServiceCredit(totalCents, accountIds) {
  if (!Number.isInteger(totalCents) || totalCents < 0) throw new RangeError("invalid total");
  if (accountIds.length === 0 || accountIds.some((id) => !id) || new Set(accountIds).size !== accountIds.length) throw new Error("invalid accounts");
  const sorted = [...accountIds].sort();
  const base = Math.floor(totalCents / sorted.length);
  const remainder = totalCents % sorted.length;
  const serviceCreditByAccount = {};
  for (const [index, id] of sorted.entries()) serviceCreditByAccount[id] = base + (index >= sorted.length - remainder ? 1 : 0);
  return serviceCreditByAccount;
}
`,
};

const baselineSources: Readonly<Record<string, string>> = {
  "retry-policy": "export function computeRetryDelay(_failureIndex, baseMs) { return baseMs; }\n",
  "order-cancellation": 'export function cancellationOutcome() { return "revocable"; }\n',
  "service-credit-allocation": `export function allocateServiceCredit(totalCents, accountIds) {
  const each = totalCents / accountIds.length;
  return Object.fromEntries([...accountIds].sort().map((id) => [id, each]));
}\n`,
};

// The harness test exercises pairing and scoring mechanics, not the scenario
// catalogue, so it runs the fixed subset it carries sources for.
const fixtureScenarios = codeQualityScenarios.filter(
  (scenario) => scenario.scenarioId in correctSources,
);

describe("code-quality harness", () => {
  it("runs paired isolated workspaces and scores held-out requirements", async () => {
    let publishedStates = 0;
    const report = await runCodeQualityEvaluation({
      repositoryRoot: "/repo",
      scenarios: fixtureScenarios,
      config: {
        runtime: "codex",
        model: "fixture-model",
        reasoningEffort: "medium",
        repetitions: 1,
        timeoutMilliseconds: 1_000,
      },
      dependencies: {
        publishState: async () => {
          publishedStates += 1;
        },
        execute: async (input) => {
          const sources = input.arm === "kontext" ? correctSources : baselineSources;
          const source = sources[input.scenario.scenarioId];
          if (!source) throw new Error(`Missing source for ${input.scenario.scenarioId}`);
          await writeFile(path.join(input.workspacePath, input.scenario.sourceFile), source);
          return {
            exitCode: 0,
            stdout: "",
            stderr: "",
            durationMilliseconds: 10,
            kontextToolsObserved:
              input.arm === "kontext" ? ["kontext_prepare_task", "kontext_begin_logic"] : [],
          };
        },
      },
    });

    expect(publishedStates).toBe(fixtureScenarios.length);
    expect(report.runs).toHaveLength(fixtureScenarios.length * 2);
    expect(report.hiddenAssertionUplift).toBeGreaterThan(0);
    expect(report.paired.kontextWins).toBe(fixtureScenarios.length);
    expect(report.summaries.find((summary) => summary.arm === "kontext")).toMatchObject({
      contextConsultationRate: 1,
      taskSuccessRate: 1,
    });
  });
});
