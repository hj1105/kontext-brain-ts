import type { ChangeBundle, LogicWorkItem } from "@kontext-brain/spec";
import { describe, expect, it } from "vitest";
import { planChangeBundleIntegration } from "../src/index.js";

describe("planChangeBundleIntegration", () => {
  it("orders dependencies and serializes bundles that share a file", () => {
    const first = workItem("work:a", ["src/shared.ts"]);
    const second = workItem("work:b", ["src/shared.ts"]);
    const plan = planChangeBundleIntegration({
      taskId: "task:integration",
      workItems: [second, first],
      changeBundles: [bundle(second, "symbol:b"), bundle(first, "symbol:a")],
      authors: [
        { workItemId: first.workItemId, provider: "codex" },
        { workItemId: second.workItemId, provider: "codex" },
      ],
    });

    expect(plan.orderedChangeBundles.map((item) => item.workItemId)).toEqual(["work:a", "work:b"]);
    expect(plan.changedPaths).toEqual(["src/shared.ts"]);
    expect(plan.changedSymbolIds).toEqual(["symbol:a", "symbol:b"]);
    expect(plan.authorProviders).toEqual(["codex"]);
  });

  it("fails closed when two bundles change the same Code Symbol", () => {
    const first = workItem("work:a", ["src/a.ts"]);
    const second = workItem("work:b", ["src/b.ts"]);

    expect(() =>
      planChangeBundleIntegration({
        taskId: "task:integration",
        workItems: [first, second],
        changeBundles: [bundle(first, "symbol:shared"), bundle(second, "symbol:shared")],
        authors: [
          { workItemId: first.workItemId, provider: "codex" },
          { workItemId: second.workItemId, provider: "claude" },
        ],
      }),
    ).toThrow("Semantic integration conflict");
  });
});

function workItem(workItemId: string, allowedPaths: readonly string[]): LogicWorkItem {
  return {
    workItemId,
    taskId: "task:integration",
    plannedSymbolIds: [`planned:${workItemId}`],
    dependsOn: [],
    allowedPaths,
    requiredVerifiers: [],
    capabilityId: `capability:${workItemId}`,
  };
}

function bundle(workItem: LogicWorkItem, symbolId: string): ChangeBundle {
  return {
    bundleId: `bundle:${workItem.workItemId}`,
    taskId: workItem.taskId,
    workItemId: workItem.workItemId,
    baseRevision: "commit:base",
    resultRevision: `workspace:${workItem.workItemId}`,
    taskContextDigest: "context:current",
    patchDigest: `patch:${workItem.workItemId}`,
    changedSymbolIds: [symbolId],
    changedPaths: workItem.allowedPaths,
    contextReceiptIds: [`receipt:${workItem.workItemId}`],
    evidenceIds: [],
    normativeRevisions: [],
    verificationRunIds: [],
    proposals: [],
    unresolved: [],
    submittedAt: "2026-08-31T00:00:00.000Z",
  };
}
