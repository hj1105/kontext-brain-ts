import { createContextReceipt } from "@kontext-brain/context";
import type { LogicWorkItem, TaskContextSnapshot } from "@kontext-brain/spec";
import { computeTaskContextDigest } from "@kontext-brain/spec";
import { describe, expect, it } from "vitest";
import { assessObservedChange, isQuarantineRecordValid } from "../src/index.js";

const snapshotInput = {
  taskId: "task:quarantine",
  baseCodeRevision: "commit:base",
  effectiveScopes: [{ kind: "personal" as const, subjectId: "user:local" }],
  normativeRevisions: [],
  requiredEvidenceIds: [],
  sourceFreshnessDigest: "freshness:current",
};
const snapshot: TaskContextSnapshot = {
  ...snapshotInput,
  contextDigest: computeTaskContextDigest(snapshotInput),
  createdAt: "2026-08-28T06:00:00.000Z",
};
const workItem: LogicWorkItem = {
  workItemId: "work-item:quarantine",
  taskId: snapshot.taskId,
  plannedSymbolIds: ["planned-symbol:handler"],
  dependsOn: [],
  allowedPaths: ["src/handler.ts"],
  requiredVerifiers: [],
  capabilityId: "capability:quarantine",
};
const receipt = createContextReceipt({
  snapshot,
  logic: workItem,
  allowedPaths: workItem.allowedPaths,
  evidenceIds: [],
  issuedAt: "2026-08-28T06:00:00.000Z",
  expiresAt: "2026-08-28T07:00:00.000Z",
});

function observed(overrides = {}) {
  return {
    codeRevision: "commit:result",
    contextDigest: snapshot.contextDigest,
    paths: ["src/handler.ts"],
    symbolIds: ["symbol:handler"],
    observedAt: "2026-08-28T06:30:00.000Z",
    preWriteAuthorizationObserved: true,
    ...overrides,
  };
}

describe("observed change quarantine", () => {
  it("accepts an observed write inside the current receipt's exact path and symbol scope", () => {
    expect(
      assessObservedChange({
        observed: observed(),
        workItem,
        snapshot,
        receipt,
        authorizedSymbolIds: ["symbol:handler"],
      }),
    ).toEqual({ quarantined: false });
  });

  it("quarantines unobserved, out-of-path, and out-of-symbol writes", () => {
    const assessment = assessObservedChange({
      observed: observed({
        paths: ["src/other.ts"],
        symbolIds: ["symbol:other"],
        preWriteAuthorizationObserved: false,
      }),
      workItem,
      snapshot,
      receipt,
      authorizedSymbolIds: ["symbol:handler"],
    });

    expect(assessment.record?.reasons).toEqual([
      "path_out_of_scope",
      "symbol_out_of_scope",
      "unobserved_write",
    ]);
    if (!assessment.record) throw new Error("expected Quarantine Record");
    expect(isQuarantineRecordValid(assessment.record)).toBe(true);
  });

  it("quarantines missing, expired, and context-mismatched capabilities distinctly", () => {
    expect(assessObservedChange({ observed: observed() }).record?.reasons).toContain(
      "missing_capability",
    );
    expect(
      assessObservedChange({
        observed: observed({ observedAt: "2026-08-28T07:00:00.000Z" }),
        workItem,
        snapshot,
        receipt,
        authorizedSymbolIds: ["symbol:handler"],
      }).record?.reasons,
    ).toContain("expired_capability");
    expect(
      assessObservedChange({
        observed: observed({ contextDigest: "context:other" }),
        workItem,
        snapshot,
        receipt,
        authorizedSymbolIds: ["symbol:handler"],
      }).record?.reasons,
    ).toContain("context_mismatch");
  });
});
