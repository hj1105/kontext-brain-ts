import { createContextReceipt } from "@kontext-brain/context";
import {
  type ChangeBundle,
  type LogicWorkItem,
  type TaskContextSnapshot,
  type TaskContract,
  type VerificationRun,
  computeTaskContextDigest,
  createChangeBundle,
  isAccuracyManifestValid,
} from "@kontext-brain/spec";
import { describe, expect, it } from "vitest";
import {
  assembleAccuracyManifest,
  auditAccuracyManifestCandidate,
  createFastVerificationPlan,
  createFullVerificationPlan,
  createQuarantineRecord,
  createTargetedVerificationPlan,
  validateChangeBundle,
} from "../src/index.js";

const contract: TaskContract = {
  taskId: "task:bundle-validation",
  intent: "Accept only a worker handoff bound to observed code and current proof.",
  acceptance: [
    {
      criterionId: "acceptance:worker-test",
      statement: "Worker tests pass.",
      verifier: { kind: "test", ref: "worker:test" },
    },
  ],
  nonGoals: [],
  targets: ["symbol:handler"],
  risk: "low",
};
const snapshotInput = {
  taskId: contract.taskId,
  baseCodeRevision: "commit:base",
  effectiveScopes: [{ kind: "codebase" as const, codebaseId: "codebase:example" }],
  normativeRevisions: [
    {
      kind: "invariant" as const,
      recordId: "invariant:proof",
      revisionId: "invariant:proof@1",
    },
  ],
  requiredEvidenceIds: ["evidence:decision"],
  sourceFreshnessDigest: "freshness:current",
};
const snapshot: TaskContextSnapshot = {
  ...snapshotInput,
  contextDigest: computeTaskContextDigest(snapshotInput),
  createdAt: "2026-08-28T09:00:00.000Z",
};
const workItem: LogicWorkItem = {
  workItemId: "work-item:handler",
  taskId: contract.taskId,
  plannedSymbolIds: ["symbol:handler"],
  dependsOn: [],
  allowedPaths: ["src/handler.ts"],
  requiredVerifiers: [{ kind: "test", ref: "worker:test" }],
  capabilityId: "capability:handler",
};
const receipt = createContextReceipt({
  snapshot,
  logic: workItem,
  allowedPaths: workItem.allowedPaths,
  evidenceIds: snapshot.requiredEvidenceIds,
  issuedAt: "2026-08-28T09:00:00.000Z",
  expiresAt: "2026-08-28T10:00:00.000Z",
});
const resultRevision = "commit:result";
const requirements = [
  ...createFastVerificationPlan({ affectedSymbolIds: ["symbol:handler"] }).requirements,
  ...createTargetedVerificationPlan({ workItem }).requirements,
];
const bundleVerificationRuns: readonly VerificationRun[] = requirements.map(
  (requirement, index) => ({
    verificationRunId: `verification:${index}`,
    tier: requirement.tier,
    verifierKind: requirement.verifier.kind,
    verifierRef: requirement.verifier.ref,
    codeRevision: resultRevision,
    contextDigest: snapshot.contextDigest,
    subjectIds: requirement.subjectIds,
    result: "passed",
    outputDigest: `sha256:output-${index}`,
    observedAt: "2026-08-28T09:20:00.000Z",
  }),
);
const completionVerificationRuns: readonly VerificationRun[] = createFullVerificationPlan({
  contract,
}).requirements.map((requirement, index) => ({
  verificationRunId: `verification:full:${index}`,
  tier: "full",
  verifierKind: requirement.verifier.kind,
  verifierRef: requirement.verifier.ref,
  codeRevision: resultRevision,
  contextDigest: snapshot.contextDigest,
  subjectIds: requirement.subjectIds,
  result: "passed",
  outputDigest: `sha256:full-output-${index}`,
  observedAt: "2026-08-28T09:35:00.000Z",
}));
const manifestAuditRun: VerificationRun = {
  verificationRunId: "verification:full:manifest-audit",
  tier: "full",
  verifierKind: "query",
  verifierRef: "kontext:manifest-audit",
  codeRevision: resultRevision,
  contextDigest: snapshot.contextDigest,
  subjectIds: [contract.taskId, ...contract.targets],
  result: "passed",
  outputDigest: "sha256:manifest-audit",
  observedAt: "2026-08-28T09:39:00.000Z",
};
const verificationRuns = [
  ...bundleVerificationRuns,
  ...completionVerificationRuns,
  manifestAuditRun,
];
const observedPatch = {
  patchDigest: "sha256:patch",
  changedPaths: ["src/handler.ts"],
  changedSymbolIds: ["symbol:handler"],
};

function bundle(overrides: Partial<Omit<ChangeBundle, "bundleId">> = {}): ChangeBundle {
  return createChangeBundle({
    taskId: contract.taskId,
    workItemId: workItem.workItemId,
    baseRevision: snapshot.baseCodeRevision,
    resultRevision,
    taskContextDigest: snapshot.contextDigest,
    patchDigest: observedPatch.patchDigest,
    changedSymbolIds: observedPatch.changedSymbolIds,
    changedPaths: observedPatch.changedPaths,
    contextReceiptIds: [receipt.receiptId],
    evidenceIds: snapshot.requiredEvidenceIds,
    normativeRevisions: snapshot.normativeRevisions,
    verificationRunIds: bundleVerificationRuns.map((run) => run.verificationRunId),
    proposals: [],
    unresolved: [],
    submittedAt: "2026-08-28T09:30:00.000Z",
    ...overrides,
  });
}

describe("Change Bundle validation", () => {
  it("accepts an immutable handoff bound to the observed patch, receipt, and all fast/targeted proof", () => {
    expect(
      validateChangeBundle({
        bundle: bundle(),
        workItem,
        snapshot,
        currentCodeRevision: resultRevision,
        observedPatch,
        receipts: [receipt],
        verificationRuns,
      }),
    ).toEqual({ accepted: true, issues: [] });
  });

  it("rejects claimed paths, stale runs, unresolved work, and active quarantine", () => {
    const staleRuns = verificationRuns.map((run) => ({ ...run, codeRevision: "commit:old" }));
    const candidate = bundle({ unresolved: ["resolve integration conflict"] });
    const quarantine = createQuarantineRecord({
      taskId: contract.taskId,
      workItemId: workItem.workItemId,
      codeRevision: resultRevision,
      contextDigest: snapshot.contextDigest,
      paths: ["src/handler.ts"],
      symbolIds: ["symbol:handler"],
      reasons: ["unobserved_write"],
      observedAt: "2026-08-28T09:25:00.000Z",
    });
    const result = validateChangeBundle({
      bundle: candidate,
      workItem,
      snapshot,
      currentCodeRevision: resultRevision,
      observedPatch: { ...observedPatch, changedPaths: ["src/actually-changed.ts"] },
      receipts: [receipt],
      verificationRuns: staleRuns,
      quarantineRecords: [quarantine],
    });

    expect(result.accepted).toBe(false);
    expect(result.issues.map((issue) => issue.code)).toEqual(
      expect.arrayContaining([
        "changed_paths_mismatch",
        "invalid_verification",
        "missing_verification",
        "unresolved_work",
        "active_quarantine",
      ]),
    );
  });
});

describe("Accuracy Manifest assembly", () => {
  it("audits every candidate condition except the audit run's own evidence edge", () => {
    const audit = auditAccuracyManifestCandidate({
      contract,
      snapshot,
      currentCodeRevision: resultRevision,
      changeBundles: [bundle()],
      verificationRuns: [...bundleVerificationRuns, ...completionVerificationRuns],
      reviewFindings: [],
      createdAt: "2026-08-28T09:40:00.000Z",
    });

    expect(audit.passed).toBe(true);
    expect(audit.blockingIssues).toEqual([]);
    expect(audit.selfEvidenceIssues).toEqual([
      expect.objectContaining({
        code: "verification_mismatch",
        ref: "full:query:kontext:manifest-audit",
      }),
    ]);
  });

  it("fails the candidate audit when any non-self completion condition is missing", () => {
    const audit = auditAccuracyManifestCandidate({
      contract,
      snapshot,
      currentCodeRevision: resultRevision,
      changeBundles: [bundle()],
      verificationRuns: bundleVerificationRuns,
      reviewFindings: [],
      createdAt: "2026-08-28T09:40:00.000Z",
    });

    expect(audit.passed).toBe(false);
    expect(audit.blockingIssues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "verification_mismatch",
          ref: "full:typecheck:workspace:typecheck",
        }),
      ]),
    );
  });

  it("assembles the final immutable audit object only from current accepted bundle proof", () => {
    const candidate = bundle();
    const manifest = assembleAccuracyManifest({
      contract,
      snapshot,
      currentCodeRevision: resultRevision,
      changeBundles: [candidate],
      verificationRuns,
      reviewFindings: [],
      createdAt: "2026-08-28T09:40:00.000Z",
    });

    expect(isAccuracyManifestValid(manifest)).toBe(true);
    expect(manifest.changeBundleIds).toEqual([candidate.bundleId]);
    expect(manifest.verificationRunIds).toEqual(
      verificationRuns.map((run) => run.verificationRunId).sort(),
    );
  });

  it("refuses to assemble completion around an unresolved Change Bundle", () => {
    expect(() =>
      assembleAccuracyManifest({
        contract,
        snapshot,
        currentCodeRevision: resultRevision,
        changeBundles: [bundle({ unresolved: ["still pending"] })],
        verificationRuns,
        reviewFindings: [],
        createdAt: "2026-08-28T09:40:00.000Z",
      }),
    ).toThrow("unresolved_change_bundle");
  });
});
