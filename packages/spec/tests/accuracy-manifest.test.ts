import { describe, expect, it } from "vitest";
import {
  type AccuracyManifestInput,
  type ChangeBundle,
  type TaskContextSnapshot,
  type TaskContract,
  type VerificationRun,
  computeTaskContextDigest,
  createAccuracyManifest,
  createChangeBundle,
  isAccuracyManifestValid,
  taskContractDigest,
  validateAccuracyManifestForTask,
} from "../src/index.js";

const contract: TaskContract = {
  taskId: "task:manifest",
  intent: "Bind completion to exact evidence.",
  acceptance: [
    {
      criterionId: "acceptance:test",
      statement: "The manifest tests pass.",
      verifier: { kind: "test", ref: "spec:accuracy-manifest" },
    },
  ],
  nonGoals: [],
  targets: ["symbol:createAccuracyManifest"],
  risk: "low",
};

const snapshotInput = {
  taskId: contract.taskId,
  baseCodeRevision: "commit:base",
  effectiveScopes: [{ kind: "personal" as const, subjectId: "user:local" }],
  normativeRevisions: [
    { kind: "decision" as const, recordId: "decision:proof", revisionId: "decision:proof@1" },
    { kind: "invariant" as const, recordId: "invariant:test", revisionId: "invariant:test@1" },
  ],
  requiredEvidenceIds: ["evidence:decision"],
  sourceFreshnessDigest: "freshness:current",
};
const snapshot: TaskContextSnapshot = {
  ...snapshotInput,
  contextDigest: computeTaskContextDigest(snapshotInput),
  createdAt: "2026-08-28T02:00:00.000Z",
};

const run: VerificationRun = {
  verificationRunId: "verification:manifest",
  tier: "full",
  verifierKind: "test",
  verifierRef: "spec:accuracy-manifest",
  codeRevision: "commit:result",
  contextDigest: snapshot.contextDigest,
  subjectIds: [contract.taskId],
  result: "passed",
  outputDigest: "sha256:test-output",
  observedAt: "2026-08-28T02:02:00.000Z",
};

const bundle: ChangeBundle = createChangeBundle({
  taskId: contract.taskId,
  workItemId: "work-item:manifest",
  baseRevision: snapshot.baseCodeRevision,
  resultRevision: run.codeRevision,
  taskContextDigest: snapshot.contextDigest,
  patchDigest: "sha256:patch",
  changedSymbolIds: ["symbol:createAccuracyManifest"],
  changedPaths: ["packages/spec/src/accuracy-manifest.ts"],
  contextReceiptIds: ["context-receipt:manifest"],
  evidenceIds: ["evidence:decision"],
  normativeRevisions: snapshot.normativeRevisions,
  verificationRunIds: [run.verificationRunId],
  proposals: [],
  unresolved: [],
  submittedAt: "2026-08-28T02:03:00.000Z",
});

function manifestInput(overrides: Partial<AccuracyManifestInput> = {}): AccuracyManifestInput {
  return {
    taskId: contract.taskId,
    taskContractDigest: taskContractDigest(contract),
    contextDigest: snapshot.contextDigest,
    baseCodeRevision: snapshot.baseCodeRevision,
    resultCodeRevision: run.codeRevision,
    normativeRevisions: snapshot.normativeRevisions,
    evidenceIds: ["evidence:decision"],
    workItemIds: [bundle.workItemId],
    changeBundleIds: [bundle.bundleId],
    changedSymbolIds: bundle.changedSymbolIds,
    verificationRunIds: [run.verificationRunId],
    reviewFindingIds: [],
    emergencyBypassIds: [],
    createdAt: "2026-08-28T02:04:00.000Z",
    ...overrides,
  };
}

describe("Accuracy Manifest", () => {
  it("canonicalizes set-like fields into one deterministic immutable manifest", () => {
    const first = createAccuracyManifest(
      manifestInput({
        evidenceIds: ["evidence:z", "evidence:decision", "evidence:z"],
        changedSymbolIds: ["symbol:z", "symbol:createAccuracyManifest"],
      }),
    );
    const second = createAccuracyManifest(
      manifestInput({
        evidenceIds: ["evidence:decision", "evidence:z"],
        changedSymbolIds: ["symbol:createAccuracyManifest", "symbol:z"],
      }),
    );

    expect(first).toEqual(second);
    expect(isAccuracyManifestValid(first)).toBe(true);
  });

  it("detects content changed without issuing a new immutable ID", () => {
    const manifest = createAccuracyManifest(manifestInput());

    expect(isAccuracyManifestValid({ ...manifest, resultCodeRevision: "commit:tampered" })).toBe(
      false,
    );
  });

  it("rejects omitted Change Bundles, Evidence, and current passing runs", () => {
    const manifest = createAccuracyManifest(
      manifestInput({ changeBundleIds: [], evidenceIds: [], verificationRunIds: [] }),
    );
    const issues = validateAccuracyManifestForTask({
      manifest,
      contract,
      snapshot,
      currentCodeRevision: run.codeRevision,
      changeBundles: [bundle],
      verificationRuns: [run],
      reviewFindings: [],
    });

    expect(issues.map((issue) => issue.code)).toEqual(
      expect.arrayContaining([
        "change_bundle_mismatch",
        "evidence_mismatch",
        "verification_mismatch",
      ]),
    );
  });

  it("rejects a Change Bundle that still contains unresolved work", () => {
    const manifest = createAccuracyManifest(manifestInput());
    const issues = validateAccuracyManifestForTask({
      manifest,
      contract,
      snapshot,
      currentCodeRevision: run.codeRevision,
      changeBundles: [{ ...bundle, unresolved: ["review the fallback"] }],
      verificationRuns: [run],
      reviewFindings: [],
    });

    expect(issues).toContainEqual(
      expect.objectContaining({ code: "unresolved_change_bundle", ref: bundle.bundleId }),
    );
  });
});
