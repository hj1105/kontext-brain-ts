import { validateContextReceipt } from "@kontext-brain/context";
import {
  type AccuracyManifest,
  type AccuracyManifestIssue,
  type ChangeBundle,
  type NormativeRevisionRef,
  type VerificationRun,
  createAccuracyManifest,
  isChangeBundleValid,
  taskContractDigest,
  validateAccuracyManifestForTask,
} from "@kontext-brain/spec";
import type {
  AccuracyManifestCandidateAudit,
  AssembleAccuracyManifestInput,
  ChangeBundleIssue,
  ChangeBundleIssueCode,
  ChangeBundleValidation,
  ValidateChangeBundleInput,
  VerificationRequirement,
} from "./domain.js";
import { createFastVerificationPlan, createTargetedVerificationPlan } from "./verification-plan.js";

export function validateChangeBundle(input: ValidateChangeBundleInput): ChangeBundleValidation {
  const issues: ChangeBundleIssue[] = [];
  const add = (code: ChangeBundleIssueCode, message: string, ref?: string): void => {
    if (!issues.some((issue) => issue.code === code && issue.ref === ref)) {
      issues.push({ code, message, ref });
    }
  };
  const { bundle, workItem, snapshot } = input;
  if (!isChangeBundleValid(bundle)) {
    add("invalid_bundle_id", "Change Bundle content does not match its immutable ID");
  }
  if (bundle.taskId !== workItem.taskId || bundle.taskId !== snapshot.taskId) {
    add("task_mismatch", "Change Bundle, Work Item, and Snapshot Task IDs must match");
  }
  if (bundle.workItemId !== workItem.workItemId) {
    add("work_item_mismatch", "Change Bundle references another Logic Work Item");
  }
  if (bundle.baseRevision !== snapshot.baseCodeRevision) {
    add("base_revision_mismatch", "Change Bundle base revision does not match the Snapshot");
  }
  if (bundle.resultRevision !== input.currentCodeRevision) {
    add("result_revision_mismatch", "Change Bundle result revision is not current");
  }
  if (bundle.taskContextDigest !== snapshot.contextDigest) {
    add("context_mismatch", "Change Bundle context digest does not match the Snapshot");
  }
  if (bundle.patchDigest !== input.observedPatch.patchDigest) {
    add("patch_mismatch", "Change Bundle patch digest does not match the observed patch");
  }
  if (
    !sameStrings(
      bundle.changedPaths.map(canonicalPath),
      input.observedPatch.changedPaths.map(canonicalPath),
    )
  ) {
    add("changed_paths_mismatch", "Change Bundle paths do not match the observed patch");
  }
  if (!sameStrings(bundle.changedSymbolIds, input.observedPatch.changedSymbolIds)) {
    add(
      "changed_symbols_mismatch",
      "Change Bundle symbols do not match semantic resynchronization",
    );
  }
  if (!sameRevisionRefs(bundle.normativeRevisions, snapshot.normativeRevisions)) {
    add(
      "normative_revision_mismatch",
      "Change Bundle normative revisions do not match the Snapshot",
    );
  }

  const receiptById = new Map(input.receipts.map((receipt) => [receipt.receiptId, receipt]));
  const referencedReceipts = bundle.contextReceiptIds
    .map((receiptId) => receiptById.get(receiptId))
    .filter((receipt) => receipt !== undefined);
  if (
    bundle.contextReceiptIds.length === 0 ||
    referencedReceipts.length !== bundle.contextReceiptIds.length
  ) {
    add("missing_context_receipt", "Change Bundle must reference every used Context Receipt");
  }
  let hasValidReceipt = false;
  for (const receipt of referencedReceipts) {
    const receiptIssues = validateContextReceipt({
      receipt,
      snapshot,
      logic: workItem,
      allowedPaths: workItem.allowedPaths,
      now: bundle.submittedAt,
    });
    if (receiptIssues.length === 0) hasValidReceipt = true;
    else {
      add(
        "invalid_context_receipt",
        `Context Receipt ${receipt.receiptId} is not valid at bundle submission`,
        receipt.receiptId,
      );
    }
  }
  if (referencedReceipts.length > 0 && !hasValidReceipt) {
    add("invalid_context_receipt", "No referenced Context Receipt is current and valid");
  }

  const requiredEvidenceIds = uniqueSorted([
    ...snapshot.requiredEvidenceIds,
    ...referencedReceipts.flatMap((receipt) => receipt.evidenceIds),
  ]);
  if (!requiredEvidenceIds.every((evidenceId) => bundle.evidenceIds.includes(evidenceId))) {
    add("evidence_mismatch", "Change Bundle omits Snapshot or Context Receipt Evidence");
  }

  const runById = new Map(input.verificationRuns.map((run) => [run.verificationRunId, run]));
  const bundleRuns = bundle.verificationRunIds
    .map((runId) => runById.get(runId))
    .filter((run) => run !== undefined);
  if (bundleRuns.length !== bundle.verificationRunIds.length) {
    add("invalid_verification", "Change Bundle references unknown Verification Runs");
  }
  for (const run of bundleRuns) {
    if (
      run.codeRevision !== input.currentCodeRevision ||
      run.contextDigest !== snapshot.contextDigest ||
      run.result !== "passed"
    ) {
      add(
        "invalid_verification",
        `Verification Run ${run.verificationRunId} is not current and passing`,
        run.verificationRunId,
      );
    }
  }
  const currentPassingBundleRuns = bundleRuns.filter(
    (run) =>
      run.codeRevision === input.currentCodeRevision &&
      run.contextDigest === snapshot.contextDigest &&
      run.result === "passed",
  );
  const requirements = uniqueRequirements([
    ...createFastVerificationPlan({ affectedSymbolIds: bundle.changedSymbolIds }).requirements,
    ...createTargetedVerificationPlan({
      workItem,
      boundInvariantVerifiers: input.boundInvariantVerifiers,
    }).requirements,
  ]);
  for (const requirement of requirements) {
    if (!currentPassingBundleRuns.some((run) => runSatisfies(run, requirement))) {
      add(
        "missing_verification",
        `Change Bundle lacks ${requirement.tier} ${requirement.verifier.kind}:${requirement.verifier.ref}`,
        `${requirement.tier}:${requirement.verifier.kind}:${requirement.verifier.ref}`,
      );
    }
  }

  if (bundle.unresolved.length > 0) {
    add("unresolved_work", "Change Bundle still contains unresolved work");
  }
  for (const record of input.quarantineRecords ?? []) {
    if (record.status === "active" && quarantineApplies(record, bundle)) {
      add(
        "active_quarantine",
        `Active Quarantine Record ${record.quarantineId} applies to this bundle`,
        record.quarantineId,
      );
    }
  }
  return { accepted: issues.length === 0, issues };
}

export function assembleAccuracyManifest(input: AssembleAccuracyManifestInput): AccuracyManifest {
  const manifest = createCandidateManifest(input);
  const issues = validateManifest(input, manifest);
  if (issues.length > 0) {
    throw new Error(
      `Cannot assemble Accuracy Manifest: ${issues.map((issue) => issue.code).join(", ")}`,
    );
  }
  return manifest;
}

export function auditAccuracyManifestCandidate(
  input: AssembleAccuracyManifestInput,
): AccuracyManifestCandidateAudit {
  const candidate = createCandidateManifest(input);
  const issues = validateManifest(input, candidate);
  const selfEvidenceIssues = issues.filter(isManifestAuditSelfEvidenceIssue);
  const blockingIssues = issues.filter((issue) => !isManifestAuditSelfEvidenceIssue(issue));
  return {
    passed: blockingIssues.length === 0,
    candidate,
    blockingIssues,
    selfEvidenceIssues,
  };
}

function createCandidateManifest(input: AssembleAccuracyManifestInput): AccuracyManifest {
  const bundles = input.changeBundles.filter((bundle) => bundle.taskId === input.contract.taskId);
  const taskSubjectIds = new Set([
    input.contract.taskId,
    ...input.contract.targets,
    ...bundles.flatMap((bundle) => [bundle.workItemId, ...bundle.changedSymbolIds]),
  ]);
  const bundleRunIds = new Set(bundles.flatMap((bundle) => bundle.verificationRunIds));
  const runs = input.verificationRuns.filter(
    (run) =>
      run.codeRevision === input.currentCodeRevision &&
      run.contextDigest === input.snapshot.contextDigest &&
      run.result === "passed" &&
      (bundleRunIds.has(run.verificationRunId) ||
        run.subjectIds.some((subjectId) => taskSubjectIds.has(subjectId))),
  );
  const manifest = createAccuracyManifest({
    taskId: input.contract.taskId,
    taskContractDigest: taskContractDigest(input.contract),
    contextDigest: input.snapshot.contextDigest,
    baseCodeRevision: input.snapshot.baseCodeRevision,
    resultCodeRevision: input.currentCodeRevision,
    normativeRevisions: input.snapshot.normativeRevisions,
    evidenceIds: uniqueSorted([
      ...input.snapshot.requiredEvidenceIds,
      ...bundles.flatMap((bundle) => bundle.evidenceIds),
      ...(input.additionalEvidenceIds ?? []),
    ]),
    workItemIds: bundles.map((bundle) => bundle.workItemId),
    changeBundleIds: bundles.map((bundle) => bundle.bundleId),
    changedSymbolIds: bundles.flatMap((bundle) => bundle.changedSymbolIds),
    verificationRunIds: runs.map((run) => run.verificationRunId),
    reviewFindingIds: input.reviewFindings.map((finding) => finding.findingId),
    emergencyBypassIds: (input.emergencyBypasses ?? []).map((bypass) => bypass.bypassId),
    createdAt: input.createdAt,
  });
  return manifest;
}

function validateManifest(
  input: AssembleAccuracyManifestInput,
  manifest: AccuracyManifest,
): readonly AccuracyManifestIssue[] {
  return validateAccuracyManifestForTask({
    manifest,
    contract: input.contract,
    snapshot: input.snapshot,
    currentCodeRevision: input.currentCodeRevision,
    changeBundles: input.changeBundles,
    verificationRuns: input.verificationRuns,
    reviewFindings: input.reviewFindings,
  });
}

function isManifestAuditSelfEvidenceIssue(issue: AccuracyManifestIssue): boolean {
  return (
    issue.code === "verification_mismatch" && issue.ref === "full:query:kontext:manifest-audit"
  );
}

function runSatisfies(run: VerificationRun, requirement: VerificationRequirement): boolean {
  return (
    run.result === "passed" &&
    run.tier === requirement.tier &&
    run.verifierKind === requirement.verifier.kind &&
    run.verifierRef === requirement.verifier.ref &&
    requirement.subjectIds.every((subjectId) => run.subjectIds.includes(subjectId))
  );
}

function uniqueRequirements(
  requirements: readonly VerificationRequirement[],
): readonly VerificationRequirement[] {
  const values = new Map<string, VerificationRequirement>();
  for (const requirement of requirements) {
    values.set(
      JSON.stringify([requirement.tier, requirement.verifier.kind, requirement.verifier.ref]),
      requirement,
    );
  }
  return Array.from(values.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, requirement]) => requirement);
}

function quarantineApplies(
  record: import("@kontext-brain/spec").QuarantineRecord,
  bundle: ChangeBundle,
): boolean {
  if (record.workItemId === bundle.workItemId || record.taskId === bundle.taskId) return true;
  if (record.codeRevision !== bundle.resultRevision) return false;
  return (
    record.paths.some((changedPath) => bundle.changedPaths.includes(changedPath)) ||
    record.symbolIds.some((symbolId) => bundle.changedSymbolIds.includes(symbolId))
  );
}

function sameRevisionRefs(
  left: readonly NormativeRevisionRef[],
  right: readonly NormativeRevisionRef[],
): boolean {
  const keys = (values: readonly NormativeRevisionRef[]) =>
    values.map((value) => JSON.stringify([value.kind, value.recordId, value.revisionId]));
  return sameStrings(keys(left), keys(right));
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return JSON.stringify(uniqueSorted(left)) === JSON.stringify(uniqueSorted(right));
}

function uniqueSorted(values: readonly string[]): readonly string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}

function canonicalPath(value: string): string {
  return value.replaceAll("\\", "/").replace(/^\.\//, "");
}
