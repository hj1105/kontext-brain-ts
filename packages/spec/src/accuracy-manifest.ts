import { createHash } from "node:crypto";
import { isChangeBundleValid } from "./change-bundle.js";
import type {
  AccuracyManifest,
  ChangeBundle,
  NormativeRevisionRef,
  ReviewFinding,
  TaskContextSnapshot,
  TaskContract,
  VerificationRun,
  VerifierRef,
} from "./domain.js";

export type AccuracyManifestInput = Omit<AccuracyManifest, "manifestId">;

export type AccuracyManifestIssueCode =
  | "invalid_manifest_id"
  | "task_mismatch"
  | "contract_mismatch"
  | "context_mismatch"
  | "code_revision_mismatch"
  | "normative_revision_mismatch"
  | "evidence_mismatch"
  | "change_bundle_mismatch"
  | "verification_mismatch"
  | "review_finding_mismatch"
  | "unresolved_change_bundle";

export interface AccuracyManifestIssue {
  readonly code: AccuracyManifestIssueCode;
  readonly message: string;
  readonly ref?: string;
}

export interface ValidateAccuracyManifestInput {
  readonly manifest: AccuracyManifest;
  readonly contract: TaskContract;
  readonly snapshot: TaskContextSnapshot;
  readonly currentCodeRevision: string;
  readonly changeBundles: readonly ChangeBundle[];
  readonly verificationRuns: readonly VerificationRun[];
  readonly reviewFindings: readonly ReviewFinding[];
}

export const REQUIRED_COMPLETION_VERIFIERS: readonly VerifierRef[] = Object.freeze([
  { kind: "typecheck", ref: "workspace:typecheck" },
  { kind: "test", ref: "workspace:test" },
  { kind: "build", ref: "workspace:build" },
  { kind: "lint", ref: "workspace:lint" },
  { kind: "manual_review", ref: "kontext:independent-review" },
  { kind: "query", ref: "kontext:manifest-audit" },
]);

export function requiredCompletionVerifiers(risk: TaskContract["risk"]): readonly VerifierRef[] {
  return risk === "low"
    ? REQUIRED_COMPLETION_VERIFIERS.filter(
        (verifier) => verifier.ref !== "kontext:independent-review",
      )
    : REQUIRED_COMPLETION_VERIFIERS;
}

export function taskContractDigest(contract: TaskContract): string {
  return digest({
    taskId: contract.taskId,
    intent: contract.intent,
    acceptance: [...contract.acceptance]
      .map((criterion) => ({
        criterionId: criterion.criterionId,
        statement: criterion.statement,
        verifier: criterion.verifier,
      }))
      .sort((left, right) => left.criterionId.localeCompare(right.criterionId)),
    nonGoals: uniqueSorted(contract.nonGoals),
    targets: uniqueSorted(contract.targets),
    risk: contract.risk,
  });
}

export function createAccuracyManifest(input: AccuracyManifestInput): AccuracyManifest {
  const { manifestId: _ignoredManifestId, ...safeInput } = input as AccuracyManifest;
  const normalized: AccuracyManifestInput = {
    ...safeInput,
    normativeRevisions: normalizeRevisionRefs(safeInput.normativeRevisions),
    evidenceIds: uniqueSorted(safeInput.evidenceIds),
    workItemIds: uniqueSorted(safeInput.workItemIds),
    changeBundleIds: uniqueSorted(safeInput.changeBundleIds),
    changedSymbolIds: uniqueSorted(safeInput.changedSymbolIds),
    verificationRunIds: uniqueSorted(safeInput.verificationRunIds),
    reviewFindingIds: uniqueSorted(safeInput.reviewFindingIds),
    emergencyBypassIds: uniqueSorted(safeInput.emergencyBypassIds),
  };
  return Object.freeze({
    ...normalized,
    manifestId: `accuracy-manifest:${sha256(JSON.stringify(stableValue(normalized)))}`,
  });
}

export function isAccuracyManifestValid(manifest: AccuracyManifest): boolean {
  const { manifestId: _manifestId, ...input } = manifest;
  return (
    JSON.stringify(stableValue(manifest)) ===
    JSON.stringify(stableValue(createAccuracyManifest(input)))
  );
}

export function validateAccuracyManifestForTask(
  input: ValidateAccuracyManifestInput,
): readonly AccuracyManifestIssue[] {
  const issues: AccuracyManifestIssue[] = [];
  const add = (code: AccuracyManifestIssueCode, message: string, ref?: string): void => {
    if (!issues.some((issue) => issue.code === code && issue.ref === ref)) {
      issues.push({ code, message, ref });
    }
  };
  const { manifest, contract, snapshot } = input;
  if (!isAccuracyManifestValid(manifest)) {
    add("invalid_manifest_id", "Accuracy Manifest content does not match its immutable ID");
  }
  if (manifest.taskId !== contract.taskId || snapshot.taskId !== contract.taskId) {
    add("task_mismatch", "Accuracy Manifest, Task Contract, and Snapshot Task IDs must match");
  }
  if (manifest.taskContractDigest !== taskContractDigest(contract)) {
    add("contract_mismatch", "Accuracy Manifest references another Task Contract");
  }
  if (manifest.contextDigest !== snapshot.contextDigest) {
    add("context_mismatch", "Accuracy Manifest references another Task Context Snapshot");
  }
  if (
    manifest.baseCodeRevision !== snapshot.baseCodeRevision ||
    manifest.resultCodeRevision !== input.currentCodeRevision
  ) {
    add("code_revision_mismatch", "Accuracy Manifest code revisions are not current");
  }
  if (
    !sameStrings(
      revisionKeys(manifest.normativeRevisions),
      revisionKeys(snapshot.normativeRevisions),
    )
  ) {
    add(
      "normative_revision_mismatch",
      "Accuracy Manifest normative revisions do not match the Snapshot",
    );
  }

  const bundles = input.changeBundles.filter((bundle) => bundle.taskId === contract.taskId);
  if (bundles.length === 0) {
    add("change_bundle_mismatch", "Accuracy Manifest requires at least one Change Bundle");
  }
  if (
    !sameStrings(
      manifest.changeBundleIds,
      bundles.map((bundle) => bundle.bundleId),
    )
  ) {
    add("change_bundle_mismatch", "Accuracy Manifest must include every submitted Change Bundle");
  }
  for (const bundle of bundles) {
    if (!isChangeBundleValid(bundle)) {
      add(
        "change_bundle_mismatch",
        `Change Bundle ${bundle.bundleId} content does not match its immutable ID`,
        bundle.bundleId,
      );
    }
    if (
      bundle.taskContextDigest !== snapshot.contextDigest ||
      !manifest.workItemIds.includes(bundle.workItemId) ||
      !bundle.changedSymbolIds.every((symbolId) => manifest.changedSymbolIds.includes(symbolId))
    ) {
      add(
        "change_bundle_mismatch",
        `Change Bundle ${bundle.bundleId} is not bound to the completion`,
        bundle.bundleId,
      );
    }
    if (bundle.unresolved.length > 0) {
      add(
        "unresolved_change_bundle",
        `Change Bundle ${bundle.bundleId} has unresolved items`,
        bundle.bundleId,
      );
    }
  }

  const currentPassingRuns = input.verificationRuns.filter(
    (run) =>
      run.codeRevision === input.currentCodeRevision &&
      run.contextDigest === snapshot.contextDigest &&
      run.result === "passed",
  );
  const validBundleRunIds = new Set(
    bundles.flatMap((bundle) =>
      bundle.verificationRunIds.filter((runId) =>
        input.verificationRuns.some(
          (run) =>
            run.verificationRunId === runId &&
            run.codeRevision === bundle.resultRevision &&
            run.contextDigest === snapshot.contextDigest &&
            run.result === "passed",
        ),
      ),
    ),
  );
  const declaredBundleRunIds = uniqueSorted(bundles.flatMap((bundle) => bundle.verificationRunIds));
  if (!declaredBundleRunIds.every((runId) => validBundleRunIds.has(runId))) {
    add(
      "verification_mismatch",
      "A Change Bundle Verification Run is not passing on its submitted revision",
    );
  }
  const requiredVerifiers = requiredCompletionVerifiers(contract.risk);
  for (const verifier of requiredVerifiers) {
    if (
      !currentPassingRuns.some(
        (run) =>
          run.tier === "full" &&
          run.verifierKind === verifier.kind &&
          run.verifierRef === verifier.ref &&
          run.subjectIds.includes(contract.taskId),
      )
    ) {
      add(
        "verification_mismatch",
        `Accuracy Manifest lacks full ${verifier.kind}:${verifier.ref}`,
        `full:${verifier.kind}:${verifier.ref}`,
      );
    }
  }
  for (const criterion of contract.acceptance) {
    if (
      !currentPassingRuns.some(
        (run) =>
          run.tier === "full" &&
          run.verifierKind === criterion.verifier.kind &&
          run.verifierRef === criterion.verifier.ref &&
          run.subjectIds.includes(contract.taskId),
      )
    ) {
      add(
        "verification_mismatch",
        `Accuracy Manifest lacks full acceptance verifier ${criterion.verifier.kind}:${criterion.verifier.ref}`,
        criterion.criterionId,
      );
    }
  }
  const requiredRunIds = uniqueSorted([
    ...bundles.flatMap((bundle) => bundle.verificationRunIds),
    ...requiredVerifiers.flatMap((verifier) =>
      currentPassingRuns
        .filter(
          (run) =>
            run.tier === "full" &&
            run.verifierKind === verifier.kind &&
            run.verifierRef === verifier.ref &&
            run.subjectIds.includes(contract.taskId),
        )
        .map((run) => run.verificationRunId),
    ),
    ...contract.acceptance.flatMap((criterion) =>
      currentPassingRuns
        .filter(
          (run) =>
            run.tier === "full" &&
            run.verifierKind === criterion.verifier.kind &&
            run.verifierRef === criterion.verifier.ref &&
            run.subjectIds.includes(contract.taskId),
        )
        .map((run) => run.verificationRunId),
    ),
  ]);
  if (
    !requiredRunIds.every((runId) => manifest.verificationRunIds.includes(runId)) ||
    !manifest.verificationRunIds.every(
      (runId) =>
        currentPassingRuns.some((run) => run.verificationRunId === runId) ||
        validBundleRunIds.has(runId),
    )
  ) {
    add(
      "verification_mismatch",
      "Accuracy Manifest verification runs are missing or not current and passing",
    );
  }

  const requiredEvidenceIds = uniqueSorted([
    ...snapshot.requiredEvidenceIds,
    ...bundles.flatMap((bundle) => bundle.evidenceIds),
  ]);
  if (!requiredEvidenceIds.every((evidenceId) => manifest.evidenceIds.includes(evidenceId))) {
    add("evidence_mismatch", "Accuracy Manifest omits required Evidence");
  }
  if (
    !sameStrings(
      manifest.reviewFindingIds,
      input.reviewFindings.map((finding) => finding.findingId),
    )
  ) {
    add("review_finding_mismatch", "Accuracy Manifest must include every Review Finding");
  }
  for (const finding of input.reviewFindings) {
    const authorProviders = new Set(finding.authorProviders);
    const invalidResolution =
      finding.status === "open"
        ? finding.resolutionMessage !== undefined ||
          finding.resolvedByProvider !== undefined ||
          finding.resolvedAt !== undefined
        : !finding.resolutionMessage || !finding.resolvedByProvider || !finding.resolvedAt;
    if (
      finding.codeRevision !== input.currentCodeRevision ||
      finding.contextDigest !== snapshot.contextDigest ||
      authorProviders.has(finding.reviewerProvider) ||
      (finding.resolvedByProvider !== undefined &&
        authorProviders.has(finding.resolvedByProvider)) ||
      invalidResolution
    ) {
      add(
        "review_finding_mismatch",
        `Review Finding ${finding.findingId} lacks current independent provenance`,
        finding.findingId,
      );
    }
  }
  return issues;
}

function normalizeRevisionRefs(
  revisions: readonly NormativeRevisionRef[],
): readonly NormativeRevisionRef[] {
  const byKey = new Map(revisions.map((revision) => [revisionKey(revision), revision] as const));
  return Array.from(byKey.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, revision]) => revision);
}

function revisionKeys(revisions: readonly NormativeRevisionRef[]): readonly string[] {
  return revisions.map(revisionKey);
}

function revisionKey(revision: NormativeRevisionRef): string {
  return JSON.stringify([revision.kind, revision.recordId, revision.revisionId]);
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return JSON.stringify(uniqueSorted(left)) === JSON.stringify(uniqueSorted(right));
}

function uniqueSorted(values: readonly string[]): string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}

function digest(value: unknown): string {
  return `sha256:${sha256(JSON.stringify(stableValue(value)))}`;
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, nested]) => [key, stableValue(nested)]),
    );
  }
  return value;
}
