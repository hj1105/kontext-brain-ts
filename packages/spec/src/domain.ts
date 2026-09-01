export type OrganizationId = string;

export type GovernanceScope =
  | { readonly kind: "personal"; readonly subjectId: string }
  | { readonly kind: "workspace"; readonly workspaceId: string }
  | { readonly kind: "codebase"; readonly codebaseId: string }
  | { readonly kind: "organization"; readonly organizationId: OrganizationId };

export interface EvidenceRef {
  readonly evidenceId: string;
  readonly sourceSpan?: string;
}

export type VerifierKind = "test" | "typecheck" | "build" | "lint" | "query" | "manual_review";

export interface VerifierRef {
  readonly kind: VerifierKind;
  readonly ref: string;
}

export type DataClassification = "public" | "internal" | "confidential" | "restricted";

export interface RuntimeEgressPolicy {
  readonly dataClassification: DataClassification;
  readonly allowedRuntimeProviders: readonly string[];
}

export interface NormativeRevisionBase {
  readonly organizationId: OrganizationId;
  readonly recordId: string;
  readonly revisionId: string;
  readonly scope: GovernanceScope;
  readonly evidence: readonly EvidenceRef[];
  readonly egress: RuntimeEgressPolicy;
  readonly authoredBy: string;
  readonly authoredAt: string;
  readonly supersedesRevisionId?: string;
}

export interface DecisionRevision extends NormativeRevisionBase {
  readonly kind: "decision";
  readonly statement: string;
}

export interface DomainTermRevision extends NormativeRevisionBase {
  readonly kind: "domain_term";
  readonly term: string;
  readonly definition: string;
  readonly avoid?: readonly string[];
}

export interface InvariantRevision extends NormativeRevisionBase {
  readonly kind: "invariant";
  readonly statement: string;
  readonly verifiers: readonly VerifierRef[];
}

export type NormativeRevision = DecisionRevision | DomainTermRevision | InvariantRevision;

export interface NormativeProposal {
  readonly proposalId: string;
  readonly candidate: NormativeRevision;
  readonly proposedBy: string;
  readonly proposedAt: string;
}

export type ApprovalRole = "code_owner" | "domain_owner";

export type NormativeApproval =
  | {
      readonly kind: "local";
      readonly approvedBy: string;
      readonly approvedAt: string;
    }
  | {
      readonly kind: "merged";
      readonly approvedBy: string;
      readonly approvedAt: string;
      readonly mergeCommit: string;
      readonly roles: readonly ApprovalRole[];
    };

export interface NormativeActivation {
  readonly organizationId: OrganizationId;
  readonly kind: NormativeRevision["kind"];
  readonly recordId: string;
  readonly revisionId: string;
  readonly scope: GovernanceScope;
  readonly state: "accepted_local" | "accepted" | "retired";
  readonly acceptedBy: string;
  readonly acceptedAt: string;
  readonly mergeCommit?: string;
}

export interface NormativeManifest {
  readonly schemaVersion: 1;
  readonly organizationId: OrganizationId;
  readonly revisions: readonly NormativeRevision[];
  readonly activations: readonly NormativeActivation[];
}

export interface NormativeManifestIssue {
  readonly code:
    | "organization_mismatch"
    | "duplicate_revision"
    | "duplicate_activation"
    | "missing_revision"
    | "activation_mismatch"
    | "missing_superseded_revision"
    | "invalid_supersedes"
    | "missing_evidence";
  readonly message: string;
  readonly ref?: string;
}

export interface EffectiveNormativeRecord {
  readonly origin: "local" | "managed";
  readonly revision: NormativeRevision;
  readonly activation: NormativeActivation;
}

export interface NormativeLayerConflict {
  readonly kind: NormativeRevision["kind"];
  readonly recordId: string;
  readonly localRevisionId: string;
  readonly managedRevisionIds: readonly string[];
}

export interface NormativeLayerResolution {
  readonly effective: readonly EffectiveNormativeRecord[];
  readonly conflicts: readonly NormativeLayerConflict[];
  readonly canonicalizedLocalRevisionIds: readonly string[];
  readonly localOnlyRevisionIds: readonly string[];
  readonly contextStale: boolean;
}

export interface AcceptanceCriterion {
  readonly criterionId: string;
  readonly statement: string;
  readonly verifier: VerifierRef;
}

export interface TaskContract {
  readonly taskId: string;
  readonly intent: string;
  readonly acceptance: readonly AcceptanceCriterion[];
  readonly nonGoals: readonly string[];
  readonly targets: readonly string[];
  readonly risk: "low" | "medium" | "high";
}

export interface NormativeRevisionRef {
  readonly kind: NormativeRevision["kind"];
  readonly recordId: string;
  readonly revisionId: string;
}

export interface TaskContextDigestInput {
  readonly taskId: string;
  readonly baseCodeRevision: string;
  readonly effectiveScopes: readonly GovernanceScope[];
  readonly normativeRevisions: readonly NormativeRevisionRef[];
  readonly requiredEvidenceIds: readonly string[];
  readonly sourceFreshnessDigest: string;
}

export interface TaskContextSnapshot extends TaskContextDigestInput {
  readonly contextDigest: string;
  readonly createdAt: string;
}

export interface ContextReceipt {
  readonly receiptId: string;
  readonly taskId: string;
  readonly workItemId: string;
  readonly plannedSymbolIds: readonly string[];
  readonly allowedPaths: readonly string[];
  readonly contextDigest: string;
  readonly normativeRevisions: readonly NormativeRevisionRef[];
  readonly evidenceIds: readonly string[];
  readonly issuedAt: string;
  readonly expiresAt: string;
}

export type ContextAvailability = "current" | "stale" | "conflict" | "inaccessible" | "unavailable";

export interface ContextAssessment {
  readonly status: ContextAvailability;
  readonly contextDigest: string;
}

export type VerificationTier = "fast" | "targeted" | "full";

export interface VerificationRun {
  readonly verificationRunId: string;
  readonly tier: VerificationTier;
  readonly verifierKind: VerifierKind;
  readonly verifierRef: string;
  readonly codeRevision: string;
  readonly contextDigest: string;
  readonly subjectIds: readonly string[];
  readonly result: "passed" | "failed" | "inconclusive";
  readonly outputDigest?: string;
  readonly observedAt: string;
}

export interface LogicWorkItem {
  readonly workItemId: string;
  readonly taskId: string;
  readonly plannedSymbolIds: readonly string[];
  readonly dependsOn: readonly string[];
  readonly allowedPaths: readonly string[];
  readonly requiredVerifiers: readonly VerifierRef[];
  readonly capabilityId: string;
}

export interface ChangeBundle {
  readonly bundleId: string;
  readonly taskId: string;
  readonly workItemId: string;
  readonly baseRevision: string;
  readonly resultRevision: string;
  readonly taskContextDigest: string;
  readonly patchDigest: string;
  readonly changedSymbolIds: readonly string[];
  readonly changedPaths: readonly string[];
  readonly contextReceiptIds: readonly string[];
  readonly evidenceIds: readonly string[];
  readonly normativeRevisions: readonly NormativeRevisionRef[];
  readonly verificationRunIds: readonly string[];
  readonly proposals: readonly string[];
  readonly unresolved: readonly string[];
  readonly submittedAt: string;
}

export interface EmergencyBypass {
  readonly bypassId: string;
  readonly taskId: string;
  readonly approvedBy: string;
  readonly approvalRole: ApprovalRole;
  readonly reason: string;
  readonly issueCodes: readonly TaskTransitionIssueCode[];
  readonly issuedAt: string;
  readonly expiresAt: string;
}

export interface AccuracyManifest {
  readonly manifestId: string;
  readonly taskId: string;
  readonly taskContractDigest: string;
  readonly contextDigest: string;
  readonly baseCodeRevision: string;
  readonly resultCodeRevision: string;
  readonly normativeRevisions: readonly NormativeRevisionRef[];
  readonly evidenceIds: readonly string[];
  readonly workItemIds: readonly string[];
  readonly changeBundleIds: readonly string[];
  readonly changedSymbolIds: readonly string[];
  readonly verificationRunIds: readonly string[];
  readonly reviewFindingIds: readonly string[];
  readonly emergencyBypassIds: readonly string[];
  readonly createdAt: string;
}

export type QuarantineReason =
  | "missing_capability"
  | "expired_capability"
  | "context_mismatch"
  | "path_out_of_scope"
  | "symbol_out_of_scope"
  | "unobserved_write";

export interface QuarantineRecord {
  readonly quarantineId: string;
  readonly taskId?: string;
  readonly workItemId?: string;
  readonly codeRevision: string;
  readonly contextDigest?: string;
  readonly paths: readonly string[];
  readonly symbolIds: readonly string[];
  readonly reasons: readonly QuarantineReason[];
  readonly status: "active" | "released";
  readonly observedAt: string;
  readonly releasedAt?: string;
  readonly releasedBy?: string;
}

export interface DriftFinding {
  readonly findingId: string;
  readonly normativeKind: NormativeRevision["kind"];
  readonly recordId: string;
  readonly fromRevisionId: string;
  readonly toRevisionId: string;
  readonly codeRevision: string;
  readonly affectedSymbolIds: readonly string[];
  readonly unresolvedSymbolIds: readonly string[];
  readonly codeSymbolOntologyLinkIds: readonly string[];
  readonly evidenceIds: readonly string[];
  readonly status: "open" | "resolved" | "dismissed";
  readonly createdAt: string;
}

export interface CommitEvidence {
  readonly kind: "commit";
  readonly ref: string;
  readonly codeRevision: string;
  readonly contextDigest: string;
  readonly observedAt: string;
}

export interface ApprovalEvidence {
  readonly kind: "approval";
  readonly role: ApprovalRole;
  readonly ref: string;
  readonly codeRevision: string;
  readonly contextDigest: string;
  readonly observedAt: string;
}

export type TaskEvidence = CommitEvidence | ApprovalEvidence;

export type InvariantEvaluationStatus =
  | "guarded"
  | "unguarded"
  | "violated"
  | "inconclusive"
  | "retired";

export interface InvariantEvaluation {
  readonly invariantId: string;
  readonly revisionId: string;
  readonly status: InvariantEvaluationStatus;
  readonly verificationRunIds: readonly string[];
}

export type AgentRuntimeProvider = "codex" | "claude";

export interface ReviewFinding {
  readonly findingId: string;
  readonly status: "open" | "resolved" | "dismissed";
  readonly codeRevision: string;
  readonly contextDigest: string;
  readonly message: string;
  readonly reviewerProvider: AgentRuntimeProvider;
  readonly authorProviders: readonly AgentRuntimeProvider[];
  readonly reviewedAt: string;
  readonly symbolId?: string;
  readonly ruleRef?: string;
  readonly evidenceIds: readonly string[];
  readonly resolutionMessage?: string;
  readonly resolvedByProvider?: AgentRuntimeProvider;
  readonly resolvedAt?: string;
}

export type TaskState = "planned" | "in_progress" | "awaiting_evidence" | "done" | "blocked";

export type TaskTransitionIssueCode =
  | "invalid_task_contract"
  | "context_stale"
  | "context_conflict"
  | "context_inaccessible"
  | "context_unavailable"
  | "context_digest_mismatch"
  | "missing_commit"
  | "missing_acceptance_verification"
  | "failed_verification"
  | "inconclusive_verification"
  | "missing_invariant_evaluation"
  | "missing_invariant_verification"
  | "unguarded_invariant"
  | "violated_invariant"
  | "inconclusive_invariant"
  | "retired_invariant"
  | "unresolved_review_finding"
  | "missing_code_owner_approval"
  | "missing_domain_owner_approval"
  | "missing_accuracy_manifest"
  | "invalid_accuracy_manifest";

export interface TaskTransitionIssue {
  readonly code: TaskTransitionIssueCode;
  readonly message: string;
  readonly ref?: string;
}

export interface EvaluateTaskStateInput {
  readonly currentState: TaskState;
  readonly workStarted: boolean;
  readonly completionRequested: boolean;
  readonly contract: TaskContract;
  readonly snapshot: TaskContextSnapshot;
  readonly context: ContextAssessment;
  readonly currentCodeRevision: string;
  readonly evidence: readonly TaskEvidence[];
  readonly verificationRuns: readonly VerificationRun[];
  readonly invariantEvaluations: readonly InvariantEvaluation[];
  readonly reviewFindings: readonly ReviewFinding[];
  readonly changeBundles: readonly ChangeBundle[];
  readonly accuracyManifest?: AccuracyManifest;
}

export interface TaskStateEvaluation {
  readonly state: TaskState;
  readonly issues: readonly TaskTransitionIssue[];
}
