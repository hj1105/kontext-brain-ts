import type {
  AccuracyManifest,
  ChangeBundle,
  ContextReceipt,
  EmergencyBypass,
  LogicWorkItem,
  QuarantineRecord,
  ReviewFinding,
  TaskContextSnapshot,
  TaskContract,
  VerificationRun,
  VerificationTier,
  VerifierRef,
} from "@kontext-brain/spec";

export interface VerificationRequirement {
  readonly tier: VerificationTier;
  readonly verifier: VerifierRef;
  readonly subjectIds: readonly string[];
}

export interface VerificationPlan {
  readonly tier: VerificationTier;
  readonly requirements: readonly VerificationRequirement[];
}

export interface FastVerificationPlanInput {
  readonly affectedSymbolIds: readonly string[];
}

export interface TargetedVerificationPlanInput {
  readonly workItem: LogicWorkItem;
  readonly boundInvariantVerifiers?: readonly VerifierRef[];
}

export interface FullVerificationPlanInput {
  readonly contract: TaskContract;
  readonly boundInvariantVerifiers?: readonly VerifierRef[];
}

export interface VerificationBinding {
  readonly workspacePath: string;
  readonly codeRevision: string;
  readonly contextDigest: string;
  readonly observedAt: string;
}

export interface VerifierExecutionRequest extends VerificationBinding {
  readonly requirement: VerificationRequirement;
}

export interface VerifierAdapterResult {
  readonly result: "passed" | "failed";
  readonly output?: unknown;
}

export interface VerifierAdapter {
  execute(request: VerifierExecutionRequest): Promise<VerifierAdapterResult>;
}

export interface VerificationExecution {
  readonly run: VerificationRun;
  readonly disposition: "settled" | "retryable";
  readonly diagnostic?: string;
}

export type VerificationRetryStatus =
  | "queued"
  | "claimed"
  | "completed"
  | "superseded"
  | "exhausted";

export interface VerificationRetryJob {
  readonly jobId: string;
  readonly taskId: string;
  readonly workItemId?: string;
  readonly requirement: VerificationRequirement;
  readonly workspacePath: string;
  readonly codeRevision: string;
  readonly contextDigest: string;
  readonly status: VerificationRetryStatus;
  readonly retryCount: number;
  readonly maxRetries: number;
  readonly nextAttemptAt: string;
  readonly initialVerificationRunId: string;
  readonly lastVerificationRunId: string;
  readonly createdAt: string;
  readonly updatedAt: string;
  readonly claimId?: string;
  readonly leaseExpiresAt?: string;
}

export interface EnqueueVerificationRetryInput {
  readonly taskId: string;
  readonly workItemId?: string;
  readonly requirement: VerificationRequirement;
  readonly binding: VerificationBinding;
  readonly verificationRunId: string;
  readonly maxRetries: number;
  readonly nextAttemptAt: string;
}

export interface ClaimVerificationRetriesInput {
  readonly taskId: string;
  readonly now: string;
  readonly leaseExpiresAt: string;
  readonly limit: number;
}

export interface VerificationRetryQueue {
  enqueue(input: EnqueueVerificationRetryInput): Promise<VerificationRetryJob>;
  claimReady(input: ClaimVerificationRetriesInput): Promise<readonly VerificationRetryJob[]>;
  complete(
    job: VerificationRetryJob,
    verificationRunId: string,
    updatedAt: string,
  ): Promise<VerificationRetryJob>;
  reschedule(
    job: VerificationRetryJob,
    verificationRunId: string,
    nextAttemptAt: string,
    updatedAt: string,
  ): Promise<VerificationRetryJob>;
  supersedeObsolete(
    taskId: string,
    binding: Pick<VerificationBinding, "codeRevision" | "contextDigest">,
    updatedAt: string,
  ): Promise<readonly VerificationRetryJob[]>;
  list(status?: VerificationRetryStatus): Promise<readonly VerificationRetryJob[]>;
}

export interface ObservedChange {
  readonly codeRevision: string;
  readonly contextDigest?: string;
  readonly paths: readonly string[];
  readonly symbolIds: readonly string[];
  readonly observedAt: string;
  readonly preWriteAuthorizationObserved: boolean;
}

export interface AssessObservedChangeInput {
  readonly observed: ObservedChange;
  readonly workItem?: LogicWorkItem;
  readonly snapshot?: TaskContextSnapshot;
  readonly receipt?: ContextReceipt;
  readonly authorizedSymbolIds?: readonly string[];
}

export interface QuarantineAssessment {
  readonly quarantined: boolean;
  readonly record?: QuarantineRecord;
}

export interface QuarantineStore {
  put(record: QuarantineRecord): Promise<QuarantineRecord>;
  get(quarantineId: string): Promise<QuarantineRecord | undefined>;
  list(status?: QuarantineRecord["status"]): Promise<readonly QuarantineRecord[]>;
  release(quarantineId: string, releasedBy: string, releasedAt: string): Promise<QuarantineRecord>;
}

export interface ObservedPatch {
  readonly patchDigest: string;
  readonly changedPaths: readonly string[];
  readonly changedSymbolIds: readonly string[];
}

export type ChangeBundleIssueCode =
  | "invalid_bundle_id"
  | "task_mismatch"
  | "work_item_mismatch"
  | "base_revision_mismatch"
  | "result_revision_mismatch"
  | "context_mismatch"
  | "patch_mismatch"
  | "changed_paths_mismatch"
  | "changed_symbols_mismatch"
  | "missing_context_receipt"
  | "invalid_context_receipt"
  | "normative_revision_mismatch"
  | "evidence_mismatch"
  | "missing_verification"
  | "invalid_verification"
  | "unresolved_work"
  | "active_quarantine";

export interface ChangeBundleIssue {
  readonly code: ChangeBundleIssueCode;
  readonly message: string;
  readonly ref?: string;
}

export interface ValidateChangeBundleInput {
  readonly bundle: import("@kontext-brain/spec").ChangeBundle;
  readonly workItem: LogicWorkItem;
  readonly snapshot: TaskContextSnapshot;
  readonly currentCodeRevision: string;
  readonly observedPatch: ObservedPatch;
  readonly receipts: readonly ContextReceipt[];
  readonly verificationRuns: readonly VerificationRun[];
  readonly boundInvariantVerifiers?: readonly VerifierRef[];
  readonly quarantineRecords?: readonly QuarantineRecord[];
}

export interface ChangeBundleValidation {
  readonly accepted: boolean;
  readonly issues: readonly ChangeBundleIssue[];
}

export interface AssembleAccuracyManifestInput {
  readonly contract: TaskContract;
  readonly snapshot: TaskContextSnapshot;
  readonly currentCodeRevision: string;
  readonly changeBundles: readonly import("@kontext-brain/spec").ChangeBundle[];
  readonly verificationRuns: readonly VerificationRun[];
  readonly reviewFindings: readonly ReviewFinding[];
  readonly additionalEvidenceIds?: readonly string[];
  readonly emergencyBypasses?: readonly EmergencyBypass[];
  readonly createdAt: string;
}

export interface AccuracyManifestCandidateAudit {
  readonly passed: boolean;
  readonly candidate: AccuracyManifest;
  readonly blockingIssues: readonly import("@kontext-brain/spec").AccuracyManifestIssue[];
  readonly selfEvidenceIssues: readonly import("@kontext-brain/spec").AccuracyManifestIssue[];
}

export interface TaskCompletionArtifactStore {
  listVerificationRuns(taskId: string): Promise<readonly VerificationRun[]>;
  putVerificationRuns(
    taskId: string,
    runs: readonly VerificationRun[],
  ): Promise<readonly VerificationRun[]>;
  listChangeBundles(taskId: string): Promise<readonly ChangeBundle[]>;
  putChangeBundle(bundle: ChangeBundle): Promise<ChangeBundle>;
  getAccuracyManifest(taskId: string): Promise<AccuracyManifest | undefined>;
  putAccuracyManifest(manifest: AccuracyManifest): Promise<AccuracyManifest>;
}
