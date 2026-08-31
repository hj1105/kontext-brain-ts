import type {
  ContextAvailability,
  ContextReceipt,
  EffectiveNormativeRecord,
  GovernanceScope,
  NormativeLayerConflict,
  NormativeRevision,
  TaskContextSnapshot,
  TaskContract,
} from "@kontext-brain/spec";

export interface ContextEvidenceItem {
  readonly evidenceId: string;
  readonly text: string;
  readonly sourceSpan?: string;
  readonly availability: ContextAvailability;
  readonly allowedRuntimeProviders: readonly string[];
  readonly relevance?: number;
}

export interface LogicContextTarget {
  readonly workItemId: string;
  readonly plannedSymbolIds: readonly string[];
}

export interface CompileTaskContextInput {
  readonly contract: TaskContract;
  readonly snapshot: TaskContextSnapshot;
  readonly currentCodeRevision: string;
  readonly currentSourceFreshnessDigest: string;
  readonly currentEffectiveScopes: readonly GovernanceScope[];
  readonly currentNormativeRecords: readonly EffectiveNormativeRecord[];
  readonly normativeRevisionCatalog: readonly NormativeRevision[];
  readonly conflicts: readonly NormativeLayerConflict[];
  readonly evidence: readonly ContextEvidenceItem[];
  readonly runtimeProvider: string;
  readonly logic: LogicContextTarget;
  readonly authorizedPaths: readonly string[];
  readonly issuedAt: string;
  readonly expiresAt: string;
  readonly totalTokenBudget: number;
  readonly optionalEvidenceTokenBudget: number;
}

export type ContextCompilationIssueCode =
  | "invalid_snapshot"
  | "code_revision_stale"
  | "source_freshness_stale"
  | "scope_stale"
  | "normative_revision_stale"
  | "normative_conflict"
  | "mandatory_context_inaccessible"
  | "mandatory_context_unavailable"
  | "mandatory_context_stale"
  | "mandatory_context_conflict"
  | "invalid_logic_target"
  | "invalid_receipt_expiry"
  | "optional_context_omitted"
  | "mandatory_budget_exceeded";

export interface ContextCompilationIssue {
  readonly code: ContextCompilationIssueCode;
  readonly message: string;
  readonly ref?: string;
}

export interface CompiledMandatoryContext {
  readonly taskContract: TaskContract;
  readonly normativeRevisions: readonly NormativeRevision[];
  readonly evidence: readonly ContextEvidenceItem[];
}

export interface ContextTokenUsage {
  readonly mandatory: number;
  readonly optional: number;
  readonly optionalBudget: number;
}

export interface CompiledTaskContext {
  readonly status: ContextAvailability;
  readonly editingAllowed: boolean;
  readonly contextDigest: string;
  readonly mandatory: CompiledMandatoryContext;
  readonly optionalEvidence: readonly ContextEvidenceItem[];
  readonly issues: readonly ContextCompilationIssue[];
  readonly tokenUsage: ContextTokenUsage;
  readonly receipt?: ContextReceipt;
}

export interface PrepareTaskContextSnapshotInput {
  readonly contract: TaskContract;
  readonly baseCodeRevision: string;
  readonly effectiveScopes: readonly GovernanceScope[];
  readonly normativeRecords: readonly EffectiveNormativeRecord[];
  readonly additionalRequiredEvidenceIds?: readonly string[];
  readonly sourceFreshnessDigest: string;
  readonly createdAt: string;
}

export interface ContextTokenEstimator {
  estimate(value: string): number;
}
