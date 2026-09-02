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

/**
 * Curated links from a Planned Symbol to the normative records that govern it.
 *
 * Without these the compiler can only filter by activation state and runtime
 * egress, so every accepted record in the organization becomes mandatory
 * context for every Work Item. At a few dozen records that is merely wasteful;
 * at the several hundred a real Codebase carries it exhausts the token budget
 * and the caller has to pre-select the right records by hand, which is not a
 * capability the product provides.
 */
export interface PlannedSymbolGovernanceLink {
  readonly plannedSymbolId: string;
  readonly recordId: string;
  readonly revisionId: string;
  readonly origin: "curated" | "deterministic" | "proposed";
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
  /**
   * When present, mandatory normative context is narrowed to records reachable
   * from this Work Item's Planned Symbols. Omitting it keeps the previous
   * organization-wide behaviour.
   */
  readonly governanceLinks?: readonly PlannedSymbolGovernanceLink[];
  /** Evidence explicitly requested by the Task stays mandatory after symbol narrowing. */
  readonly additionalRequiredEvidenceIds?: readonly string[];
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
  | "mandatory_budget_exceeded"
  | "ungoverned_planned_symbol";

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
