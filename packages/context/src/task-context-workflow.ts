import type { PlannedSymbolRecord } from "@kontext-brain/code";
import type {
  EffectiveNormativeRecord,
  GovernanceScope,
  NormativeLayerConflict,
  NormativeRevision,
  TaskContextSnapshot,
  TaskContract,
  VerifierRef,
} from "@kontext-brain/spec";
import { ContextCompiler } from "./context-compiler.js";
import { prepareTaskContextSnapshot } from "./context-compiler.js";
import type {
  CompiledTaskContext,
  ContextEvidenceItem,
  LogicContextTarget,
  PlannedSymbolGovernanceLink,
} from "./domain.js";

export interface CurrentTaskContextState {
  readonly codeRevision: string;
  readonly sourceFreshnessDigest: string;
  readonly effectiveScopes: readonly GovernanceScope[];
  readonly normativeRecords: readonly EffectiveNormativeRecord[];
  readonly normativeRevisionCatalog: readonly NormativeRevision[];
  readonly conflicts: readonly NormativeLayerConflict[];
  readonly evidence: readonly ContextEvidenceItem[];
  readonly logicPlans: readonly LogicWorkPlan[];
  readonly governanceLinks?: readonly PlannedSymbolGovernanceLink[];
}

export interface LogicWorkPlan {
  readonly workItemId: string;
  readonly plannedSymbolIds: readonly string[];
  readonly plannedSymbols?: readonly PlannedSymbolRecord[];
  readonly allowedPaths: readonly string[];
  readonly dependsOn?: readonly string[];
  readonly requiredVerifiers?: readonly VerifierRef[];
  readonly capabilityId?: string;
}

export interface PreparedTaskContext {
  readonly contract: TaskContract;
  readonly snapshot: TaskContextSnapshot;
  readonly additionalRequiredEvidenceIds: readonly string[];
}

export interface TaskContextStateProvider {
  getCurrent(taskId: string): Promise<CurrentTaskContextState>;
}

export interface PreparedTaskContextStore {
  get(taskId: string): Promise<PreparedTaskContext | undefined>;
  put(value: PreparedTaskContext): Promise<void>;
}

export interface PrepareTaskRequest {
  readonly contract: TaskContract;
  readonly additionalRequiredEvidenceIds?: readonly string[];
  readonly createdAt: string;
}

export interface BeginLogicRequest {
  readonly taskId: string;
  readonly logic: LogicContextTarget;
  readonly runtimeProvider: string;
  readonly issuedAt: string;
  readonly expiresAt: string;
  readonly totalTokenBudget: number;
  readonly optionalEvidenceTokenBudget: number;
}

export interface RefreshTaskContextRequest {
  readonly taskId: string;
  readonly createdAt: string;
}

export interface TaskContextRefreshResult {
  readonly previous: TaskContextSnapshot;
  readonly current: TaskContextSnapshot;
  readonly changed: boolean;
  readonly addedNormativeRevisionIds: readonly string[];
  readonly removedNormativeRevisionIds: readonly string[];
  readonly addedEvidenceIds: readonly string[];
  readonly removedEvidenceIds: readonly string[];
}

export class TaskContextWorkflow {
  constructor(
    private readonly stateProvider: TaskContextStateProvider,
    private readonly store: PreparedTaskContextStore,
    private readonly compiler: ContextCompiler = new ContextCompiler(),
  ) {}

  async prepareTask(request: PrepareTaskRequest): Promise<PreparedTaskContext> {
    const state = await this.stateProvider.getCurrent(request.contract.taskId);
    const prepared: PreparedTaskContext = {
      contract: request.contract,
      snapshot: prepareTaskContextSnapshot({
        contract: request.contract,
        baseCodeRevision: state.codeRevision,
        effectiveScopes: state.effectiveScopes,
        normativeRecords: state.normativeRecords,
        additionalRequiredEvidenceIds: request.additionalRequiredEvidenceIds,
        sourceFreshnessDigest: state.sourceFreshnessDigest,
        createdAt: request.createdAt,
      }),
      additionalRequiredEvidenceIds: uniqueSorted(request.additionalRequiredEvidenceIds ?? []),
    };
    await this.store.put(prepared);
    return prepared;
  }

  async beginLogic(request: BeginLogicRequest): Promise<CompiledTaskContext> {
    const prepared = await this.requirePrepared(request.taskId);
    const state = await this.stateProvider.getCurrent(request.taskId);
    const plan = state.logicPlans.find(
      (candidate) =>
        candidate.workItemId === request.logic.workItemId &&
        sameStrings(candidate.plannedSymbolIds, request.logic.plannedSymbolIds),
    );
    return this.compiler.compile({
      contract: prepared.contract,
      snapshot: prepared.snapshot,
      currentCodeRevision: state.codeRevision,
      currentSourceFreshnessDigest: state.sourceFreshnessDigest,
      currentEffectiveScopes: state.effectiveScopes,
      currentNormativeRecords: state.normativeRecords,
      normativeRevisionCatalog: state.normativeRevisionCatalog,
      conflicts: state.conflicts,
      evidence: state.evidence,
      runtimeProvider: request.runtimeProvider,
      logic: request.logic,
      governanceLinks: state.governanceLinks,
      additionalRequiredEvidenceIds: prepared.additionalRequiredEvidenceIds,
      authorizedPaths: plan?.allowedPaths ?? [],
      issuedAt: request.issuedAt,
      expiresAt: request.expiresAt,
      totalTokenBudget: request.totalTokenBudget,
      optionalEvidenceTokenBudget: request.optionalEvidenceTokenBudget,
    });
  }

  async refreshTaskContext(request: RefreshTaskContextRequest): Promise<TaskContextRefreshResult> {
    const previous = await this.requirePrepared(request.taskId);
    const currentState = await this.stateProvider.getCurrent(request.taskId);
    const current: PreparedTaskContext = {
      ...previous,
      snapshot: prepareTaskContextSnapshot({
        contract: previous.contract,
        baseCodeRevision: currentState.codeRevision,
        effectiveScopes: currentState.effectiveScopes,
        normativeRecords: currentState.normativeRecords,
        additionalRequiredEvidenceIds: previous.additionalRequiredEvidenceIds,
        sourceFreshnessDigest: currentState.sourceFreshnessDigest,
        createdAt: request.createdAt,
      }),
    };
    await this.store.put(current);
    return {
      previous: previous.snapshot,
      current: current.snapshot,
      changed: previous.snapshot.contextDigest !== current.snapshot.contextDigest,
      ...snapshotDiff(previous.snapshot, current.snapshot),
    };
  }

  private async requirePrepared(taskId: string): Promise<PreparedTaskContext> {
    const prepared = await this.store.get(taskId);
    if (!prepared) throw new Error(`Task "${taskId}" has no prepared context`);
    return prepared;
  }
}

export class InMemoryPreparedTaskContextStore implements PreparedTaskContextStore {
  private readonly values = new Map<string, PreparedTaskContext>();

  async get(taskId: string): Promise<PreparedTaskContext | undefined> {
    return this.values.get(taskId);
  }

  async put(value: PreparedTaskContext): Promise<void> {
    this.values.set(value.contract.taskId, value);
  }
}

export class InMemoryTaskContextStateProvider implements TaskContextStateProvider {
  private readonly values = new Map<string, CurrentTaskContextState>();

  set(taskId: string, state: CurrentTaskContextState): void {
    this.values.set(taskId, state);
  }

  async getCurrent(taskId: string): Promise<CurrentTaskContextState> {
    const state = this.values.get(taskId);
    if (!state) throw new Error(`Task "${taskId}" has no current context state`);
    return state;
  }
}

function snapshotDiff(
  previous: TaskContextSnapshot,
  current: TaskContextSnapshot,
): Omit<TaskContextRefreshResult, "previous" | "current" | "changed"> {
  const previousRevisions = new Set(
    previous.normativeRevisions.map((revision) => revision.revisionId),
  );
  const currentRevisions = new Set(
    current.normativeRevisions.map((revision) => revision.revisionId),
  );
  const previousEvidence = new Set(previous.requiredEvidenceIds);
  const currentEvidence = new Set(current.requiredEvidenceIds);
  return {
    addedNormativeRevisionIds: sortedDifference(currentRevisions, previousRevisions),
    removedNormativeRevisionIds: sortedDifference(previousRevisions, currentRevisions),
    addedEvidenceIds: sortedDifference(currentEvidence, previousEvidence),
    removedEvidenceIds: sortedDifference(previousEvidence, currentEvidence),
  };
}

function sortedDifference(
  left: ReadonlySet<string>,
  right: ReadonlySet<string>,
): readonly string[] {
  return Array.from(left)
    .filter((value) => !right.has(value))
    .sort((a, b) => a.localeCompare(b));
}

function uniqueSorted(values: readonly string[]): readonly string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return JSON.stringify(uniqueSorted(left)) === JSON.stringify(uniqueSorted(right));
}
