import { createHash, randomUUID } from "node:crypto";
import type { LogicWorkItem } from "@kontext-brain/spec";
import type {
  AgentRuntimePort,
  RuntimeCapabilitySnapshot,
  RuntimeCheckpoint,
  RuntimeLease,
  RuntimeLeaseStore,
  RuntimeProvider,
  RuntimeSession,
  RuntimeWorkPreparationPort,
  RuntimeWorktree,
  RuntimeWorktreePort,
} from "./runtime.js";
import { createRuntimeCheckpoint } from "./runtime.js";

export interface ScheduledLogicWork {
  readonly workItem: LogicWorkItem;
  readonly prompt: string;
  readonly codeRevision: string;
  readonly contextDigest: string;
  readonly eligibleProviders: readonly RuntimeProvider[];
  readonly pinnedProvider?: RuntimeProvider;
  readonly totalTokenBudget?: number;
  readonly optionalEvidenceTokenBudget?: number;
  readonly receiptTtlSeconds?: number;
}

export interface WorkItemScheduleInput {
  readonly taskId: string;
  readonly work: readonly ScheduledLogicWork[];
  readonly initialCapabilities?: readonly RuntimeCapabilitySnapshot[];
  readonly initialResults?: readonly ScheduledWorkResult[];
  readonly onProgress?: (result: WorkItemScheduleResult) => Promise<void> | void;
  readonly maxConcurrency?: number;
  readonly maxRetries?: number;
  readonly leaseDurationMilliseconds?: number;
  readonly signal?: AbortSignal;
}

export interface ScheduledWorkResult {
  readonly workItemId: string;
  readonly status: "completed" | "failed";
  readonly provider?: RuntimeProvider;
  readonly worktree?: RuntimeWorktree;
  readonly session?: RuntimeSession;
  readonly attempts: number;
  readonly checkpoints: readonly RuntimeCheckpoint[];
  readonly diagnostics: readonly string[];
}

export interface WorkItemScheduleResult {
  readonly capabilities: readonly RuntimeCapabilitySnapshot[];
  readonly results: readonly ScheduledWorkResult[];
}

export class RuntimeScheduleCancelledError extends Error {
  constructor() {
    super("Runtime schedule was cancelled");
    this.name = "RuntimeScheduleCancelledError";
  }
}

export class WorkItemScheduler {
  private readonly runtimes: ReadonlyMap<RuntimeProvider, AgentRuntimePort>;

  constructor(
    runtimes: readonly AgentRuntimePort[],
    private readonly worktrees: RuntimeWorktreePort,
    private readonly leases: RuntimeLeaseStore,
    private readonly now: () => Date = () => new Date(),
    private readonly preparation?: RuntimeWorkPreparationPort,
  ) {
    this.runtimes = new Map(runtimes.map((runtime) => [runtime.provider, runtime]));
  }

  async run(input: WorkItemScheduleInput): Promise<WorkItemScheduleResult> {
    throwIfCancelled(input.signal);
    const maxConcurrency = Math.max(1, Math.min(4, input.maxConcurrency ?? 4));
    const maxRetries = Math.max(0, Math.min(2, input.maxRetries ?? 2));
    const workById = new Map(input.work.map((work) => [work.workItem.workItemId, work] as const));
    if (workById.size !== input.work.length) throw new Error("Logic Work Item IDs must be unique");
    const dependencies = effectiveDependencies(input.work);
    assertKnownDependencies(workById, dependencies);

    const results = validatedInitialResults(input.initialResults ?? [], workById);
    const pending = new Set(
      Array.from(workById.keys())
        .filter((workItemId) => !results.has(workItemId))
        .sort(),
    );
    const completed = new Set(
      Array.from(results.values())
        .filter((result) => result.status === "completed")
        .map((result) => result.workItemId),
    );
    const running = new Map<string, Promise<{ id: string; result: ScheduledWorkResult }>>();
    propagateBlockedResults(pending, results, dependencies);
    if (pending.size === 0) {
      return scheduleResult(input.initialCapabilities ?? [], results);
    }
    const capabilities = await Promise.all(
      Array.from(this.runtimes.values()).map((runtime) => runtime.inspectCapabilities()),
    );
    throwIfCancelled(input.signal);
    const capabilityByProvider = new Map(
      capabilities.map((capability) => [capability.provider, capability] as const),
    );

    while (pending.size > 0 || running.size > 0) {
      if (input.signal?.aborted) {
        await Promise.allSettled(running.values());
        throw new RuntimeScheduleCancelledError();
      }
      const ready = Array.from(pending)
        .filter((workItemId) =>
          (dependencies.get(workItemId) ?? []).every((dependency) => completed.has(dependency)),
        )
        .sort();
      while (ready.length > 0 && running.size < maxConcurrency) {
        const workItemId = ready.shift();
        if (!workItemId || !pending.has(workItemId)) continue;
        const work = workById.get(workItemId);
        if (!work) continue;
        pending.delete(workItemId);
        const promise = this.executeWork(
          input.taskId,
          work,
          capabilityByProvider,
          maxRetries,
          input.leaseDurationMilliseconds ?? 15 * 60_000,
          input.signal,
        ).then((result) => ({ id: workItemId, result }));
        running.set(workItemId, promise);
      }
      if (running.size === 0) {
        throw new Error("Logic Work Item dependency graph contains a cycle or blocked dependency");
      }
      let settled: { id: string; result: ScheduledWorkResult };
      try {
        settled = await Promise.race(running.values());
      } catch (error) {
        await Promise.allSettled(running.values());
        throw error;
      }
      running.delete(settled.id);
      results.set(settled.id, settled.result);
      if (settled.result.status === "completed") completed.add(settled.id);
      else propagateBlockedResults(pending, results, dependencies);
      await input.onProgress?.(scheduleResult(capabilities, results));
    }
    return scheduleResult(capabilities, results);
  }

  private async executeWork(
    taskId: string,
    work: ScheduledLogicWork,
    capabilities: ReadonlyMap<RuntimeProvider, RuntimeCapabilitySnapshot>,
    maxRetries: number,
    leaseDurationMilliseconds: number,
    signal?: AbortSignal,
  ): Promise<ScheduledWorkResult> {
    throwIfCancelled(signal);
    const providers = eligibleProviders(work, capabilities, this.runtimes);
    if (providers.length === 0) {
      return {
        workItemId: work.workItem.workItemId,
        status: "failed",
        attempts: 0,
        checkpoints: [],
        diagnostics: ["No authenticated subscription runtime is eligible"],
      };
    }
    const worktree = await this.worktrees.prepare({
      taskId,
      workItem: work.workItem,
      baseRevision: work.codeRevision,
    });
    throwIfCancelled(signal);
    const checkpoints: RuntimeCheckpoint[] = [];
    const diagnostics: string[] = [];
    let lastSession: RuntimeSession | undefined;
    let lastProvider: RuntimeProvider | undefined;
    const maximumAttempts = maxRetries + 1;
    for (let attempt = 0; attempt < maximumAttempts; attempt++) {
      throwIfCancelled(signal);
      const provider = providers[attempt % providers.length];
      if (!provider) break;
      const runtime = this.runtimes.get(provider);
      if (!runtime) continue;
      const acquiredAt = this.now().toISOString();
      const lease: RuntimeLease = {
        leaseId: `runtime-lease:${randomUUID()}`,
        taskId,
        workItemId: work.workItem.workItemId,
        provider,
        workspacePath: worktree.workspacePath,
        symbolIds: work.workItem.plannedSymbolIds,
        paths: work.workItem.allowedPaths,
        acquiredAt,
        expiresAt: new Date(Date.parse(acquiredAt) + leaseDurationMilliseconds).toISOString(),
      };
      if (!(await this.leases.acquire(lease))) {
        diagnostics.push(`Write lease conflict for ${work.workItem.workItemId}`);
        continue;
      }
      try {
        throwIfCancelled(signal);
        await this.preparation?.prepare({
          taskId,
          workItem: work.workItem,
          worktree,
          provider,
          attempt: attempt + 1,
          totalTokenBudget: work.totalTokenBudget ?? 100_000,
          optionalEvidenceTokenBudget: work.optionalEvidenceTokenBudget ?? 10_000,
          receiptTtlSeconds: work.receiptTtlSeconds ?? 900,
        });
        throwIfCancelled(signal);
        lastProvider = provider;
        lastSession = await runtime.start({
          taskId,
          workItem: work.workItem,
          workspacePath: worktree.workspacePath,
          prompt: work.prompt,
          codeRevision: work.codeRevision,
          contextDigest: work.contextDigest,
          checkpoint: checkpoints.at(-1),
          signal,
        });
        throwIfCancelled(signal);
        const checkpoint = createRuntimeCheckpoint({
          taskId,
          workItemId: work.workItem.workItemId,
          provider,
          providerSessionId: lastSession.providerSessionId,
          workspacePath: worktree.workspacePath,
          codeRevision: work.codeRevision,
          contextDigest: work.contextDigest,
          createdAt: lastSession.completedAt,
        });
        checkpoints.push(checkpoint);
        if (lastSession.status === "completed") {
          return {
            workItemId: work.workItem.workItemId,
            status: "completed",
            provider,
            worktree,
            session: lastSession,
            attempts: attempt + 1,
            checkpoints,
            diagnostics,
          };
        }
        diagnostics.push(lastSession.diagnostic ?? `${provider} worker failed`);
      } catch (error) {
        if (error instanceof RuntimeScheduleCancelledError || signal?.aborted) {
          throw new RuntimeScheduleCancelledError();
        }
        diagnostics.push(error instanceof Error ? error.message : String(error));
      } finally {
        await this.leases.release(lease.leaseId, this.now().toISOString());
      }
    }
    return {
      workItemId: work.workItem.workItemId,
      status: "failed",
      provider: lastProvider,
      worktree,
      session: lastSession,
      attempts: maximumAttempts,
      checkpoints,
      diagnostics,
    };
  }
}

function validatedInitialResults(
  initialResults: readonly ScheduledWorkResult[],
  workById: ReadonlyMap<string, ScheduledLogicWork>,
): Map<string, ScheduledWorkResult> {
  const results = new Map<string, ScheduledWorkResult>();
  for (const result of initialResults) {
    const work = workById.get(result.workItemId);
    if (!work) throw new Error(`Checkpoint contains unknown Logic Work Item ${result.workItemId}`);
    if (results.has(result.workItemId)) {
      throw new Error(`Checkpoint repeats Logic Work Item ${result.workItemId}`);
    }
    for (const checkpoint of result.checkpoints) {
      if (
        checkpoint.taskId !== work.workItem.taskId ||
        checkpoint.workItemId !== result.workItemId ||
        checkpoint.codeRevision !== work.codeRevision ||
        checkpoint.contextDigest !== work.contextDigest
      ) {
        throw new Error(`Checkpoint does not match current work ${result.workItemId}`);
      }
    }
    if (result.worktree && result.worktree.baseRevision !== work.codeRevision) {
      throw new Error(`Checkpoint worktree does not match revision for ${result.workItemId}`);
    }
    results.set(result.workItemId, result);
  }
  return results;
}

function propagateBlockedResults(
  pending: Set<string>,
  results: Map<string, ScheduledWorkResult>,
  dependencies: ReadonlyMap<string, readonly string[]>,
): void {
  while (true) {
    const blocked = Array.from(pending)
      .sort()
      .find((workItemId) =>
        (dependencies.get(workItemId) ?? []).some(
          (dependency) => results.get(dependency)?.status === "failed",
        ),
      );
    if (!blocked) return;
    const failedDependency = (dependencies.get(blocked) ?? []).find(
      (dependency) => results.get(dependency)?.status === "failed",
    );
    pending.delete(blocked);
    results.set(blocked, {
      workItemId: blocked,
      status: "failed",
      attempts: 0,
      checkpoints: [],
      diagnostics: [`Dependency ${failedDependency ?? "unknown"} failed`],
    });
  }
}

function scheduleResult(
  capabilities: readonly RuntimeCapabilitySnapshot[],
  results: ReadonlyMap<string, ScheduledWorkResult>,
): WorkItemScheduleResult {
  return {
    capabilities,
    results: Array.from(results.values()).sort((left, right) =>
      left.workItemId.localeCompare(right.workItemId),
    ),
  };
}

function throwIfCancelled(signal?: AbortSignal): void {
  if (signal?.aborted) throw new RuntimeScheduleCancelledError();
}

export class InMemoryRuntimeLeaseStore implements RuntimeLeaseStore {
  private readonly leases = new Map<string, RuntimeLease>();

  async acquire(lease: RuntimeLease): Promise<boolean> {
    const active = await this.listActive(lease.acquiredAt);
    if (active.some((candidate) => scopesConflict(candidate, lease))) return false;
    this.leases.set(lease.leaseId, lease);
    return true;
  }

  async release(leaseId: string, releasedAt: string): Promise<void> {
    const lease = this.leases.get(leaseId);
    if (!lease || lease.releasedAt) return;
    this.leases.set(leaseId, { ...lease, releasedAt });
  }

  async listActive(now: string): Promise<readonly RuntimeLease[]> {
    return Array.from(this.leases.values())
      .filter((lease) => !lease.releasedAt && lease.expiresAt > now)
      .sort((left, right) => left.leaseId.localeCompare(right.leaseId));
  }
}

function eligibleProviders(
  work: ScheduledLogicWork,
  capabilities: ReadonlyMap<RuntimeProvider, RuntimeCapabilitySnapshot>,
  runtimes: ReadonlyMap<RuntimeProvider, AgentRuntimePort>,
): readonly RuntimeProvider[] {
  const requested = work.pinnedProvider ? [work.pinnedProvider] : work.eligibleProviders;
  return Array.from(new Set(requested)).filter((provider) => {
    const capability = capabilities.get(provider);
    return (
      runtimes.has(provider) &&
      capability?.installed === true &&
      capability.authenticated === true &&
      capability.billingPath === "subscription" &&
      capability.supports.structuredOutput &&
      capability.supports.workspaceSandbox
    );
  });
}

function effectiveDependencies(
  work: readonly ScheduledLogicWork[],
): ReadonlyMap<string, readonly string[]> {
  const ordered = [...work].sort((left, right) =>
    left.workItem.workItemId.localeCompare(right.workItem.workItemId),
  );
  const dependencies = new Map(
    ordered.map((item) => [item.workItem.workItemId, new Set(item.workItem.dependsOn)] as const),
  );
  for (let leftIndex = 0; leftIndex < ordered.length; leftIndex++) {
    const left = ordered[leftIndex];
    if (!left) continue;
    for (let rightIndex = leftIndex + 1; rightIndex < ordered.length; rightIndex++) {
      const right = ordered[rightIndex];
      if (!right || !workItemsConflict(left.workItem, right.workItem)) continue;
      dependencies.get(right.workItem.workItemId)?.add(left.workItem.workItemId);
    }
  }
  return new Map(
    Array.from(dependencies.entries()).map(([workItemId, values]) => [
      workItemId,
      Array.from(values).sort(),
    ]),
  );
}

function assertKnownDependencies(
  workById: ReadonlyMap<string, ScheduledLogicWork>,
  dependencies: ReadonlyMap<string, readonly string[]>,
): void {
  for (const [workItemId, values] of dependencies) {
    for (const dependency of values) {
      if (!workById.has(dependency)) {
        throw new Error(`Logic Work Item ${workItemId} depends on unknown ${dependency}`);
      }
    }
  }
}

function workItemsConflict(left: LogicWorkItem, right: LogicWorkItem): boolean {
  return (
    left.plannedSymbolIds.some((symbolId) => right.plannedSymbolIds.includes(symbolId)) ||
    left.allowedPaths.some((allowedPath) => right.allowedPaths.includes(allowedPath))
  );
}

function scopesConflict(left: RuntimeLease, right: RuntimeLease): boolean {
  return (
    left.workspacePath === right.workspacePath ||
    left.symbolIds.some((symbolId) => right.symbolIds.includes(symbolId)) ||
    left.paths.some((allowedPath) => right.paths.includes(allowedPath))
  );
}

export function deterministicWorktreeId(taskId: string, workItemId: string): string {
  return `runtime-worktree:${createHash("sha256")
    .update(JSON.stringify([taskId, workItemId]))
    .digest("hex")}`;
}
