import { createHash } from "node:crypto";
import path from "node:path";
import type { PreparedTaskContextStore, TaskContextStateProvider } from "@kontext-brain/context";
import { FileRuntimeLeaseStore, GitRuntimeWorktreeManager } from "@kontext-brain/local";
import {
  type AgentRuntimePort,
  RuntimeDoctor,
  type RuntimeProvider,
  WorkItemScheduler,
} from "@kontext-brain/orchestrator";
import type { LogicWorkItem } from "@kontext-brain/spec";
import { z } from "zod";
import {
  FileRuntimeScheduleJobStore,
  RuntimeScheduleJobManager,
} from "./file-runtime-schedule-job-store.js";
import {
  type CancelScheduleRequest,
  type GetScheduleRequest,
  type ScheduleLogicRequest,
  cancelScheduleToolShape,
  getScheduleToolShape,
  inspectRuntimesToolShape,
  scheduleLogicRequestSchema,
} from "./runtime-schedule-contract.js";
import {
  type KontextTaskWorkflowOperations,
  KontextTaskWorkflowToolRouter,
  type WriteAuthorizationBindingStore,
} from "./task-workflow-tools.js";

export interface KontextRuntimeOperations {
  inspectRuntimes(): Promise<unknown>;
  scheduleLogic(request: ScheduleLogicRequest): Promise<unknown>;
  getSchedule(request: GetScheduleRequest): Promise<unknown>;
  cancelSchedule(request: CancelScheduleRequest): Promise<unknown>;
}

export class KontextRuntimeToolRouter {
  constructor(private readonly operations: KontextRuntimeOperations) {}

  async inspectRuntimes(input: unknown): Promise<unknown> {
    z.object(inspectRuntimesToolShape).strict().parse(input);
    return this.operations.inspectRuntimes();
  }

  async scheduleLogic(input: unknown): Promise<unknown> {
    return this.operations.scheduleLogic(scheduleLogicRequestSchema.parse(input));
  }

  async getSchedule(input: unknown): Promise<unknown> {
    return this.operations.getSchedule(z.object(getScheduleToolShape).strict().parse(input));
  }

  async cancelSchedule(input: unknown): Promise<unknown> {
    return this.operations.cancelSchedule(z.object(cancelScheduleToolShape).strict().parse(input));
  }
}

export class LocalKontextRuntimeOperations implements KontextRuntimeOperations {
  private readonly leases: FileRuntimeLeaseStore;
  private readonly scheduleJobs: RuntimeScheduleJobManager;

  constructor(
    private readonly currentState: TaskContextStateProvider,
    private readonly preparedTasks: PreparedTaskContextStore,
    private readonly workflow: KontextTaskWorkflowOperations,
    private readonly bindings: WriteAuthorizationBindingStore,
    private readonly dataDirectory: string,
    private readonly runtimes: readonly AgentRuntimePort[],
    private readonly now: () => Date = () => new Date(),
  ) {
    this.leases = new FileRuntimeLeaseStore(dataDirectory);
    this.scheduleJobs = new RuntimeScheduleJobManager(
      new FileRuntimeScheduleJobStore(dataDirectory),
      now,
    );
  }

  async inspectRuntimes(): Promise<unknown> {
    return new RuntimeDoctor().inspect(this.runtimes);
  }

  async scheduleLogic(request: ScheduleLogicRequest): Promise<unknown> {
    const prepared = await this.preparedTasks.get(request.taskId);
    if (!prepared) throw new Error(`Task "${request.taskId}" has no prepared context`);
    const current = await this.currentState.getCurrent(request.taskId);
    const repositoryPath = path.resolve(request.repositoryPath);
    const worktreeRoot = path.join(
      this.dataDirectory,
      "runtime-worktrees",
      createHash("sha256").update(repositoryPath).digest("hex"),
    );
    const manager = new GitRuntimeWorktreeManager(repositoryPath, worktreeRoot);
    const contextRouter = new KontextTaskWorkflowToolRouter(this.workflow, this.now, this.bindings);
    const allowedByEvidence = providersAllowedByEvidence(
      current.evidence,
      prepared.snapshot.requiredEvidenceIds,
    );
    const work = request.work.map((requested) => {
      const plan = current.logicPlans.find(
        (candidate) => candidate.workItemId === requested.workItemId,
      );
      if (!plan) {
        throw new Error(`Logic Work Item "${requested.workItemId}" is not sidecar-planned`);
      }
      const eligibleProviders = requested.eligibleProviders.filter((provider) =>
        allowedByEvidence.has(provider),
      );
      if (requested.pinnedProvider && !eligibleProviders.includes(requested.pinnedProvider)) {
        throw new Error(
          `Pinned provider ${requested.pinnedProvider} is not eligible for ${requested.workItemId}`,
        );
      }
      return {
        workItem: logicWorkItem(request.taskId, plan),
        prompt: requested.prompt,
        codeRevision: current.codeRevision,
        contextDigest: prepared.snapshot.contextDigest,
        eligibleProviders,
        pinnedProvider: requested.pinnedProvider,
        totalTokenBudget: requested.totalTokenBudget,
        optionalEvidenceTokenBudget: requested.optionalEvidenceTokenBudget,
        receiptTtlSeconds: requested.receiptTtlSeconds,
      };
    });
    const scheduler = new WorkItemScheduler(this.runtimes, manager, this.leases, this.now, {
      prepare: async (input) => {
        const compiled = await contextRouter.beginLogic({
          taskId: input.taskId,
          workspacePath: input.worktree.workspacePath,
          logic: {
            workItemId: input.workItem.workItemId,
            plannedSymbolIds: input.workItem.plannedSymbolIds,
          },
          runtimeProvider: input.provider,
          receiptTtlSeconds: input.receiptTtlSeconds,
          totalTokenBudget: input.totalTokenBudget,
          optionalEvidenceTokenBudget: input.optionalEvidenceTokenBudget,
        });
        if (!compiled.editingAllowed || !compiled.receipt) {
          throw new Error(
            `Context is ${compiled.status} for ${input.workItem.workItemId} on ${input.provider}`,
          );
        }
      },
    });
    return this.scheduleJobs.enqueue(
      request,
      current.codeRevision,
      prepared.snapshot.contextDigest,
      (signal) =>
        scheduler.run({
          taskId: request.taskId,
          work,
          maxConcurrency: request.maxConcurrency,
          maxRetries: request.maxRetries,
          signal,
        }),
    );
  }

  async getSchedule(request: GetScheduleRequest): Promise<unknown> {
    return this.scheduleJobs.get(request.jobId);
  }

  async cancelSchedule(request: CancelScheduleRequest): Promise<unknown> {
    return this.scheduleJobs.cancel(request.jobId);
  }
}

function logicWorkItem(
  taskId: string,
  plan: {
    readonly workItemId: string;
    readonly plannedSymbolIds: readonly string[];
    readonly allowedPaths: readonly string[];
    readonly dependsOn?: readonly string[];
    readonly requiredVerifiers?: LogicWorkItem["requiredVerifiers"];
    readonly capabilityId?: string;
  },
): LogicWorkItem {
  return {
    workItemId: plan.workItemId,
    taskId,
    plannedSymbolIds: plan.plannedSymbolIds,
    allowedPaths: plan.allowedPaths,
    dependsOn: plan.dependsOn ?? [],
    requiredVerifiers: plan.requiredVerifiers ?? [],
    capabilityId:
      plan.capabilityId ??
      `capability:${createHash("sha256")
        .update(JSON.stringify([taskId, plan.workItemId, plan.plannedSymbolIds, plan.allowedPaths]))
        .digest("hex")}`,
  };
}

function providersAllowedByEvidence(
  evidence: readonly {
    readonly evidenceId: string;
    readonly allowedRuntimeProviders: readonly string[];
  }[],
  requiredEvidenceIds: readonly string[],
): ReadonlySet<RuntimeProvider> {
  const allowed = new Set<RuntimeProvider>(["codex", "claude"]);
  const byId = new Map(evidence.map((item) => [item.evidenceId, item] as const));
  for (const evidenceId of requiredEvidenceIds) {
    const item = byId.get(evidenceId);
    if (!item) return new Set();
    for (const provider of [...allowed]) {
      if (!item.allowedRuntimeProviders.includes(provider)) allowed.delete(provider);
    }
  }
  return allowed;
}
