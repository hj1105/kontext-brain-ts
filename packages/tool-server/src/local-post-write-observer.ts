import { createHash } from "node:crypto";
import path from "node:path";
import type {
  CurrentTaskContextState,
  PreparedTaskContextStore,
  TaskContextStateProvider,
} from "@kontext-brain/context";
import {
  type QuarantineAssessment,
  type QuarantineStore,
  assessObservedChange,
} from "@kontext-brain/orchestrator";
import type { LogicWorkItem } from "@kontext-brain/spec";
import { FileWriteAuthorizationEventStore } from "./file-write-authorization-event-store.js";
import type { WriteAuthorizationBindingStore } from "./task-workflow-tools.js";
import { extractWritePaths } from "./task-workflow-tools.js";
import { captureWorkspaceSnapshot, changedPathsBetween } from "./workspace-change-observer.js";

export interface ObservePostWriteInput {
  readonly cwd: string;
  readonly toolName: string;
  readonly toolUseId?: string;
  readonly toolInput?: unknown;
  readonly observedAt: string;
}

export interface PostWriteObservation {
  readonly changed: boolean;
  readonly codeRevision: string;
  readonly changedPaths: readonly string[];
  readonly preWriteAuthorizationObserved: boolean;
  readonly assessment: QuarantineAssessment;
}

export class LocalPostWriteObserver {
  constructor(
    private readonly currentState: TaskContextStateProvider,
    private readonly preparedTasks: PreparedTaskContextStore,
    private readonly bindings: WriteAuthorizationBindingStore,
    private readonly authorizationEvents: FileWriteAuthorizationEventStore,
    private readonly quarantine: QuarantineStore,
  ) {}

  async observe(input: ObservePostWriteInput): Promise<PostWriteObservation> {
    const workspacePath = path.resolve(input.cwd);
    const claimedPaths = claimedToolPaths(workspacePath, input.toolName, input.toolInput);
    const binding = await this.bindings.get(workspacePath);
    const after = await captureWorkspaceSnapshot(
      workspacePath,
      binding?.allowedPaths ?? claimedPaths,
    );
    const changedPaths = binding
      ? changedPathsBetween(binding.baseline, after)
      : claimedPaths.map((filePath) => canonicalRelativePath(workspacePath, filePath));
    const authorization = input.toolUseId
      ? await this.authorizationEvents.consume(input.toolUseId, input.observedAt)
      : binding && changedPaths.length > 0
        ? await this.authorizationEvents.consumeForWorkspace(
            workspacePath,
            binding.baseline.revision,
            input.observedAt,
          )
        : undefined;
    const preWriteAuthorizationObserved =
      binding !== undefined &&
      authorization !== undefined &&
      authorization.workspacePath === workspacePath &&
      authorization.taskId === binding.request.taskId &&
      authorization.workItemId === binding.request.logic.workItemId &&
      authorization.receiptId === binding.receipt.receiptId &&
      authorization.contextDigest === binding.receipt.contextDigest &&
      authorization.baselineRevision === binding.baseline.revision &&
      changedPaths.every((changedPath) => authorization.authorizedPaths.includes(changedPath));

    if (binding && changedPaths.length === 0) {
      await this.bindings.put(workspacePath, { ...binding, baseline: after });
      return {
        changed: false,
        codeRevision: after.revision,
        changedPaths,
        preWriteAuthorizationObserved,
        assessment: { quarantined: false },
      };
    }

    const prepared = binding ? await this.preparedTasks.get(binding.request.taskId) : undefined;
    const workItem = binding ? await this.workItem(binding.request.taskId, binding) : undefined;
    const assessment = assessObservedChange({
      observed: {
        codeRevision: after.revision,
        contextDigest: binding?.receipt.contextDigest,
        paths: changedPaths,
        symbolIds: [],
        observedAt: input.observedAt,
        preWriteAuthorizationObserved,
      },
      workItem,
      snapshot: prepared?.snapshot,
      receipt: binding?.receipt,
      authorizedSymbolIds: binding?.request.logic.plannedSymbolIds,
    });
    if (assessment.record) await this.quarantine.put(assessment.record);
    if (binding) await this.bindings.put(workspacePath, { ...binding, baseline: after });
    return {
      changed: changedPaths.length > 0,
      codeRevision: after.revision,
      changedPaths,
      preWriteAuthorizationObserved,
      assessment,
    };
  }

  private async workItem(
    taskId: string,
    binding: NonNullable<Awaited<ReturnType<WriteAuthorizationBindingStore["get"]>>>,
  ): Promise<LogicWorkItem | undefined> {
    let current: CurrentTaskContextState;
    try {
      current = await this.currentState.getCurrent(taskId);
    } catch {
      return undefined;
    }
    const plan = current.logicPlans.find(
      (candidate) => candidate.workItemId === binding.request.logic.workItemId,
    );
    if (!plan) return undefined;
    return {
      workItemId: plan.workItemId,
      taskId,
      plannedSymbolIds: plan.plannedSymbolIds,
      dependsOn: plan.dependsOn ?? [],
      allowedPaths: plan.allowedPaths,
      requiredVerifiers: plan.requiredVerifiers ?? [],
      capabilityId:
        plan.capabilityId ??
        `capability:${createHash("sha256")
          .update(
            JSON.stringify([taskId, plan.workItemId, plan.plannedSymbolIds, plan.allowedPaths]),
          )
          .digest("hex")}`,
    };
  }
}

function claimedToolPaths(
  workspacePath: string,
  toolName: string,
  toolInput: unknown,
): readonly string[] {
  if (!isWriteToolName(toolName) || !isRecord(toolInput)) {
    return [];
  }
  return extractWritePaths(toolName, toolInput).map((filePath) =>
    path.resolve(workspacePath, filePath),
  );
}

function isWriteToolName(value: string): value is "apply_patch" | "Write" | "Edit" {
  return value === "apply_patch" || value === "Write" || value === "Edit";
}

function canonicalRelativePath(workspacePath: string, filePath: string): string {
  return path.relative(workspacePath, filePath).replaceAll("\\", "/").replace(/^\.\//, "");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
