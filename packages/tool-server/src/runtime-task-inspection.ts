import type { PreparedTaskContextStore, TaskContextStateProvider } from "@kontext-brain/context";
import { assessCurrentContext } from "./local-completion-operations.js";

export async function inspectRuntimeTask(
  taskId: string,
  currentState: TaskContextStateProvider,
  preparedTasks: PreparedTaskContextStore,
) {
  const [current, prepared] = await Promise.all([
    currentState.getCurrent(taskId),
    preparedTasks.get(taskId),
  ]);
  if (prepared && (prepared.contract.taskId !== taskId || prepared.snapshot.taskId !== taskId)) {
    throw new Error("Prepared Kontext state belongs to a different task.");
  }
  const status = !prepared
    ? "unprepared"
    : prepared.snapshot.baseCodeRevision !== current.codeRevision
      ? "stale"
      : assessCurrentContext(prepared, current).status;
  return {
    taskId,
    status,
    contract: prepared?.contract ?? null,
    codeRevision: current.codeRevision,
    contextDigest: prepared?.snapshot.contextDigest ?? null,
    requiredEvidenceIds: prepared?.snapshot.requiredEvidenceIds ?? [],
    normativeRevisionCount: current.normativeRecords.length,
    conflictCount: current.conflicts.length,
    logic: current.logicPlans.map((plan) => ({
      workItemId: plan.workItemId,
      plannedSymbolIds: plan.plannedSymbolIds,
      allowedPaths: plan.allowedPaths,
      dependsOn: plan.dependsOn ?? [],
    })),
  };
}
