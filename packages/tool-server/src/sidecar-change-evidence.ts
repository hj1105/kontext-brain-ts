import path from "node:path";
import {
  type PlannedSymbolBinding,
  type PlannedSymbolBindingIssue,
  type PlannedSymbolRecord,
  resolvePlannedSymbols,
} from "@kontext-brain/code";
import type { ObservedPatch } from "@kontext-brain/orchestrator";
import type { ContextReceipt, LogicWorkItem } from "@kontext-brain/spec";
import type { WriteAuthorizationBindingStore } from "./task-workflow-tools.js";
import { captureWorkspaceSnapshot, observeWorkspacePatch } from "./workspace-change-observer.js";
import {
  type WorkspaceCodeSymbolState,
  captureWorkspaceCodeSymbols,
  changedWorkspaceCodeSymbolIds,
} from "./workspace-code-symbol-observer.js";

export interface SidecarChangeEvidenceRequest {
  readonly workspacePath: string;
  readonly taskId: string;
  readonly workItem: LogicWorkItem;
  readonly plannedSymbols?: readonly PlannedSymbolRecord[];
}

export interface SidecarChangeEvidence {
  readonly currentCodeRevision: string;
  readonly observedPatch: ObservedPatch;
  readonly receipts: readonly ContextReceipt[];
  readonly plannedSymbolBindings: readonly PlannedSymbolBinding[];
  readonly plannedSymbolIssues: readonly PlannedSymbolBindingIssue[];
  readonly unauthorizedChangedSymbolIds: readonly string[];
}

export interface SidecarChangeEvidenceProvider {
  observe(request: SidecarChangeEvidenceRequest): Promise<SidecarChangeEvidence>;
}

export class BoundWorkspaceChangeEvidenceProvider implements SidecarChangeEvidenceProvider {
  constructor(private readonly bindings: WriteAuthorizationBindingStore) {}

  async observe(request: SidecarChangeEvidenceRequest): Promise<SidecarChangeEvidence> {
    const workspacePath = path.resolve(request.workspacePath);
    const binding = await this.bindings.get(workspacePath);
    if (!binding) throw new Error("No sidecar-owned write binding exists for this workspace");
    if (
      binding.request.taskId !== request.taskId ||
      binding.request.logic.workItemId !== request.workItem.workItemId
    ) {
      throw new Error("Workspace write binding belongs to another Task or Logic Work Item");
    }
    if (!sameStrings(binding.request.logic.plannedSymbolIds, request.workItem.plannedSymbolIds)) {
      throw new Error("Workspace write binding Planned Symbols no longer match sidecar state");
    }
    if (binding.initialBaseline.revision !== binding.symbolBaseline.workspaceRevision) {
      throw new Error("Workspace file and Code Symbol baselines do not describe one revision");
    }

    const current = await captureWorkspaceSnapshot(workspacePath, binding.allowedPaths);
    const currentSymbols = await captureWorkspaceCodeSymbols(
      workspacePath,
      binding.allowedPaths,
      current,
    );
    const confirmedCurrent = await captureWorkspaceSnapshot(workspacePath, binding.allowedPaths);
    if (currentSymbols.workspaceRevision !== confirmedCurrent.revision) {
      throw new Error("Workspace changed while sidecar completion evidence was captured");
    }
    const patch = observeWorkspacePatch(binding.initialBaseline, confirmedCurrent);
    const changedSymbolIds = changedWorkspaceCodeSymbolIds(binding.symbolBaseline, currentSymbols);
    const planned =
      request.plannedSymbols ??
      request.workItem.plannedSymbolIds.map((plannedSymbolId) => ({
        plannedSymbolId,
        taskId: request.taskId,
        intendedIdentity: {},
        responsibility: "Legacy exact Code Symbol binding",
      }));
    const resolution = resolvePlannedSymbols(
      planned,
      mergeSymbols(binding.symbolBaseline.symbols, currentSymbols.symbols),
    );
    const authorized = new Set(resolution.bindings.map((item) => item.symbolId));

    return {
      currentCodeRevision: confirmedCurrent.revision,
      observedPatch: {
        patchDigest: patch.patchDigest,
        changedPaths: patch.changedPaths,
        changedSymbolIds,
      },
      receipts: [binding.receipt],
      plannedSymbolBindings: resolution.bindings,
      plannedSymbolIssues: resolution.issues,
      unauthorizedChangedSymbolIds: changedSymbolIds.filter(
        (symbolId) => !authorized.has(symbolId),
      ),
    };
  }
}

function mergeSymbols(
  before: readonly WorkspaceCodeSymbolState[],
  after: readonly WorkspaceCodeSymbolState[],
): readonly WorkspaceCodeSymbolState[] {
  const symbols = new Map(before.map((symbol) => [symbol.symbolId, symbol] as const));
  for (const symbol of after) symbols.set(symbol.symbolId, symbol);
  return Array.from(symbols.values()).sort((left, right) =>
    left.symbolId.localeCompare(right.symbolId),
  );
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  const normalize = (values: readonly string[]) => Array.from(new Set(values)).sort();
  return JSON.stringify(normalize(left)) === JSON.stringify(normalize(right));
}
