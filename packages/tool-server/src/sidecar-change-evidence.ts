import path from "node:path";
import {
  type CodeSymbolIdentity,
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
    const resolution = resolvePlannedSymbols(planned, currentSymbols.symbols);
    const authorized = authorizedSymbolIds(
      resolution.bindings,
      binding.symbolBaseline.symbols,
      currentSymbols.symbols,
    );

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

/**
 * A Planned Symbol binds only against synchronized current code. Resolving it
 * against the baseline as well made the pre-edit and post-edit revisions of one
 * function two candidates sharing the same intended identity, so implementing
 * the target symbol reported identity_ambiguous and blocked its own
 * verification.
 *
 * The baseline still authorizes the predecessor that a bound symbol replaced. A
 * behavior change gives that predecessor a different symbol ID and
 * changedWorkspaceCodeSymbolIds reports both IDs, so the superseded one stays in
 * scope without ever authorizing an unrelated symbol.
 */
function authorizedSymbolIds(
  bindings: readonly PlannedSymbolBinding[],
  baseline: readonly WorkspaceCodeSymbolState[],
  current: readonly WorkspaceCodeSymbolState[],
): ReadonlySet<string> {
  const currentById = new Map(current.map((symbol) => [symbol.symbolId, symbol] as const));
  const authorized = new Set(bindings.map((item) => item.symbolId));
  const boundIdentities = new Set(
    bindings
      .map((item) => currentById.get(item.symbolId))
      .filter((symbol): symbol is WorkspaceCodeSymbolState => symbol !== undefined)
      .map((symbol) => identityKey(symbol.identity)),
  );
  for (const symbol of baseline) {
    if (currentById.has(symbol.symbolId)) continue;
    if (boundIdentities.has(identityKey(symbol.identity))) authorized.add(symbol.symbolId);
  }
  return authorized;
}

function identityKey(identity: CodeSymbolIdentity): string {
  return JSON.stringify([
    identity.relativePath,
    identity.language,
    identity.kind,
    identity.qualifiedName,
  ]);
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  const normalize = (values: readonly string[]) => Array.from(new Set(values)).sort();
  return JSON.stringify(normalize(left)) === JSON.stringify(normalize(right));
}
