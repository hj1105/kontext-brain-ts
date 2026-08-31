import { createHash } from "node:crypto";
import path from "node:path";
import type {
  ContextEvidenceItem,
  CurrentTaskContextState,
  LogicWorkPlan,
} from "@kontext-brain/context";
import {
  type GovernanceScope,
  type NormativeManifest,
  type NormativeRevision,
  encodeNormativeManifest,
  resolveNormativeLayers,
} from "@kontext-brain/spec";

export interface TaskContextStateAssemblyInput {
  readonly taskId: string;
  readonly organizationId: string;
  readonly codeRevision: string;
  readonly baseScopes?: readonly GovernanceScope[];
  readonly localManifest?: NormativeManifest;
  readonly managedManifest?: NormativeManifest;
  readonly evidence: readonly ContextEvidenceItem[];
  readonly logicPlans: readonly LogicWorkPlan[];
}

/**
 * Builds the sidecar-owned current state from collected provenance and the two
 * normative layers. Missing required Evidence stays explicit and fail-closed.
 */
export function assembleCurrentTaskContextState(
  input: TaskContextStateAssemblyInput,
): CurrentTaskContextState {
  requireNonEmpty(input.taskId, "Task ID");
  requireNonEmpty(input.organizationId, "Organization ID");
  requireNonEmpty(input.codeRevision, "code revision");
  const local = normalizedManifest(input.organizationId, input.localManifest, "local");
  const managed = normalizedManifest(input.organizationId, input.managedManifest, "managed");
  const resolution = resolveNormativeLayers(local, managed);
  const catalog = uniqueRevisions([...local.revisions, ...managed.revisions]);
  const evidence = normalizeEvidence(
    input.evidence,
    resolution.effective.map((record) => record.revision),
  );
  const logicPlans = normalizeLogicPlans(input.taskId, input.logicPlans);
  const effectiveScopes = uniqueScopes([
    ...(input.baseScopes ?? []),
    ...resolution.effective.map((record) => record.revision.scope),
  ]);

  return {
    codeRevision: input.codeRevision,
    sourceFreshnessDigest: freshnessDigest(resolution.effective, resolution.conflicts, evidence),
    effectiveScopes,
    normativeRecords: resolution.effective,
    normativeRevisionCatalog: catalog,
    conflicts: resolution.conflicts,
    evidence,
    logicPlans,
  };
}

function normalizedManifest(
  organizationId: string,
  manifest: NormativeManifest | undefined,
  layer: "local" | "managed",
): NormativeManifest {
  const value = manifest ?? {
    schemaVersion: 1 as const,
    organizationId,
    revisions: [],
    activations: [],
  };
  if (value.organizationId !== organizationId) {
    throw new Error(`${layer} normative manifest belongs to another Organization`);
  }
  return JSON.parse(encodeNormativeManifest(value)) as NormativeManifest;
}

function normalizeEvidence(
  provided: readonly ContextEvidenceItem[],
  revisions: readonly NormativeRevision[],
): readonly ContextEvidenceItem[] {
  const byId = new Map<string, ContextEvidenceItem>();
  for (const evidence of provided) {
    requireNonEmpty(evidence.evidenceId, "Evidence ID");
    const existing = byId.get(evidence.evidenceId);
    if (
      existing &&
      JSON.stringify(stableValue(existing)) !== JSON.stringify(stableValue(evidence))
    ) {
      throw new Error(`Conflicting Evidence payloads for "${evidence.evidenceId}"`);
    }
    byId.set(evidence.evidenceId, {
      ...evidence,
      allowedRuntimeProviders: uniqueSorted(evidence.allowedRuntimeProviders),
    });
  }

  const policies = new Map<string, string[]>();
  for (const revision of revisions) {
    for (const reference of revision.evidence) {
      const current = policies.get(reference.evidenceId);
      policies.set(
        reference.evidenceId,
        current
          ? intersection(current, revision.egress.allowedRuntimeProviders)
          : uniqueSorted(revision.egress.allowedRuntimeProviders),
      );
    }
  }

  for (const [evidenceId, policyProviders] of policies) {
    const current = byId.get(evidenceId);
    if (!current) {
      byId.set(evidenceId, {
        evidenceId,
        text: "",
        availability: "unavailable",
        allowedRuntimeProviders: policyProviders,
      });
      continue;
    }
    byId.set(evidenceId, {
      ...current,
      allowedRuntimeProviders: intersection(current.allowedRuntimeProviders, policyProviders),
    });
  }

  return Array.from(byId.values()).sort((left, right) =>
    left.evidenceId.localeCompare(right.evidenceId),
  );
}

function normalizeLogicPlans(
  taskId: string,
  plans: readonly LogicWorkPlan[],
): readonly LogicWorkPlan[] {
  const keys = new Set<string>();
  return plans
    .map((plan) => {
      requireNonEmpty(plan.workItemId, "Logic Work Item ID");
      const plannedSymbolIds = uniqueSorted(plan.plannedSymbolIds);
      const allowedPaths = uniqueSorted(plan.allowedPaths.map(normalizeAllowedPath));
      if (plannedSymbolIds.length === 0 || allowedPaths.length === 0) {
        throw new Error(`Logic Work Item "${plan.workItemId}" requires symbols and paths`);
      }
      const plannedSymbols = plan.plannedSymbols
        ? [...plan.plannedSymbols]
            .map((record) => {
              if (record.taskId !== taskId) {
                throw new Error(
                  `Planned Symbol "${record.plannedSymbolId}" belongs to another Task`,
                );
              }
              requireNonEmpty(record.plannedSymbolId, "Planned Symbol ID");
              requireNonEmpty(record.responsibility, "Planned Symbol responsibility");
              return {
                ...record,
                intendedIdentity: {
                  ...record.intendedIdentity,
                  relativePath: record.intendedIdentity.relativePath
                    ? normalizeAllowedPath(record.intendedIdentity.relativePath)
                    : undefined,
                },
              };
            })
            .sort((left, right) => left.plannedSymbolId.localeCompare(right.plannedSymbolId))
        : undefined;
      if (
        plannedSymbols &&
        JSON.stringify(plannedSymbols.map((record) => record.plannedSymbolId)) !==
          JSON.stringify(plannedSymbolIds)
      ) {
        throw new Error(
          `Logic Work Item "${plan.workItemId}" must describe every Planned Symbol ID exactly once`,
        );
      }
      const key = plan.workItemId;
      if (keys.has(key)) throw new Error(`Duplicate Logic Work Item plan: ${plan.workItemId}`);
      keys.add(key);
      return {
        workItemId: plan.workItemId,
        plannedSymbolIds,
        plannedSymbols,
        allowedPaths,
        dependsOn: uniqueSorted(plan.dependsOn ?? []),
        requiredVerifiers: [...(plan.requiredVerifiers ?? [])].sort(
          (left, right) => left.kind.localeCompare(right.kind) || left.ref.localeCompare(right.ref),
        ),
        capabilityId: plan.capabilityId,
      };
    })
    .sort((left, right) => left.workItemId.localeCompare(right.workItemId));
}

function normalizeAllowedPath(value: string): string {
  requireNonEmpty(value, "allowed path");
  if (path.isAbsolute(value) || path.win32.isAbsolute(value)) {
    throw new Error(`Allowed path must be workspace-relative: ${value}`);
  }
  const normalized = value.replaceAll("\\", "/");
  const segments = normalized.split("/");
  if (segments.includes("..") || normalized === ".") {
    throw new Error(`Allowed path must stay inside the workspace: ${value}`);
  }
  return path.posix.normalize(normalized).replace(/^\.\//, "");
}

function freshnessDigest(
  effective: CurrentTaskContextState["normativeRecords"],
  conflicts: CurrentTaskContextState["conflicts"],
  evidence: readonly ContextEvidenceItem[],
): string {
  const value = {
    normative: effective,
    conflicts,
    evidence: evidence.map((item) => ({
      evidenceId: item.evidenceId,
      availability: item.availability,
      sourceSpan: item.sourceSpan,
      allowedRuntimeProviders: uniqueSorted(item.allowedRuntimeProviders),
      textDigest: sha256(item.text),
    })),
  };
  return `sha256:${sha256(JSON.stringify(stableValue(value)))}`;
}

function uniqueRevisions(revisions: readonly NormativeRevision[]): readonly NormativeRevision[] {
  const byKey = new Map<string, NormativeRevision>();
  for (const revision of revisions) {
    const key = JSON.stringify([revision.kind, revision.recordId, revision.revisionId]);
    const existing = byKey.get(key);
    if (
      existing &&
      JSON.stringify(stableValue(existing)) !== JSON.stringify(stableValue(revision))
    ) {
      throw new Error(`Immutable normative revision collision: ${revision.revisionId}`);
    }
    byKey.set(key, revision);
  }
  return Array.from(byKey.values()).sort(
    (left, right) =>
      left.kind.localeCompare(right.kind) ||
      left.recordId.localeCompare(right.recordId) ||
      left.revisionId.localeCompare(right.revisionId),
  );
}

function uniqueScopes(scopes: readonly GovernanceScope[]): readonly GovernanceScope[] {
  const byKey = new Map(scopes.map((scope) => [scopeKey(scope), scope] as const));
  return Array.from(byKey.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, scope]) => scope);
}

function scopeKey(scope: GovernanceScope): string {
  switch (scope.kind) {
    case "personal":
      return JSON.stringify([scope.kind, scope.subjectId]);
    case "workspace":
      return JSON.stringify([scope.kind, scope.workspaceId]);
    case "codebase":
      return JSON.stringify([scope.kind, scope.codebaseId]);
    case "organization":
      return JSON.stringify([scope.kind, scope.organizationId]);
  }
}

function intersection(left: readonly string[], right: readonly string[]): string[] {
  const rightSet = new Set(right);
  return uniqueSorted(left.filter((value) => rightSet.has(value)));
}

function uniqueSorted(values: readonly string[]): string[] {
  return Array.from(new Set(values.filter((value) => value.trim()))).sort((left, right) =>
    left.localeCompare(right),
  );
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function requireNonEmpty(value: string, label: string): void {
  if (!value.trim()) throw new Error(`${label} is required`);
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, nested]) => [key, stableValue(nested)]),
    );
  }
  return value;
}
