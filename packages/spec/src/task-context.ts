import { createHash } from "node:crypto";
import type {
  GovernanceScope,
  NormativeRevisionRef,
  TaskContextDigestInput,
  TaskContextSnapshot,
} from "./domain.js";

export function computeTaskContextDigest(input: TaskContextDigestInput): string {
  const normalized = {
    taskId: input.taskId,
    baseCodeRevision: input.baseCodeRevision,
    effectiveScopes: uniqueSorted(input.effectiveScopes.map(scopeKey)),
    normativeRevisions: uniqueSorted(input.normativeRevisions.map(normativeRevisionKey)),
    requiredEvidenceIds: uniqueSorted(input.requiredEvidenceIds),
    sourceFreshnessDigest: input.sourceFreshnessDigest,
  };
  return `sha256:${createHash("sha256").update(JSON.stringify(normalized)).digest("hex")}`;
}

export function createTaskContextSnapshot(
  input: TaskContextDigestInput,
  createdAt: string,
): TaskContextSnapshot {
  return Object.freeze({
    ...input,
    effectiveScopes: Object.freeze([...input.effectiveScopes]),
    normativeRevisions: Object.freeze([...input.normativeRevisions]),
    requiredEvidenceIds: Object.freeze([...input.requiredEvidenceIds]),
    contextDigest: computeTaskContextDigest(input),
    createdAt,
  });
}

export function isTaskContextSnapshotValid(snapshot: TaskContextSnapshot): boolean {
  return computeTaskContextDigest(snapshot) === snapshot.contextDigest;
}

function scopeKey(scope: GovernanceScope): string {
  switch (scope.kind) {
    case "personal":
      return JSON.stringify(["personal", scope.subjectId]);
    case "workspace":
      return JSON.stringify(["workspace", scope.workspaceId]);
    case "codebase":
      return JSON.stringify(["codebase", scope.codebaseId]);
    case "organization":
      return JSON.stringify(["organization", scope.organizationId]);
  }
}

function normativeRevisionKey(revision: NormativeRevisionRef): string {
  return JSON.stringify([revision.kind, revision.recordId, revision.revisionId]);
}

function uniqueSorted(values: readonly string[]): readonly string[] {
  return Array.from(new Set(values)).sort((left, right) => {
    if (left < right) return -1;
    if (left > right) return 1;
    return 0;
  });
}
