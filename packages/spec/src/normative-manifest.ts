import { createHash } from "node:crypto";
import type {
  GovernanceScope,
  NormativeActivation,
  NormativeManifest,
  NormativeManifestIssue,
  NormativeRevision,
} from "./domain.js";

export function validateNormativeManifest(
  manifest: NormativeManifest,
): readonly NormativeManifestIssue[] {
  const issues: NormativeManifestIssue[] = [];
  const revisions = new Map<string, NormativeRevision>();

  for (const revision of manifest.revisions) {
    const key = revisionKey(revision);
    if (revision.organizationId !== manifest.organizationId) {
      issues.push({
        code: "organization_mismatch",
        message: `Revision ${revision.revisionId} belongs to another Organization`,
        ref: revision.revisionId,
      });
    }
    if (revisions.has(key)) {
      issues.push({
        code: "duplicate_revision",
        message: `Duplicate normative revision: ${revision.revisionId}`,
        ref: revision.revisionId,
      });
    } else {
      revisions.set(key, revision);
    }
    if (
      revision.evidence.length === 0 ||
      revision.evidence.some((evidence) => !evidence.evidenceId.trim())
    ) {
      issues.push({
        code: "missing_evidence",
        message: `Normative revision ${revision.revisionId} requires Evidence`,
        ref: revision.revisionId,
      });
    }
  }

  for (const revision of manifest.revisions) {
    if (!revision.supersedesRevisionId) continue;
    if (revision.supersedesRevisionId === revision.revisionId) {
      issues.push({
        code: "invalid_supersedes",
        message: `Revision ${revision.revisionId} cannot supersede itself`,
        ref: revision.revisionId,
      });
      continue;
    }
    const superseded = revisions.get(
      revisionKey({ ...revision, revisionId: revision.supersedesRevisionId }),
    );
    if (!superseded) {
      issues.push({
        code: "missing_superseded_revision",
        message: `Revision ${revision.revisionId} supersedes an unknown revision`,
        ref: revision.revisionId,
      });
    }
  }

  const activationKeys = new Set<string>();
  for (const activation of manifest.activations) {
    const pointerKey = activationPointerKey(activation);
    if (activationKeys.has(pointerKey)) {
      issues.push({
        code: "duplicate_activation",
        message: `Multiple activation pointers exist for ${activation.recordId}`,
        ref: activation.recordId,
      });
    }
    activationKeys.add(pointerKey);
    if (activation.organizationId !== manifest.organizationId) {
      issues.push({
        code: "organization_mismatch",
        message: `Activation ${activation.recordId} belongs to another Organization`,
        ref: activation.recordId,
      });
    }
    const revision = revisions.get(revisionKey(activation));
    if (!revision) {
      issues.push({
        code: "missing_revision",
        message: `Activation ${activation.recordId} references an unknown revision`,
        ref: activation.revisionId,
      });
    } else if (scopeKey(revision.scope) !== scopeKey(activation.scope)) {
      issues.push({
        code: "activation_mismatch",
        message: `Activation ${activation.recordId} scope does not match its revision`,
        ref: activation.revisionId,
      });
    }
  }
  return issues;
}

export function encodeNormativeManifest(manifest: NormativeManifest): string {
  assertValidNormativeManifest(manifest);
  const normalized: NormativeManifest = {
    schemaVersion: 1,
    organizationId: manifest.organizationId,
    revisions: [...manifest.revisions].sort((left, right) =>
      revisionKey(left).localeCompare(revisionKey(right)),
    ),
    activations: [...manifest.activations].sort((left, right) =>
      activationPointerKey(left).localeCompare(activationPointerKey(right)),
    ),
  };
  return `${JSON.stringify(stableValue(normalized), null, 2)}\n`;
}

export function decodeNormativeManifest(serialized: string): NormativeManifest {
  const parsed: unknown = JSON.parse(serialized);
  assertNormativeManifestShape(parsed);
  assertValidNormativeManifest(parsed);
  return parsed;
}

export function normativeManifestDigest(manifest: NormativeManifest): string {
  return `sha256:${createHash("sha256").update(encodeNormativeManifest(manifest)).digest("hex")}`;
}

export function updateNormativeManifest(
  manifest: NormativeManifest,
  revision: NormativeRevision,
  activation: NormativeActivation,
): NormativeManifest {
  if (revision.organizationId !== manifest.organizationId) {
    throw new Error("Normative revision Organization does not match the manifest");
  }
  if (
    activation.organizationId !== manifest.organizationId ||
    activation.kind !== revision.kind ||
    activation.recordId !== revision.recordId ||
    activation.revisionId !== revision.revisionId ||
    scopeKey(activation.scope) !== scopeKey(revision.scope)
  ) {
    throw new Error("Normative activation does not match its revision");
  }

  const key = revisionKey(revision);
  const existing = manifest.revisions.find((candidate) => revisionKey(candidate) === key);
  if (existing && JSON.stringify(stableValue(existing)) !== JSON.stringify(stableValue(revision))) {
    throw new Error(`Immutable normative revision collision: ${revision.revisionId}`);
  }
  const next: NormativeManifest = {
    ...manifest,
    revisions: existing ? manifest.revisions : [...manifest.revisions, revision],
    activations: [
      ...manifest.activations.filter(
        (candidate) => activationPointerKey(candidate) !== activationPointerKey(activation),
      ),
      activation,
    ],
  };
  assertValidNormativeManifest(next);
  return next;
}

function assertValidNormativeManifest(manifest: NormativeManifest): void {
  const issues = validateNormativeManifest(manifest);
  if (issues.length > 0) {
    throw new Error(
      `Invalid normative manifest: ${issues.map((issue) => issue.message).join("; ")}`,
    );
  }
}

function revisionKey(
  revision: Pick<NormativeRevision, "kind" | "recordId" | "revisionId">,
): string {
  return JSON.stringify([revision.kind, revision.recordId, revision.revisionId]);
}

function activationPointerKey(
  activation: Pick<NormativeActivation, "kind" | "recordId" | "scope">,
): string {
  return JSON.stringify([activation.kind, activation.recordId, scopeKey(activation.scope)]);
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

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, nested]) => [key, stableValue(nested)]),
    );
  }
  return value;
}

function assertNormativeManifestShape(value: unknown): asserts value is NormativeManifest {
  if (!isRecord(value) || value.schemaVersion !== 1 || !nonEmptyString(value.organizationId)) {
    throw new Error("Invalid normative manifest header");
  }
  if (!Array.isArray(value.revisions) || !value.revisions.every(isNormativeRevision)) {
    throw new Error("Invalid normative manifest revisions");
  }
  if (!Array.isArray(value.activations) || !value.activations.every(isNormativeActivation)) {
    throw new Error("Invalid normative manifest activations");
  }
}

function isNormativeRevision(value: unknown): value is NormativeRevision {
  if (
    !isRecord(value) ||
    !["decision", "domain_term", "invariant"].includes(String(value.kind)) ||
    !nonEmptyString(value.organizationId) ||
    !nonEmptyString(value.recordId) ||
    !nonEmptyString(value.revisionId) ||
    !isGovernanceScope(value.scope) ||
    !Array.isArray(value.evidence) ||
    !value.evidence.every(
      (item) =>
        isRecord(item) &&
        nonEmptyString(item.evidenceId) &&
        (item.sourceSpan === undefined || typeof item.sourceSpan === "string"),
    ) ||
    !isRecord(value.egress) ||
    !["public", "internal", "confidential", "restricted"].includes(
      String(value.egress.dataClassification),
    ) ||
    !Array.isArray(value.egress.allowedRuntimeProviders) ||
    !value.egress.allowedRuntimeProviders.every(nonEmptyString) ||
    !nonEmptyString(value.authoredBy) ||
    !nonEmptyString(value.authoredAt) ||
    (value.supersedesRevisionId !== undefined && !nonEmptyString(value.supersedesRevisionId))
  ) {
    return false;
  }
  if (value.kind === "decision") return nonEmptyString(value.statement);
  if (value.kind === "domain_term") {
    return (
      nonEmptyString(value.term) &&
      nonEmptyString(value.definition) &&
      (value.avoid === undefined ||
        (Array.isArray(value.avoid) && value.avoid.every((item) => typeof item === "string")))
    );
  }
  return (
    nonEmptyString(value.statement) &&
    Array.isArray(value.verifiers) &&
    value.verifiers.every(
      (verifier) =>
        isRecord(verifier) &&
        ["test", "typecheck", "build", "lint", "query", "manual_review"].includes(
          String(verifier.kind),
        ) &&
        nonEmptyString(verifier.ref),
    )
  );
}

function isNormativeActivation(value: unknown): value is NormativeActivation {
  return (
    isRecord(value) &&
    nonEmptyString(value.organizationId) &&
    ["decision", "domain_term", "invariant"].includes(String(value.kind)) &&
    nonEmptyString(value.recordId) &&
    nonEmptyString(value.revisionId) &&
    isGovernanceScope(value.scope) &&
    ["accepted_local", "accepted", "retired"].includes(String(value.state)) &&
    nonEmptyString(value.acceptedBy) &&
    nonEmptyString(value.acceptedAt) &&
    (value.mergeCommit === undefined || nonEmptyString(value.mergeCommit))
  );
}

function isGovernanceScope(value: unknown): value is GovernanceScope {
  if (!isRecord(value)) return false;
  if (value.kind === "personal") return nonEmptyString(value.subjectId);
  if (value.kind === "workspace") return nonEmptyString(value.workspaceId);
  if (value.kind === "codebase") return nonEmptyString(value.codebaseId);
  if (value.kind === "organization") return nonEmptyString(value.organizationId);
  return false;
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
