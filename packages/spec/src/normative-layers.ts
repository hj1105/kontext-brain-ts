import { createHash } from "node:crypto";
import type {
  EffectiveNormativeRecord,
  NormativeActivation,
  NormativeLayerConflict,
  NormativeLayerResolution,
  NormativeManifest,
  NormativeRevision,
} from "./domain.js";

export function resolveNormativeLayers(
  local: NormativeManifest,
  managed: NormativeManifest,
): NormativeLayerResolution {
  if (local.organizationId !== managed.organizationId) {
    throw new Error("Local and managed normative layers must belong to the same Organization");
  }
  const localActive = activeRecords(local, "local");
  const managedActive = activeRecords(managed, "managed");
  const effective: EffectiveNormativeRecord[] = [...managedActive];
  const conflicts: NormativeLayerConflict[] = [];
  const canonicalizedLocalRevisionIds: string[] = [];
  const localOnlyRevisionIds: string[] = [];

  for (const localRecord of localActive) {
    const candidates = managedActive.filter(
      (managedRecord) =>
        managedRecord.revision.kind === localRecord.revision.kind &&
        managedRecord.revision.recordId === localRecord.revision.recordId,
    );
    if (candidates.length === 0) {
      effective.push(localRecord);
      localOnlyRevisionIds.push(localRecord.revision.revisionId);
      continue;
    }
    const localDigest = normativeRevisionContentDigest(localRecord.revision);
    const changed = candidates.filter(
      (candidate) => normativeRevisionContentDigest(candidate.revision) !== localDigest,
    );
    if (changed.length === 0) {
      canonicalizedLocalRevisionIds.push(localRecord.revision.revisionId);
      continue;
    }
    effective.push(localRecord);
    conflicts.push({
      kind: localRecord.revision.kind,
      recordId: localRecord.revision.recordId,
      localRevisionId: localRecord.revision.revisionId,
      managedRevisionIds: uniqueSorted(changed.map((candidate) => candidate.revision.revisionId)),
    });
  }

  return {
    effective: effective.sort(compareEffectiveRecord),
    conflicts: conflicts.sort(
      (left, right) =>
        left.kind.localeCompare(right.kind) || left.recordId.localeCompare(right.recordId),
    ),
    canonicalizedLocalRevisionIds: uniqueSorted(canonicalizedLocalRevisionIds),
    localOnlyRevisionIds: uniqueSorted(localOnlyRevisionIds),
    contextStale: conflicts.length > 0,
  };
}

export function normativeRevisionContentDigest(revision: NormativeRevision): string {
  const semanticContent =
    revision.kind === "decision"
      ? { kind: revision.kind, statement: revision.statement }
      : revision.kind === "domain_term"
        ? {
            kind: revision.kind,
            term: revision.term,
            definition: revision.definition,
            avoid: uniqueSorted(revision.avoid ?? []),
          }
        : {
            kind: revision.kind,
            statement: revision.statement,
            verifiers: [...revision.verifiers]
              .map((verifier) => JSON.stringify([verifier.kind, verifier.ref]))
              .sort(),
          };
  const content = {
    semanticContent,
    egress: {
      dataClassification: revision.egress.dataClassification,
      allowedRuntimeProviders: uniqueSorted(revision.egress.allowedRuntimeProviders),
    },
  };
  return `sha256:${createHash("sha256").update(JSON.stringify(content)).digest("hex")}`;
}

function activeRecords(
  manifest: NormativeManifest,
  origin: EffectiveNormativeRecord["origin"],
): EffectiveNormativeRecord[] {
  const revisions = new Map(
    manifest.revisions.map((revision) => [revisionKey(revision), revision] as const),
  );
  return manifest.activations
    .filter((activation) =>
      origin === "local" ? activation.state === "accepted_local" : activation.state === "accepted",
    )
    .map((activation) => {
      const revision = revisions.get(revisionKey(activation));
      if (!revision) {
        throw new Error(
          `Normative activation ${activation.recordId} references an unknown revision`,
        );
      }
      return { origin, revision, activation };
    });
}

function revisionKey(
  value: Pick<NormativeRevision | NormativeActivation, "kind" | "recordId" | "revisionId">,
): string {
  return JSON.stringify([value.kind, value.recordId, value.revisionId]);
}

function compareEffectiveRecord(
  left: EffectiveNormativeRecord,
  right: EffectiveNormativeRecord,
): number {
  return (
    left.revision.kind.localeCompare(right.revision.kind) ||
    left.revision.recordId.localeCompare(right.revision.recordId) ||
    left.origin.localeCompare(right.origin) ||
    left.revision.revisionId.localeCompare(right.revision.revisionId)
  );
}

function uniqueSorted(values: readonly string[]): readonly string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}
