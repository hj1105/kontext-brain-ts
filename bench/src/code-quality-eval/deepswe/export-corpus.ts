import type { CurrentTaskContextState, PreparedTaskContext } from "@kontext-brain/context";
import {
  type NormativeRevision,
  isTaskContextSnapshotValid,
  normativeRevisionContentDigest,
} from "@kontext-brain/spec";
import type {
  DeepSweContextCorpus,
  DeepSweEvidenceSnapshot,
  DeepSweNormativeRecord,
} from "./contracts.js";
import { sha256 } from "./corpus.js";

export interface DeepSweCorpusExportInput {
  readonly taskId: string;
  readonly organizationId: string;
  readonly runtimeProvider: string;
  readonly generatorRevision: string;
  readonly prepared: PreparedTaskContext;
  readonly current: CurrentTaskContextState;
  readonly allowEmpty?: boolean;
}

/**
 * Freezes a prepared Task Context Snapshot without re-extracting or inventing
 * benchmark context. Every exported byte comes from the sidecar's current,
 * provenance-bearing state and must still agree with the prepared snapshot.
 */
export function exportDeepSweCorpus(input: DeepSweCorpusExportInput): DeepSweContextCorpus {
  requireNonEmpty(input.taskId, "Task ID");
  requireNonEmpty(input.organizationId, "Organization ID");
  requireNonEmpty(input.runtimeProvider, "runtime provider");
  requireNonEmpty(input.generatorRevision, "generator revision");
  const snapshot = input.prepared.snapshot;
  if (
    input.prepared.contract.taskId !== input.taskId ||
    snapshot.taskId !== input.taskId ||
    !isTaskContextSnapshotValid(snapshot)
  ) {
    throw new Error("Prepared Task Context Snapshot is invalid for this benchmark task");
  }
  if (
    input.current.codeRevision !== snapshot.baseCodeRevision ||
    input.current.sourceFreshnessDigest !== snapshot.sourceFreshnessDigest ||
    !sameScopes(input.current.effectiveScopes, snapshot.effectiveScopes)
  ) {
    throw new Error(
      "Current Kontext state is stale relative to the prepared Task Context Snapshot",
    );
  }
  if (input.current.conflicts.length > 0) {
    throw new Error("Cannot export a benchmark corpus with normative conflicts");
  }

  const selected = selectSnapshotRevisions(input);
  const evidence = selectSnapshotEvidence(
    input,
    selected.map((entry) => entry.revision),
  );
  if (!input.allowEmpty && (selected.length === 0 || evidence.length === 0)) {
    throw new Error(
      "A scored DeepSWE corpus requires at least one Normative Revision and its Evidence",
    );
  }
  const normativeRecords = selected.map(({ revision, currentRevision }) => ({
    revision,
    ...selectorsForRevision(input.current, currentRevision),
  }));
  return {
    schemaVersion: 1,
    taskId: input.taskId,
    organizationId: input.organizationId,
    runtimeProvider: input.runtimeProvider,
    baseCodeRevision: snapshot.baseCodeRevision,
    contextDigest: snapshot.contextDigest,
    sourceFreshnessDigest: snapshot.sourceFreshnessDigest,
    snapshotAt: snapshot.createdAt,
    generator: { name: "kontext-brain", revision: input.generatorRevision },
    evidence,
    normativeRecords,
  };
}

function selectSnapshotRevisions(input: DeepSweCorpusExportInput): readonly {
  readonly revision: NormativeRevision;
  readonly currentRevision: NormativeRevision;
}[] {
  const catalog = new Map(
    input.current.normativeRevisionCatalog.map((revision) => [revisionKey(revision), revision]),
  );
  const current = input.current.normativeRecords.filter(
    (record) =>
      record.activation.state === "accepted" || record.activation.state === "accepted_local",
  );
  const selected = input.prepared.snapshot.normativeRevisions.map((reference) => {
    const revision = catalog.get(revisionKey(reference));
    if (!revision) {
      throw new Error(`Snapshot Normative Revision is unavailable: ${reference.revisionId}`);
    }
    if (revision.organizationId !== input.organizationId) {
      throw new Error(`Normative Revision belongs to another Organization: ${revision.revisionId}`);
    }
    const currentRecord = current.find(
      (record) =>
        record.revision.kind === revision.kind &&
        record.revision.recordId === revision.recordId &&
        normativeRevisionContentDigest(record.revision) ===
          normativeRevisionContentDigest(revision),
    );
    if (!currentRecord) {
      throw new Error(`Snapshot Normative Revision is stale: ${revision.revisionId}`);
    }
    if (!revision.egress.allowedRuntimeProviders.includes(input.runtimeProvider)) {
      throw new Error(`Normative Revision is not available to ${input.runtimeProvider}`);
    }
    return { revision, currentRevision: currentRecord.revision };
  });
  for (const record of current) {
    const represented = selected.some(
      ({ revision }) =>
        revision.kind === record.revision.kind &&
        revision.recordId === record.revision.recordId &&
        normativeRevisionContentDigest(revision) ===
          normativeRevisionContentDigest(record.revision),
    );
    if (!represented) {
      throw new Error(
        `Current Normative Revision is absent from the snapshot: ${record.revision.revisionId}`,
      );
    }
  }
  return selected.sort((left, right) =>
    revisionKey(left.revision).localeCompare(revisionKey(right.revision)),
  );
}

function selectSnapshotEvidence(
  input: DeepSweCorpusExportInput,
  revisions: readonly NormativeRevision[],
): readonly DeepSweEvidenceSnapshot[] {
  const required = new Set(input.prepared.snapshot.requiredEvidenceIds);
  for (const revision of revisions) {
    for (const reference of revision.evidence) {
      if (!required.has(reference.evidenceId)) {
        throw new Error(
          `Task Context Snapshot omits Evidence ${reference.evidenceId} for ${revision.revisionId}`,
        );
      }
    }
  }
  const evidenceById = new Map(
    input.current.evidence.map((evidence) => [evidence.evidenceId, evidence]),
  );
  return [...required]
    .sort((left, right) => left.localeCompare(right))
    .map((evidenceId) => {
      const evidence = evidenceById.get(evidenceId);
      if (!evidence || evidence.availability !== "current") {
        throw new Error(`Snapshot Evidence is unavailable or stale: ${evidenceId}`);
      }
      if (!evidence.allowedRuntimeProviders.includes(input.runtimeProvider)) {
        throw new Error(
          `Snapshot Evidence is not available to ${input.runtimeProvider}: ${evidenceId}`,
        );
      }
      const provenance = evidence.provenance;
      if (!provenance) {
        throw new Error(`Snapshot Evidence has no Resource/Chunk provenance: ${evidenceId}`);
      }
      const observedAt = Date.parse(provenance.observedAt);
      if (
        !Number.isFinite(observedAt) ||
        observedAt > Date.parse(input.prepared.snapshot.createdAt)
      ) {
        throw new Error(`Snapshot Evidence provenance is newer than the snapshot: ${evidenceId}`);
      }
      if (provenance.contentHash.replace(/^sha256:/, "") !== sha256(evidence.text)) {
        throw new Error(`Snapshot Evidence content hash does not match: ${evidenceId}`);
      }
      return {
        evidenceId,
        resourceId: provenance.resourceId,
        chunkId: provenance.chunkId,
        title: provenance.resourceTitle,
        text: evidence.text,
        ...(evidence.sourceSpan === undefined ? {} : { sourceSpan: evidence.sourceSpan }),
        source: provenance.source,
        observedAt: provenance.observedAt,
        contentSha256: provenance.contentHash,
        ontologyNodeIds: uniqueSorted(provenance.ontologyNodeIds),
        allowedRuntimeProviders: uniqueSorted(evidence.allowedRuntimeProviders),
      };
    });
}

function selectorsForRevision(
  current: CurrentTaskContextState,
  revision: NormativeRevision,
): Pick<DeepSweNormativeRecord, "symbolSelectors"> | Record<string, never> {
  const authoritativeLinks = (current.governanceLinks ?? []).filter(
    (link) =>
      link.origin !== "proposed" &&
      link.recordId === revision.recordId &&
      link.revisionId === revision.revisionId,
  );
  const plannedSymbols = new Map(
    current.logicPlans.flatMap((plan) =>
      (plan.plannedSymbols ?? []).map((symbol) => [symbol.plannedSymbolId, symbol] as const),
    ),
  );
  const selectors = authoritativeLinks.flatMap((link) => {
    const identity = plannedSymbols.get(link.plannedSymbolId)?.intendedIdentity;
    if (!identity?.relativePath && !identity?.qualifiedName) return [];
    return [
      {
        ...(identity.relativePath ? { relativePath: identity.relativePath } : {}),
        ...(identity.qualifiedName ? { qualifiedName: identity.qualifiedName } : {}),
      },
    ];
  });
  const unique = uniqueBy(selectors, (selector) =>
    JSON.stringify([selector.relativePath ?? "", selector.qualifiedName ?? ""]),
  ).sort(
    (left, right) =>
      (left.relativePath ?? "").localeCompare(right.relativePath ?? "") ||
      (left.qualifiedName ?? "").localeCompare(right.qualifiedName ?? ""),
  );
  return unique.length ? { symbolSelectors: unique } : {};
}

function sameScopes(
  left: CurrentTaskContextState["effectiveScopes"],
  right: CurrentTaskContextState["effectiveScopes"],
): boolean {
  return JSON.stringify(left.map(scopeKey).sort()) === JSON.stringify(right.map(scopeKey).sort());
}

function scopeKey(scope: CurrentTaskContextState["effectiveScopes"][number]): string {
  return JSON.stringify(scope);
}

function revisionKey(value: {
  readonly kind: string;
  readonly recordId: string;
  readonly revisionId: string;
}): string {
  return JSON.stringify([value.kind, value.recordId, value.revisionId]);
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort((left, right) => left.localeCompare(right));
}

function uniqueBy<T>(values: readonly T[], keyFor: (value: T) => string): T[] {
  const seen = new Set<string>();
  return values.filter((value) => {
    const key = keyFor(value);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function requireNonEmpty(value: string, label: string): void {
  if (!value.trim()) throw new Error(`${label} is required`);
}
