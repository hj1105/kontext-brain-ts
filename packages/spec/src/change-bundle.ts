import { createHash } from "node:crypto";
import type { ChangeBundle, NormativeRevisionRef } from "./domain.js";

export type ChangeBundleInput = Omit<ChangeBundle, "bundleId">;

export function createChangeBundle(input: ChangeBundleInput): ChangeBundle {
  const { bundleId: _ignoredBundleId, ...safeInput } = input as ChangeBundle;
  const normalized: ChangeBundleInput = {
    ...safeInput,
    changedSymbolIds: uniqueSorted(safeInput.changedSymbolIds),
    changedPaths: uniqueSorted(safeInput.changedPaths),
    contextReceiptIds: uniqueSorted(safeInput.contextReceiptIds),
    evidenceIds: uniqueSorted(safeInput.evidenceIds),
    normativeRevisions: normalizeRevisionRefs(safeInput.normativeRevisions),
    verificationRunIds: uniqueSorted(safeInput.verificationRunIds),
    proposals: uniqueSorted(safeInput.proposals),
    unresolved: uniqueSorted(safeInput.unresolved),
  };
  return Object.freeze({
    ...normalized,
    bundleId: `change-bundle:${sha256(stableJson(normalized))}`,
  });
}

export function isChangeBundleValid(bundle: ChangeBundle): boolean {
  const { bundleId: _bundleId, ...input } = bundle;
  return stableJson(bundle) === stableJson(createChangeBundle(input));
}

function normalizeRevisionRefs(
  revisions: readonly NormativeRevisionRef[],
): readonly NormativeRevisionRef[] {
  const byKey = new Map(revisions.map((revision) => [revisionKey(revision), revision] as const));
  return Array.from(byKey.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, revision]) => revision);
}

function revisionKey(revision: NormativeRevisionRef): string {
  return JSON.stringify([revision.kind, revision.recordId, revision.revisionId]);
}

function uniqueSorted(values: readonly string[]): readonly string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function stableJson(value: unknown): string {
  return JSON.stringify(stableValue(value));
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
