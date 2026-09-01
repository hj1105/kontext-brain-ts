import { createHash } from "node:crypto";
import { validateContextReceipt } from "@kontext-brain/context";
import type { QuarantineReason, QuarantineRecord } from "@kontext-brain/spec";
import type { AssessObservedChangeInput, QuarantineAssessment } from "./domain.js";

export type ActiveQuarantineRecordInput = Omit<
  QuarantineRecord,
  "quarantineId" | "status" | "releasedAt" | "releasedBy"
>;

export function assessObservedChange(input: AssessObservedChangeInput): QuarantineAssessment {
  const reasons = new Set<QuarantineReason>();
  const { observed, workItem, snapshot, receipt } = input;
  if (!observed.preWriteAuthorizationObserved) reasons.add("unobserved_write");

  if (!workItem || !snapshot || !receipt) {
    reasons.add("missing_capability");
  } else {
    const receiptIssues = validateContextReceipt({
      receipt,
      snapshot,
      logic: {
        workItemId: workItem.workItemId,
        plannedSymbolIds: workItem.plannedSymbolIds,
      },
      allowedPaths: workItem.allowedPaths,
      now: observed.observedAt,
    });
    if (receiptIssues.includes("expired") || receiptIssues.includes("invalid_time")) {
      reasons.add("expired_capability");
    }
    if (receiptIssues.some((issue) => issue !== "expired" && issue !== "invalid_time")) {
      reasons.add("context_mismatch");
    }
    if (observed.contextDigest !== undefined && observed.contextDigest !== receipt.contextDigest) {
      reasons.add("context_mismatch");
    }

    const allowedPaths = new Set(workItem.allowedPaths.map(canonicalPath));
    if (observed.paths.some((changedPath) => !allowedPaths.has(canonicalPath(changedPath)))) {
      reasons.add("path_out_of_scope");
    }
    const authorizedSymbols = new Set(input.authorizedSymbolIds ?? workItem.plannedSymbolIds);
    if (observed.symbolIds.some((symbolId) => !authorizedSymbols.has(symbolId))) {
      reasons.add("symbol_out_of_scope");
    }
  }

  if (reasons.size === 0) return { quarantined: false };
  const record = createQuarantineRecord({
    taskId: workItem?.taskId,
    workItemId: workItem?.workItemId,
    codeRevision: observed.codeRevision,
    contextDigest: observed.contextDigest,
    paths: uniqueSorted(observed.paths.map(canonicalPath)),
    symbolIds: uniqueSorted(observed.symbolIds),
    reasons: Array.from(reasons).sort(),
    observedAt: observed.observedAt,
  });
  return { quarantined: true, record };
}

export function createQuarantineRecord(input: ActiveQuarantineRecordInput): QuarantineRecord {
  const {
    quarantineId: _ignoredQuarantineId,
    status: _ignoredStatus,
    releasedAt: _ignoredReleasedAt,
    releasedBy: _ignoredReleasedBy,
    ...safeInput
  } = input as QuarantineRecord;
  const identity = {
    ...safeInput,
    paths: uniqueSorted(safeInput.paths.map(canonicalPath)),
    symbolIds: uniqueSorted(safeInput.symbolIds),
    reasons: Array.from(new Set(safeInput.reasons)).sort(),
    status: "active" as const,
  };
  return Object.freeze({
    ...identity,
    quarantineId: `quarantine:${sha256(stableJson(identity))}`,
  });
}

export function isQuarantineRecordValid(record: QuarantineRecord): boolean {
  const { quarantineId, releasedAt: _releasedAt, releasedBy: _releasedBy, ...identity } = record;
  return quarantineId === `quarantine:${sha256(stableJson({ ...identity, status: "active" }))}`;
}

function canonicalPath(value: string): string {
  const normalized = value.replaceAll("\\", "/");
  const absolute = normalized.startsWith("/");
  const segments: string[] = [];
  for (const segment of normalized.split("/")) {
    if (!segment || segment === ".") continue;
    if (segment === "..") {
      if (segments.length > 0 && segments.at(-1) !== "..") segments.pop();
      else if (!absolute) segments.push(segment);
    } else segments.push(segment);
  }
  return `${absolute ? "/" : ""}${segments.join("/")}`;
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
