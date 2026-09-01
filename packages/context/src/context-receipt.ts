import { createHash } from "node:crypto";
import type { ContextReceipt, TaskContextSnapshot } from "@kontext-brain/spec";
import type { LogicContextTarget } from "./domain.js";

export interface ValidateContextReceiptInput {
  readonly receipt: ContextReceipt;
  readonly snapshot: TaskContextSnapshot;
  readonly logic: LogicContextTarget;
  readonly allowedPaths: readonly string[];
  readonly now: string;
}

export function createContextReceipt(input: {
  readonly snapshot: TaskContextSnapshot;
  readonly logic: LogicContextTarget;
  readonly allowedPaths: readonly string[];
  readonly evidenceIds: readonly string[];
  readonly issuedAt: string;
  readonly expiresAt: string;
}): ContextReceipt {
  const plannedSymbolIds = uniqueSorted(input.logic.plannedSymbolIds);
  const allowedPaths = uniqueSorted(input.allowedPaths);
  const evidenceIds = uniqueSorted(input.evidenceIds);
  const normativeRevisions = [...input.snapshot.normativeRevisions].sort((left, right) =>
    revisionKey(left).localeCompare(revisionKey(right)),
  );
  const identity = [
    input.snapshot.taskId,
    input.logic.workItemId,
    plannedSymbolIds,
    allowedPaths,
    input.snapshot.contextDigest,
    normativeRevisions.map(revisionKey),
    evidenceIds,
    input.issuedAt,
    input.expiresAt,
  ];
  return Object.freeze({
    receiptId: `context-receipt:${createHash("sha256")
      .update(JSON.stringify(identity))
      .digest("hex")}`,
    taskId: input.snapshot.taskId,
    workItemId: input.logic.workItemId,
    plannedSymbolIds,
    allowedPaths,
    contextDigest: input.snapshot.contextDigest,
    normativeRevisions,
    evidenceIds,
    issuedAt: input.issuedAt,
    expiresAt: input.expiresAt,
  });
}

export function validateContextReceipt(input: ValidateContextReceiptInput): readonly string[] {
  const issues: string[] = [];
  const expected = createContextReceipt({
    snapshot: input.snapshot,
    logic: input.logic,
    allowedPaths: input.allowedPaths,
    evidenceIds: input.receipt.evidenceIds,
    issuedAt: input.receipt.issuedAt,
    expiresAt: input.receipt.expiresAt,
  });
  if (expected.receiptId !== input.receipt.receiptId) issues.push("receipt_id_mismatch");
  if (input.receipt.taskId !== input.snapshot.taskId) issues.push("task_mismatch");
  if (input.receipt.workItemId !== input.logic.workItemId) issues.push("work_item_mismatch");
  if (input.receipt.contextDigest !== input.snapshot.contextDigest) {
    issues.push("context_digest_mismatch");
  }
  if (
    JSON.stringify(uniqueSorted(input.receipt.plannedSymbolIds)) !==
    JSON.stringify(uniqueSorted(input.logic.plannedSymbolIds))
  ) {
    issues.push("planned_symbols_mismatch");
  }
  if (
    JSON.stringify(uniqueSorted(input.receipt.allowedPaths)) !==
    JSON.stringify(uniqueSorted(input.allowedPaths))
  ) {
    issues.push("allowed_paths_mismatch");
  }
  if (
    JSON.stringify(input.receipt.normativeRevisions.map(revisionKey).sort()) !==
    JSON.stringify(input.snapshot.normativeRevisions.map(revisionKey).sort())
  ) {
    issues.push("normative_revisions_mismatch");
  }
  const now = Date.parse(input.now);
  const issuedAt = Date.parse(input.receipt.issuedAt);
  const expiresAt = Date.parse(input.receipt.expiresAt);
  if (
    !Number.isFinite(now) ||
    !Number.isFinite(issuedAt) ||
    !Number.isFinite(expiresAt) ||
    expiresAt <= issuedAt
  ) {
    issues.push("invalid_time");
  } else if (now < issuedAt || now >= expiresAt) {
    issues.push("expired");
  }
  return issues;
}

function revisionKey(value: {
  readonly kind: string;
  readonly recordId: string;
  readonly revisionId: string;
}): string {
  return JSON.stringify([value.kind, value.recordId, value.revisionId]);
}

function uniqueSorted(values: readonly string[]): readonly string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}
