import {
  type ContextAvailability,
  type EffectiveNormativeRecord,
  type GovernanceScope,
  type NormativeRevision,
  createTaskContextSnapshot,
  isTaskContextSnapshotValid,
  normativeRevisionContentDigest,
} from "@kontext-brain/spec";
import { createContextReceipt } from "./context-receipt.js";
import type {
  CompileTaskContextInput,
  CompiledTaskContext,
  ContextCompilationIssue,
  ContextEvidenceItem,
  ContextTokenEstimator,
  PrepareTaskContextSnapshotInput,
} from "./domain.js";

export class ConservativeContextTokenEstimator implements ContextTokenEstimator {
  estimate(value: string): number {
    return Math.max(1, Math.ceil(value.length / 4));
  }
}

export class ContextCompiler {
  constructor(
    private readonly tokenEstimator: ContextTokenEstimator = new ConservativeContextTokenEstimator(),
  ) {}

  compile(input: CompileTaskContextInput): CompiledTaskContext {
    const issues: ContextCompilationIssue[] = [];
    const addIssue = (
      code: ContextCompilationIssue["code"],
      message: string,
      ref?: string,
    ): void => {
      if (!issues.some((issue) => issue.code === code && issue.ref === ref)) {
        issues.push({ code, message, ref });
      }
    };

    if (!isTaskContextSnapshotValid(input.snapshot)) {
      addIssue("invalid_snapshot", "Task Context Snapshot digest is invalid");
    }
    if (input.contract.taskId !== input.snapshot.taskId) {
      addIssue("invalid_snapshot", "Task Contract does not match the Task Context Snapshot");
    }
    if (input.currentCodeRevision !== input.snapshot.baseCodeRevision) {
      addIssue("code_revision_stale", "Task base code revision has changed");
    }
    if (input.currentSourceFreshnessDigest !== input.snapshot.sourceFreshnessDigest) {
      addIssue("source_freshness_stale", "Required source freshness has changed");
    }
    if (!sameScopes(input.currentEffectiveScopes, input.snapshot.effectiveScopes)) {
      addIssue("scope_stale", "Effective Governance Scopes have changed");
    }
    if (input.conflicts.length > 0) {
      addIssue("normative_conflict", "Applicable mandatory normative records conflict");
    }
    if (
      !input.logic.workItemId.trim() ||
      input.logic.plannedSymbolIds.length === 0 ||
      input.logic.plannedSymbolIds.some((symbolId) => !symbolId.trim()) ||
      input.authorizedPaths.length === 0 ||
      input.authorizedPaths.some((allowedPath) => !allowedPath.trim())
    ) {
      addIssue(
        "invalid_logic_target",
        "Logic context requires a Work Item and at least one Planned Symbol",
      );
    }
    const issuedAt = Date.parse(input.issuedAt);
    const expiresAt = Date.parse(input.expiresAt);
    if (!Number.isFinite(issuedAt) || !Number.isFinite(expiresAt) || expiresAt <= issuedAt) {
      addIssue("invalid_receipt_expiry", "Context Receipt expiry must be after issuance");
    }

    const catalog = new Map(
      input.normativeRevisionCatalog.map((revision) => [revisionKey(revision), revision]),
    );
    const currentRecords = input.currentNormativeRecords.filter(
      (record) =>
        record.activation.state === "accepted" || record.activation.state === "accepted_local",
    );
    assessNormativeFreshness(input, catalog, currentRecords, addIssue);

    const governing = governingRevisionKeys(input, addIssue);
    const accessibleNormative = currentRecords
      .filter((record) => {
        // Narrowing by Planned Symbol is what makes a several-hundred-record
        // Codebase compilable at all. Without it every accepted record in the
        // organization is mandatory for every Work Item.
        if (governing && !governing.has(governanceKey(record.revision))) return false;
        const allowed = record.revision.egress.allowedRuntimeProviders.includes(
          input.runtimeProvider,
        );
        if (!allowed) {
          addIssue(
            "mandatory_context_inaccessible",
            "Mandatory context is not available to the selected runtime",
          );
        }
        return allowed;
      })
      .map((record) => record.revision)
      .sort(compareRevision);

    const evidenceById = new Map(input.evidence.map((evidence) => [evidence.evidenceId, evidence]));
    const mandatoryEvidence: ContextEvidenceItem[] = [];
    for (const evidenceId of uniqueSorted(input.snapshot.requiredEvidenceIds)) {
      const evidence = evidenceById.get(evidenceId);
      if (!evidence) {
        addIssue("mandatory_context_unavailable", "Required Evidence is unavailable", evidenceId);
        continue;
      }
      if (!evidence.allowedRuntimeProviders.includes(input.runtimeProvider)) {
        addIssue(
          "mandatory_context_inaccessible",
          "Mandatory context is not available to the selected runtime",
        );
        continue;
      }
      if (evidence.availability === "inaccessible") {
        addIssue(
          "mandatory_context_inaccessible",
          "Mandatory context is not available to the selected runtime",
        );
        continue;
      }
      if (evidence.availability === "unavailable") {
        addIssue("mandatory_context_unavailable", "Required Evidence is unavailable", evidenceId);
        continue;
      }
      if (evidence.availability === "conflict") {
        addIssue("mandatory_context_conflict", "Required Evidence is conflicting", evidenceId);
        mandatoryEvidence.push(evidence);
        continue;
      }
      if (evidence.availability !== "current") {
        addIssue("mandatory_context_stale", "Required Evidence is stale", evidenceId);
      }
      mandatoryEvidence.push(evidence);
    }

    const mandatory = {
      taskContract: input.contract,
      normativeRevisions: accessibleNormative,
      evidence: mandatoryEvidence.sort((left, right) =>
        left.evidenceId.localeCompare(right.evidenceId),
      ),
    };
    const mandatoryTokens = this.tokenEstimator.estimate(JSON.stringify(mandatory));
    if (mandatoryTokens > input.totalTokenBudget) {
      addIssue(
        "mandatory_budget_exceeded",
        "Mandatory context exceeds the optional Evidence budget and was retained in full",
      );
    }
    const optionalEvidence = this.selectOptionalEvidence(input, evidenceById, addIssue);
    const optionalTokens = optionalEvidence.reduce(
      (total, evidence) => total + this.tokenEstimator.estimate(evidence.text),
      0,
    );
    const status = compilationStatus(issues);
    const editingAllowed = status === "current";
    const receipt = editingAllowed
      ? createContextReceipt({
          snapshot: input.snapshot,
          logic: input.logic,
          allowedPaths: input.authorizedPaths,
          evidenceIds: [
            ...mandatoryEvidence.map((evidence) => evidence.evidenceId),
            ...optionalEvidence.map((evidence) => evidence.evidenceId),
          ],
          issuedAt: input.issuedAt,
          expiresAt: input.expiresAt,
        })
      : undefined;

    return {
      status,
      editingAllowed,
      contextDigest: input.snapshot.contextDigest,
      mandatory,
      optionalEvidence,
      issues,
      tokenUsage: {
        mandatory: mandatoryTokens,
        optional: optionalTokens,
        optionalBudget: input.optionalEvidenceTokenBudget,
      },
      receipt,
    };
  }

  private selectOptionalEvidence(
    input: CompileTaskContextInput,
    evidenceById: ReadonlyMap<string, ContextEvidenceItem>,
    addIssue: (code: ContextCompilationIssue["code"], message: string, ref?: string) => void,
  ): readonly ContextEvidenceItem[] {
    const required = new Set(input.snapshot.requiredEvidenceIds);
    const candidates = Array.from(evidenceById.values())
      .filter(
        (evidence) =>
          !required.has(evidence.evidenceId) &&
          evidence.availability === "current" &&
          evidence.allowedRuntimeProviders.includes(input.runtimeProvider),
      )
      .sort(
        (left, right) =>
          (right.relevance ?? 0) - (left.relevance ?? 0) ||
          left.evidenceId.localeCompare(right.evidenceId),
      );
    const selected: ContextEvidenceItem[] = [];
    let used = 0;
    for (const evidence of candidates) {
      const tokens = this.tokenEstimator.estimate(evidence.text);
      if (used + tokens > input.optionalEvidenceTokenBudget) {
        addIssue(
          "optional_context_omitted",
          "Optional Evidence was omitted to honor the token budget",
          evidence.evidenceId,
        );
        continue;
      }
      selected.push(evidence);
      used += tokens;
    }
    return selected;
  }
}

export function prepareTaskContextSnapshot(input: PrepareTaskContextSnapshotInput) {
  const normativeRevisions = uniqueBy(
    input.normativeRecords.map((record) => ({
      kind: record.revision.kind,
      recordId: record.revision.recordId,
      revisionId: record.revision.revisionId,
    })),
    revisionKey,
  ).sort((left, right) => revisionKey(left).localeCompare(revisionKey(right)));
  const requiredEvidenceIds = uniqueSorted([
    ...input.normativeRecords.flatMap((record) =>
      record.revision.evidence.map((evidence) => evidence.evidenceId),
    ),
    ...(input.additionalRequiredEvidenceIds ?? []),
  ]);
  return createTaskContextSnapshot(
    {
      taskId: input.contract.taskId,
      baseCodeRevision: input.baseCodeRevision,
      effectiveScopes: input.effectiveScopes,
      normativeRevisions,
      requiredEvidenceIds,
      sourceFreshnessDigest: input.sourceFreshnessDigest,
    },
    input.createdAt,
  );
}

function assessNormativeFreshness(
  input: CompileTaskContextInput,
  catalog: ReadonlyMap<string, NormativeRevision>,
  currentRecords: readonly EffectiveNormativeRecord[],
  addIssue: (code: ContextCompilationIssue["code"], message: string, ref?: string) => void,
): void {
  const snapshotRevisions = input.snapshot.normativeRevisions.map((reference) => ({
    reference,
    revision: catalog.get(revisionKey(reference)),
  }));
  for (const item of snapshotRevisions) {
    if (!item.revision) {
      addIssue(
        "mandatory_context_unavailable",
        "A Task normative revision is unavailable",
        item.reference.revisionId,
      );
      continue;
    }
    const current = currentRecords.filter(
      (record) =>
        record.revision.kind === item.revision?.kind &&
        record.revision.recordId === item.revision.recordId,
    );
    if (
      current.length === 0 ||
      !current.some(
        (record) =>
          normativeRevisionContentDigest(record.revision) ===
          normativeRevisionContentDigest(item.revision as NormativeRevision),
      )
    ) {
      addIssue(
        "normative_revision_stale",
        "A Task normative revision is no longer current",
        item.reference.revisionId,
      );
    }
  }
  for (const current of currentRecords) {
    const represented = snapshotRevisions.some(
      (item) =>
        item.revision?.kind === current.revision.kind &&
        item.revision.recordId === current.revision.recordId &&
        normativeRevisionContentDigest(item.revision) ===
          normativeRevisionContentDigest(current.revision),
    );
    if (!represented) {
      addIssue(
        "normative_revision_stale",
        "A new mandatory normative revision is not present in the Task snapshot",
        current.revision.revisionId,
      );
    }
  }
}

function compilationStatus(issues: readonly ContextCompilationIssue[]): ContextAvailability {
  if (
    issues.some(
      (issue) => issue.code === "normative_conflict" || issue.code === "mandatory_context_conflict",
    )
  ) {
    return "conflict";
  }
  if (issues.some((issue) => issue.code === "mandatory_context_inaccessible")) {
    return "inaccessible";
  }
  if (
    issues.some(
      (issue) =>
        issue.code === "mandatory_context_unavailable" ||
        issue.code === "invalid_logic_target" ||
        issue.code === "invalid_receipt_expiry",
    )
  ) {
    return "unavailable";
  }
  if (
    issues.some((issue) =>
      [
        "invalid_snapshot",
        "code_revision_stale",
        "source_freshness_stale",
        "scope_stale",
        "normative_revision_stale",
        "mandatory_context_stale",
      ].includes(issue.code),
    )
  ) {
    return "stale";
  }
  return "current";
}

function sameScopes(left: readonly GovernanceScope[], right: readonly GovernanceScope[]): boolean {
  return JSON.stringify(left.map(scopeKey).sort()) === JSON.stringify(right.map(scopeKey).sort());
}

function scopeKey(scope: GovernanceScope): string {
  return JSON.stringify(scope);
}

function revisionKey(value: {
  readonly kind: string;
  readonly recordId: string;
  readonly revisionId: string;
}): string {
  return JSON.stringify([value.kind, value.recordId, value.revisionId]);
}

function compareRevision(left: NormativeRevision, right: NormativeRevision): number {
  return revisionKey(left).localeCompare(revisionKey(right));
}

function uniqueSorted(values: readonly string[]): readonly string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
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

/**
 * The set of revisions the Work Item's Planned Symbols are governed by, or
 * undefined when no links were supplied and the caller wants the previous
 * organization-wide behaviour.
 *
 * A proposed link carries no enforcement authority, so it does not admit a
 * record into mandatory context. A Planned Symbol with no authoritative link is
 * reported rather than silently receiving nothing.
 */
function governanceKey(value: {
  readonly recordId: string;
  readonly revisionId: string;
}): string {
  return JSON.stringify([value.recordId, value.revisionId]);
}

function governingRevisionKeys(
  input: CompileTaskContextInput,
  addIssue: (code: ContextCompilationIssue["code"], message: string, ref?: string) => void,
): Set<string> | undefined {
  const links = input.governanceLinks;
  if (links === undefined) return undefined;
  const authoritative = links.filter((link) => link.origin !== "proposed");
  const keys = new Set<string>();
  for (const link of authoritative) {
    if (!input.logic.plannedSymbolIds.includes(link.plannedSymbolId)) continue;
    keys.add(governanceKey(link));
  }
  for (const plannedSymbolId of input.logic.plannedSymbolIds) {
    if (!authoritative.some((link) => link.plannedSymbolId === plannedSymbolId)) {
      addIssue(
        "ungoverned_planned_symbol",
        "No authoritative governance link exists for this Planned Symbol",
        plannedSymbolId,
      );
    }
  }
  return keys;
}
