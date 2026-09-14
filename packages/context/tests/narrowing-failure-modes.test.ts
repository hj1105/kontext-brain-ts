import type {
  EffectiveNormativeRecord,
  GovernanceScope,
  NormativeRevision,
  TaskContract,
} from "@kontext-brain/spec";
import { describe, expect, it } from "vitest";
import { ContextCompiler } from "../src/context-compiler.js";
import type { CompileTaskContextInput, PlannedSymbolGovernanceLink } from "../src/domain.js";

/**
 * Narrowing makes a large Codebase compilable, but it can also drop governance
 * that should have applied. These cases cover where it is wrong or risky rather
 * than where it works, because a suite that only confirms the happy path will
 * happily ship a filter that loses mandatory rules.
 */
const organizationId = "org:test";
const codebaseScope: GovernanceScope = { kind: "codebase", codebaseId: "codebase:test" };
const organizationScope: GovernanceScope = { kind: "organization", organizationId };

function revision(input: {
  readonly recordId: string;
  readonly scope: GovernanceScope;
  readonly statement?: string;
}): NormativeRevision {
  return {
    kind: "decision",
    organizationId,
    recordId: input.recordId,
    revisionId: `${input.recordId}:1`,
    scope: input.scope,
    statement: input.statement ?? `${input.recordId} applies.`,
    evidence: [{ evidenceId: `evidence:${input.recordId}`, kind: "chunk" }],
    egress: { allowedRuntimeProviders: ["codex"], requiresApproval: false },
    authoredBy: "owner",
    authoredAt: "2026-08-01T00:00:00.000Z",
  };
}

function effective(item: NormativeRevision): EffectiveNormativeRecord {
  return {
    revision: item,
    activation: { state: "accepted", activatedAt: "2026-08-02T00:00:00.000Z" },
  };
}

const contract: TaskContract = {
  organizationId,
  taskId: "task:failure-modes",
  intent: "Apply the current approved policy.",
  acceptance: [
    {
      criterionId: "criterion:policy",
      statement: "The target follows the approved policy.",
      verifier: { kind: "test", ref: "workspace:test" },
    },
  ],
  nonGoals: ["Editing unrelated subsystems"],
  targets: ["planned-symbol:charge"],
  risk: "low",
};

function input(
  revisions: readonly NormativeRevision[],
  links: readonly PlannedSymbolGovernanceLink[],
  plannedSymbolIds: readonly string[] = ["planned-symbol:charge"],
): CompileTaskContextInput {
  return {
    contract,
    snapshot: {
      taskId: contract.taskId,
      contractDigest: "digest:contract",
      contextDigest: "digest:context",
      codeRevision: "revision:code",
      sourceFreshnessDigest: "digest:freshness",
      effectiveScopes: [codebaseScope, organizationScope],
      normativeRevisions: revisions.map((item) => ({
        kind: item.kind,
        recordId: item.recordId,
        revisionId: item.revisionId,
      })),
      requiredEvidenceIds: [],
      frozenAt: "2026-08-03T00:00:00.000Z",
    },
    currentCodeRevision: "revision:code",
    currentSourceFreshnessDigest: "digest:freshness",
    currentEffectiveScopes: [codebaseScope, organizationScope],
    currentNormativeRecords: revisions.map(effective),
    normativeRevisionCatalog: revisions,
    conflicts: [],
    evidence: [],
    runtimeProvider: "codex",
    logic: { workItemId: "work-item:charge", plannedSymbolIds },
    governanceLinks: links,
    authorizedPaths: ["src/billing/charge.js"],
    issuedAt: "2026-08-04T00:00:00.000Z",
    expiresAt: "2026-08-04T00:15:00.000Z",
    totalTokenBudget: 8_000,
    optionalEvidenceTokenBudget: 2_000,
  };
}

const compiler = new ContextCompiler();
const compile = (value: CompileTaskContextInput) => compiler.compile(value);

function link(
  recordId: string,
  plannedSymbolId = "planned-symbol:charge",
): PlannedSymbolGovernanceLink {
  return { plannedSymbolId, recordId, revisionId: `${recordId}:1`, origin: "curated" };
}

describe("narrowing failure modes", () => {
  it("never drops an organization-scoped record for lack of a symbol link", () => {
    const revisions = [
      revision({ recordId: "decision:billing-retry", scope: codebaseScope }),
      revision({
        recordId: "decision:company-wide-rounding",
        scope: organizationScope,
        statement: "Monetary amounts round half to even everywhere.",
      }),
    ];
    const result = compile(input(revisions, [link("decision:billing-retry")]));
    // A narrower scope may add constraints but cannot weaken an organization
    // rule, so narrowing must not be able to hide one.
    expect(result.mandatory.normativeRevisions.map((item) => item.recordId)).toEqual([
      "decision:billing-retry",
      "decision:company-wide-rounding",
    ]);
  });

  it("loses a cross-cutting codebase record that no link points at", () => {
    const revisions = [
      revision({ recordId: "decision:billing-retry", scope: codebaseScope }),
      revision({
        recordId: "decision:shared-logging",
        scope: codebaseScope,
        statement: "Every subsystem logs a correlation id.",
      }),
    ];
    const result = compile(input(revisions, [link("decision:billing-retry")]));
    // This is a real limitation, not a feature: a codebase-wide rule that the
    // ontology never linked to this symbol is silently absent. Retrieval would
    // at least have had a chance to surface it.
    expect(result.mandatory.normativeRevisions.map((item) => item.recordId)).toEqual([
      "decision:billing-retry",
    ]);
  });

  it("hands over the wrong policy when a symbol is linked to the wrong record", () => {
    const revisions = [
      revision({
        recordId: "decision:billing-retry",
        scope: codebaseScope,
        statement: "Billing uses a factor of 3.",
      }),
      revision({
        recordId: "decision:notify-retry",
        scope: codebaseScope,
        statement: "Notify uses a factor of 2.",
      }),
    ];
    // A misclassified file produces a confidently wrong single answer, which is
    // worse than a ranked list containing both candidates.
    const result = compile(input(revisions, [link("decision:notify-retry")]));
    expect(result.mandatory.normativeRevisions.map((item) => item.recordId)).toEqual([
      "decision:notify-retry",
    ]);
    expect(result.issues.map((issue) => issue.code)).not.toContain("ungoverned_planned_symbol");
  });

  it("stops narrowing anything when the ontology is too coarse to separate areas", () => {
    const revisions = Array.from({ length: 200 }, (_, index) =>
      revision({ recordId: `decision:area-${index}`, scope: codebaseScope }),
    );
    // One node holding everything links every record to the symbol, so the
    // filter passes all of them and the budget problem returns.
    const links = revisions.map((item) => link(item.recordId));
    const result = compile(input(revisions, links));
    expect(result.mandatory.normativeRevisions).toHaveLength(revisions.length);
    expect(result.issues.map((issue) => issue.code)).toContain("mandatory_budget_exceeded");
  });

  it("reports every ungoverned symbol, not just the first", () => {
    const revisions = [revision({ recordId: "decision:billing-retry", scope: codebaseScope })];
    const result = compile(
      input(
        revisions,
        [link("decision:billing-retry", "planned-symbol:charge")],
        ["planned-symbol:charge", "planned-symbol:refund", "planned-symbol:payout"],
      ),
    );
    const ungoverned = result.issues
      .filter((issue) => issue.code === "ungoverned_planned_symbol")
      .map((issue) => issue.ref);
    expect(ungoverned).toEqual(["planned-symbol:refund", "planned-symbol:payout"]);
  });
});
