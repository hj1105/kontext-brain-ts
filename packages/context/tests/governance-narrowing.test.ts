import type {
  EffectiveNormativeRecord,
  GovernanceScope,
  NormativeRevision,
  TaskContract,
} from "@kontext-brain/spec";
import { describe, expect, it } from "vitest";
import { ContextCompiler } from "../src/context-compiler.js";
import type { CompileTaskContextInput, PlannedSymbolGovernanceLink } from "../src/domain.js";

const organizationId = "org:test";
const scope: GovernanceScope = { kind: "codebase", codebaseId: "codebase:test" };

function decision(index: number): NormativeRevision {
  return {
    kind: "decision",
    organizationId,
    recordId: `decision:area-${index}`,
    revisionId: `revision:area-${index}:1`,
    scope,
    statement: `Area ${index} retry delays use a factor of 2 and a ceiling of 20000 milliseconds.`,
    evidence: [{ evidenceId: `evidence:area-${index}`, kind: "chunk" }],
    egress: { allowedRuntimeProviders: ["codex"], requiresApproval: false },
    authoredBy: "owner",
    authoredAt: "2026-08-01T00:00:00.000Z",
  };
}

function effective(revision: NormativeRevision): EffectiveNormativeRecord {
  return {
    revision,
    activation: { state: "accepted", activatedAt: "2026-08-02T00:00:00.000Z" },
  };
}

/** One accepted record per area, as a real Codebase would carry. */
const corpusSize = 200;
const revisions = Array.from({ length: corpusSize }, (_, index) => decision(index));

const contract: TaskContract = {
  organizationId,
  taskId: "task:narrowing",
  intent: "Apply the current approved retry policy.",
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

function baseInput(
  links: readonly PlannedSymbolGovernanceLink[] | undefined,
): CompileTaskContextInput {
  return {
    contract,
    snapshot: {
      taskId: contract.taskId,
      contractDigest: "digest:contract",
      contextDigest: "digest:context",
      codeRevision: "revision:code",
      sourceFreshnessDigest: "digest:freshness",
      effectiveScopes: [scope],
      normativeRevisions: revisions.map((revision) => ({
        kind: revision.kind,
        recordId: revision.recordId,
        revisionId: revision.revisionId,
      })),
      requiredEvidenceIds: [],
      frozenAt: "2026-08-03T00:00:00.000Z",
    },
    currentCodeRevision: "revision:code",
    currentSourceFreshnessDigest: "digest:freshness",
    currentEffectiveScopes: [scope],
    currentNormativeRecords: revisions.map(effective),
    normativeRevisionCatalog: revisions,
    conflicts: [],
    evidence: [],
    runtimeProvider: "codex",
    logic: { workItemId: "work-item:charge", plannedSymbolIds: ["planned-symbol:charge"] },
    ...(links === undefined ? {} : { governanceLinks: links }),
    authorizedPaths: ["src/billing/charge.js"],
    issuedAt: "2026-08-04T00:00:00.000Z",
    expiresAt: "2026-08-04T00:15:00.000Z",
    totalTokenBudget: 8_000,
    optionalEvidenceTokenBudget: 2_000,
  };
}

const compiler = new ContextCompiler();
const compileTaskContext = (input: CompileTaskContextInput) => compiler.compile(input);

describe("governance narrowing by Planned Symbol", () => {
  it("compiles every accepted record when no links are supplied", () => {
    const result = compileTaskContext(baseInput(undefined));
    expect(result.mandatory.normativeRevisions).toHaveLength(corpusSize);
  });

  it("compiles only the records the Planned Symbol is governed by", () => {
    const links: PlannedSymbolGovernanceLink[] = [
      {
        plannedSymbolId: "planned-symbol:charge",
        recordId: "decision:area-7",
        revisionId: "revision:area-7:1",
        origin: "curated",
      },
      {
        plannedSymbolId: "planned-symbol:charge",
        recordId: "decision:area-42",
        revisionId: "revision:area-42:1",
        origin: "deterministic",
      },
      // Another symbol's link must not leak into this Work Item.
      {
        plannedSymbolId: "planned-symbol:elsewhere",
        recordId: "decision:area-9",
        revisionId: "revision:area-9:1",
        origin: "curated",
      },
    ];
    const result = compileTaskContext(baseInput(links));
    expect(result.mandatory.normativeRevisions.map((revision) => revision.recordId)).toEqual([
      "decision:area-42",
      "decision:area-7",
    ]);
  });

  it("gives a proposed link no enforcement authority", () => {
    const result = compileTaskContext(
      baseInput([
        {
          plannedSymbolId: "planned-symbol:charge",
          recordId: "decision:area-7",
          revisionId: "revision:area-7:1",
          origin: "proposed",
        },
      ]),
    );
    expect(result.mandatory.normativeRevisions).toEqual([]);
    expect(result.issues.map((issue) => issue.code)).toContain("ungoverned_planned_symbol");
  });

  it("reports a Planned Symbol that no authoritative link governs", () => {
    const result = compileTaskContext(baseInput([]));
    expect(result.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "ungoverned_planned_symbol",
          ref: "planned-symbol:charge",
        }),
      ]),
    );
    expect(result.mandatory.normativeRevisions).toEqual([]);
  });

  it("keeps the whole corpus from fitting, which is why narrowing exists", () => {
    const wide = compileTaskContext(baseInput(undefined));
    const narrow = compileTaskContext(
      baseInput([
        {
          plannedSymbolId: "planned-symbol:charge",
          recordId: "decision:area-7",
          revisionId: "revision:area-7:1",
          origin: "curated",
        },
      ]),
    );
    // The unnarrowed compilation blows the mandatory budget; the narrowed one
    // does not. That is the failure the product had no way to avoid before.
    expect(wide.issues.map((issue) => issue.code)).toContain("mandatory_budget_exceeded");
    expect(narrow.issues.map((issue) => issue.code)).not.toContain("mandatory_budget_exceeded");
  });
});
