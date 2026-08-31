import type {
  DecisionRevision,
  EffectiveNormativeRecord,
  NormativeActivation,
  TaskContract,
} from "@kontext-brain/spec";
import { describe, expect, it } from "vitest";
import {
  type CompileTaskContextInput,
  ContextCompiler,
  type ContextEvidenceItem,
  prepareTaskContextSnapshot,
  validateContextReceipt,
} from "../src/index.js";

const contract: TaskContract = {
  taskId: "task:context",
  intent: "Compile exact context before editing one logic unit.",
  acceptance: [
    {
      criterionId: "acceptance:receipt",
      statement: "Every logic unit receives a Context Receipt.",
      verifier: { kind: "test", ref: "context-compiler.test.ts" },
    },
  ],
  nonGoals: ["Execute a provider runtime"],
  targets: ["planned-symbol:run"],
  risk: "low",
};
const localRevision = decision(
  "revision:local:1",
  "Use the accepted Domain Terms for public interfaces.",
  { kind: "workspace", workspaceId: "workspace:local" },
);
const localRecord = effective(localRevision, "accepted_local", "local");
const snapshot = prepareTaskContextSnapshot({
  contract,
  baseCodeRevision: "commit:base",
  effectiveScopes: [localRevision.scope],
  normativeRecords: [localRecord],
  additionalRequiredEvidenceIds: ["evidence:task"],
  sourceFreshnessDigest: "freshness:1",
  createdAt: "2026-08-28T00:00:00.000Z",
});
const requiredEvidence: ContextEvidenceItem[] = [
  {
    evidenceId: "evidence:decision",
    text: "The owner accepted the terminology decision.",
    availability: "current",
    allowedRuntimeProviders: ["codex", "claude"],
  },
  {
    evidenceId: "evidence:task",
    text: "The user asked for exact context before every logic unit.",
    availability: "current",
    allowedRuntimeProviders: ["codex", "claude"],
  },
];

describe("ContextCompiler", () => {
  it("never trims mandatory context and issues a receipt for a current logic unit", () => {
    const compiler = new ContextCompiler({ estimate: (value) => value.length });
    const result = compiler.compile(
      input({
        evidence: [
          ...requiredEvidence,
          {
            evidenceId: "evidence:optional:high",
            text: "near",
            availability: "current",
            allowedRuntimeProviders: ["codex"],
            relevance: 10,
          },
          {
            evidenceId: "evidence:optional:low",
            text: "too-large",
            availability: "current",
            allowedRuntimeProviders: ["codex"],
            relevance: 1,
          },
        ],
        totalTokenBudget: 1,
        optionalEvidenceTokenBudget: 4,
      }),
    );

    expect(result.status).toBe("current");
    expect(result.editingAllowed).toBe(true);
    expect(result.mandatory.normativeRevisions).toEqual([localRevision]);
    expect(result.mandatory.evidence.map((evidence) => evidence.evidenceId)).toEqual([
      "evidence:decision",
      "evidence:task",
    ]);
    expect(result.optionalEvidence.map((evidence) => evidence.evidenceId)).toEqual([
      "evidence:optional:high",
    ]);
    expect(result.issues.map((issue) => issue.code)).toEqual(
      expect.arrayContaining(["mandatory_budget_exceeded", "optional_context_omitted"]),
    );
    expect(result.receipt).toMatchObject({
      taskId: contract.taskId,
      workItemId: "work-item:run",
      plannedSymbolIds: ["planned-symbol:run"],
      contextDigest: snapshot.contextDigest,
    });
    expect(
      result.receipt
        ? validateContextReceipt({
            receipt: result.receipt,
            snapshot,
            logic: {
              workItemId: "work-item:run",
              plannedSymbolIds: ["planned-symbol:run"],
            },
            allowedPaths: ["src/run.ts"],
            now: "2026-08-28T00:30:00.000Z",
          })
        : ["missing_receipt"],
    ).toEqual([]);
  });

  it("fails closed without revealing which mandatory rule denied provider egress", () => {
    const result = new ContextCompiler().compile(input({ runtimeProvider: "untrusted-runtime" }));

    expect(result.status).toBe("inaccessible");
    expect(result.editingAllowed).toBe(false);
    expect(result.receipt).toBeUndefined();
    expect(result.mandatory.normativeRevisions).toEqual([]);
    expect(
      result.issues.filter((issue) => issue.code === "mandatory_context_inaccessible"),
    ).toEqual([
      {
        code: "mandatory_context_inaccessible",
        message: "Mandatory context is not available to the selected runtime",
        ref: undefined,
      },
    ]);
  });

  it("marks a changed active revision stale but accepts an identical canonical replacement", () => {
    const changed = decision("revision:managed:changed", "Use different terminology.", {
      kind: "codebase",
      codebaseId: "codebase:example",
    });
    const changedResult = new ContextCompiler().compile(
      input({
        currentNormativeRecords: [effective(changed, "accepted", "managed")],
        normativeRevisionCatalog: [localRevision, changed],
      }),
    );
    expect(changedResult.status).toBe("stale");
    expect(changedResult.receipt).toBeUndefined();

    const identical = decision("revision:managed:identical", localRevision.statement, {
      kind: "codebase",
      codebaseId: "codebase:example",
    });
    const identicalResult = new ContextCompiler().compile(
      input({
        currentNormativeRecords: [effective(identical, "accepted", "managed")],
        normativeRevisionCatalog: [localRevision, identical],
      }),
    );
    expect(identicalResult.status).toBe("current");
    expect(identicalResult.receipt).toBeDefined();
    expect(identicalResult.mandatory.normativeRevisions[0]?.revisionId).toBe(
      "revision:managed:identical",
    );
  });

  it("blocks conflicting required Evidence and detects receipt tampering", () => {
    const result = new ContextCompiler().compile(
      input({
        evidence: requiredEvidence.map((evidence) =>
          evidence.evidenceId === "evidence:task"
            ? { ...evidence, availability: "conflict" as const }
            : evidence,
        ),
      }),
    );
    expect(result.status).toBe("conflict");
    expect(result.receipt).toBeUndefined();

    const current = new ContextCompiler().compile(input());
    if (!current.receipt) throw new Error("Test fixture requires a Context Receipt");
    expect(
      validateContextReceipt({
        receipt: { ...current.receipt, receiptId: "tampered" },
        snapshot,
        logic: {
          workItemId: "work-item:run",
          plannedSymbolIds: ["planned-symbol:run"],
        },
        allowedPaths: ["src/run.ts"],
        now: "2026-08-28T00:30:00.000Z",
      }),
    ).toContain("receipt_id_mismatch");
  });
});

function input(overrides: Partial<CompileTaskContextInput> = {}): CompileTaskContextInput {
  return {
    contract,
    snapshot,
    currentCodeRevision: "commit:base",
    currentSourceFreshnessDigest: "freshness:1",
    currentEffectiveScopes: [localRevision.scope],
    currentNormativeRecords: [localRecord],
    normativeRevisionCatalog: [localRevision],
    conflicts: [],
    evidence: requiredEvidence,
    runtimeProvider: "codex",
    logic: {
      workItemId: "work-item:run",
      plannedSymbolIds: ["planned-symbol:run"],
    },
    authorizedPaths: ["src/run.ts"],
    issuedAt: "2026-08-28T00:10:00.000Z",
    expiresAt: "2026-08-28T01:10:00.000Z",
    totalTokenBudget: 10_000,
    optionalEvidenceTokenBudget: 100,
    ...overrides,
  };
}

function decision(
  revisionId: string,
  statement: string,
  scope: DecisionRevision["scope"],
): DecisionRevision {
  return {
    kind: "decision",
    organizationId: "org:acme",
    recordId: "decision:terms",
    revisionId,
    scope,
    evidence: [{ evidenceId: "evidence:decision", sourceSpan: "decision 1" }],
    egress: {
      dataClassification: "internal",
      allowedRuntimeProviders: ["codex", "claude"],
    },
    authoredBy: "user:owner",
    authoredAt: "2026-08-28T00:00:00.000Z",
    statement,
  };
}

function effective(
  revision: DecisionRevision,
  state: NormativeActivation["state"],
  origin: EffectiveNormativeRecord["origin"],
): EffectiveNormativeRecord {
  return {
    origin,
    revision,
    activation: {
      organizationId: revision.organizationId,
      kind: revision.kind,
      recordId: revision.recordId,
      revisionId: revision.revisionId,
      scope: revision.scope,
      state,
      acceptedBy: "user:owner",
      acceptedAt: "2026-08-28T00:01:00.000Z",
      mergeCommit: state === "accepted" ? "abc123" : undefined,
    },
  };
}
