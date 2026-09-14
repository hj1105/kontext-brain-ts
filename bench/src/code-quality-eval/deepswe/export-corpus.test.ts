import { prepareTaskContextSnapshot } from "@kontext-brain/context";
import type { CurrentTaskContextState, PreparedTaskContext } from "@kontext-brain/context";
import type { DecisionRevision, EffectiveNormativeRecord, TaskContract } from "@kontext-brain/spec";
import { describe, expect, it } from "vitest";
import { sha256, validateCorpus } from "./corpus.js";
import { type DeepSweCorpusExportInput, exportDeepSweCorpus } from "./export-corpus.js";

describe("DeepSWE corpus export", () => {
  it("freezes the exact Task Context Snapshot with Evidence provenance and symbol selectors", () => {
    const input = fixture();
    const corpus = exportDeepSweCorpus(input);

    expect(corpus).toMatchObject({
      taskId: input.taskId,
      organizationId: input.organizationId,
      runtimeProvider: input.runtimeProvider,
      baseCodeRevision: input.prepared.snapshot.baseCodeRevision,
      contextDigest: input.prepared.snapshot.contextDigest,
      sourceFreshnessDigest: input.prepared.snapshot.sourceFreshnessDigest,
      snapshotAt: input.prepared.snapshot.createdAt,
      generator: { name: "kontext-brain", revision: "commit:exporter" },
    });
    expect(corpus.evidence).toEqual([
      {
        evidenceId: "evidence:design",
        resourceId: "resource:design",
        chunkId: "chunk:delegation",
        title: "Delegation design",
        text: "Delegated output must be returned as one tool result.",
        sourceSpan: "Delegation protocol",
        source: {
          connectorId: "github",
          externalId: "https://github.example/acme/service/issues/42",
          type: "github_issue",
        },
        observedAt: "2026-01-01T00:00:00.000Z",
        contentSha256: sha256("Delegated output must be returned as one tool result."),
        ontologyNodeIds: ["agent", "delegation"],
        allowedRuntimeProviders: ["openai"],
      },
    ]);
    expect(corpus.normativeRecords).toEqual([
      {
        revision: input.current.normativeRevisionCatalog[0],
        symbolSelectors: [
          {
            relativePath: "src/delegate.ts",
            qualifiedName: "executeDelegation",
          },
        ],
      },
    ]);
    expect(() =>
      validateCorpus(corpus, input.taskId, "/tmp/deep-swe/tasks/delegation-task"),
    ).not.toThrow();
  });

  it("fails closed for stale, inaccessible, or untraceable Evidence", () => {
    const input = fixture();
    expect(() =>
      exportDeepSweCorpus({
        ...input,
        current: { ...input.current, sourceFreshnessDigest: "sha256:newer" },
      }),
    ).toThrow(/stale relative/);
    expect(() =>
      exportDeepSweCorpus({
        ...input,
        current: {
          ...input.current,
          evidence: input.current.evidence.map(
            ({ provenance: _provenance, ...evidence }) => evidence,
          ),
        },
      }),
    ).toThrow(/no Resource\/Chunk provenance/);
    expect(() => exportDeepSweCorpus({ ...input, runtimeProvider: "claude" })).toThrow(
      /not available to claude/,
    );
  });

  it("requires explicit opt-in for an infrastructure-only empty export", () => {
    const contract: TaskContract = {
      taskId: "empty-task",
      intent: "Validate infrastructure only.",
      acceptance: [
        {
          criterionId: "acceptance:smoke",
          statement: "The harness starts.",
          verifier: { kind: "test", ref: "smoke" },
        },
      ],
      nonGoals: [],
      targets: ["harness"],
      risk: "low",
    };
    const current: CurrentTaskContextState = {
      codeRevision: "a".repeat(40),
      sourceFreshnessDigest: "sha256:empty",
      effectiveScopes: [],
      normativeRecords: [],
      normativeRevisionCatalog: [],
      conflicts: [],
      evidence: [],
      logicPlans: [],
    };
    const prepared: PreparedTaskContext = {
      contract,
      snapshot: prepareTaskContextSnapshot({
        contract,
        baseCodeRevision: current.codeRevision,
        effectiveScopes: current.effectiveScopes,
        normativeRecords: [],
        sourceFreshnessDigest: current.sourceFreshnessDigest,
        createdAt: "2026-01-02T00:00:00.000Z",
      }),
      additionalRequiredEvidenceIds: [],
    };
    const input: DeepSweCorpusExportInput = {
      taskId: contract.taskId,
      organizationId: "organization:fixture",
      runtimeProvider: "openai",
      generatorRevision: "commit:exporter",
      prepared,
      current,
    };

    expect(() => exportDeepSweCorpus(input)).toThrow(/requires at least one/);
    expect(exportDeepSweCorpus({ ...input, allowEmpty: true })).toMatchObject({
      evidence: [],
      normativeRecords: [],
    });
  });
});

function fixture(): DeepSweCorpusExportInput {
  const taskId = "delegation-task";
  const organizationId = "organization:fixture";
  const text = "Delegated output must be returned as one tool result.";
  const revision: DecisionRevision = {
    kind: "decision",
    organizationId,
    recordId: "decision:delegation-result",
    revisionId: "revision:delegation-result:1",
    scope: { kind: "codebase", codebaseId: "codebase:service" },
    evidence: [{ evidenceId: "evidence:design", sourceSpan: "Delegation protocol" }],
    egress: { dataClassification: "public", allowedRuntimeProviders: ["openai"] },
    authoredBy: "user:owner",
    authoredAt: "2026-01-01T00:00:00.000Z",
    statement: "Return delegated output as exactly one tool result.",
  };
  const effective: EffectiveNormativeRecord = {
    origin: "managed",
    revision,
    activation: {
      organizationId,
      kind: revision.kind,
      recordId: revision.recordId,
      revisionId: revision.revisionId,
      scope: revision.scope,
      state: "accepted",
      acceptedBy: "user:owner",
      acceptedAt: "2026-01-01T00:01:00.000Z",
      mergeCommit: "commit:approval",
    },
  };
  const contract: TaskContract = {
    taskId,
    intent: "Implement delegated result handling.",
    acceptance: [
      {
        criterionId: "acceptance:delegation",
        statement: "Delegated output returns to the caller.",
        verifier: { kind: "test", ref: "tests/delegate.test.ts" },
      },
    ],
    nonGoals: [],
    targets: ["planned-symbol:delegation"],
    risk: "medium",
  };
  const current: CurrentTaskContextState = {
    codeRevision: "a".repeat(40),
    sourceFreshnessDigest: "sha256:fresh",
    effectiveScopes: [revision.scope],
    normativeRecords: [effective],
    normativeRevisionCatalog: [revision],
    conflicts: [],
    evidence: [
      {
        evidenceId: "evidence:design",
        text,
        sourceSpan: "Delegation protocol",
        availability: "current",
        allowedRuntimeProviders: ["openai"],
        provenance: {
          resourceId: "resource:design",
          chunkId: "chunk:delegation",
          resourceTitle: "Delegation design",
          source: {
            connectorId: "github",
            externalId: "https://github.example/acme/service/issues/42",
            type: "github_issue",
          },
          observedAt: "2026-01-01T00:00:00.000Z",
          contentHash: sha256(text),
          ontologyNodeIds: ["delegation", "agent", "delegation"],
        },
      },
    ],
    logicPlans: [
      {
        workItemId: "work-item:delegation",
        plannedSymbolIds: ["planned-symbol:delegation"],
        plannedSymbols: [
          {
            plannedSymbolId: "planned-symbol:delegation",
            taskId,
            intendedIdentity: {
              relativePath: "src/delegate.ts",
              kind: "function",
              qualifiedName: "executeDelegation",
            },
            responsibility: "Return delegated results to the caller",
          },
        ],
        allowedPaths: ["src/delegate.ts"],
      },
    ],
    governanceLinks: [
      {
        plannedSymbolId: "planned-symbol:delegation",
        recordId: revision.recordId,
        revisionId: revision.revisionId,
        origin: "curated",
      },
    ],
  };
  const prepared: PreparedTaskContext = {
    contract,
    snapshot: prepareTaskContextSnapshot({
      contract,
      baseCodeRevision: current.codeRevision,
      effectiveScopes: current.effectiveScopes,
      normativeRecords: current.normativeRecords,
      sourceFreshnessDigest: current.sourceFreshnessDigest,
      createdAt: "2026-01-02T00:00:00.000Z",
    }),
    additionalRequiredEvidenceIds: [],
  };
  return {
    taskId,
    organizationId,
    runtimeProvider: "openai",
    generatorRevision: "commit:exporter",
    prepared,
    current,
  };
}
