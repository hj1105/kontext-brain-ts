import type { DecisionRevision, EffectiveNormativeRecord, TaskContract } from "@kontext-brain/spec";
import { describe, expect, it } from "vitest";
import {
  type CurrentTaskContextState,
  InMemoryPreparedTaskContextStore,
  InMemoryTaskContextStateProvider,
  TaskContextWorkflow,
} from "../src/index.js";

const contract: TaskContract = {
  taskId: "task:workflow",
  intent: "Require current Kontext context for each logic unit.",
  acceptance: [
    {
      criterionId: "acceptance:receipt",
      statement: "begin logic returns a receipt",
      verifier: { kind: "test", ref: "task-context-workflow.test.ts" },
    },
  ],
  nonGoals: [],
  targets: ["planned-symbol:execute"],
  risk: "low",
};
const firstRevision = decision("revision:1", "Use Context Receipt before editing.");

describe("TaskContextWorkflow", () => {
  it("prepares, detects drift on begin, and explicitly refreshes before issuing a new receipt", async () => {
    const states = new InMemoryTaskContextStateProvider();
    const workflow = new TaskContextWorkflow(states, new InMemoryPreparedTaskContextStore());
    states.set(contract.taskId, state("commit:1", firstRevision));

    const prepared = await workflow.prepareTask({
      contract,
      createdAt: "2026-08-28T00:00:00.000Z",
    });
    const firstBegin = await workflow.beginLogic(beginRequest());
    expect(firstBegin.status).toBe("current");
    expect(firstBegin.receipt?.contextDigest).toBe(prepared.snapshot.contextDigest);

    const secondRevision = {
      ...decision("revision:2", "Use a current Context Receipt before editing."),
      supersedesRevisionId: firstRevision.revisionId,
    };
    states.set(contract.taskId, state("commit:2", secondRevision, [firstRevision]));
    const staleBegin = await workflow.beginLogic(beginRequest());
    expect(staleBegin.status).toBe("stale");
    expect(staleBegin.receipt).toBeUndefined();

    const refreshed = await workflow.refreshTaskContext({
      taskId: contract.taskId,
      createdAt: "2026-08-28T00:20:00.000Z",
    });
    expect(refreshed.changed).toBe(true);
    expect(refreshed.addedNormativeRevisionIds).toEqual(["revision:2"]);
    expect(refreshed.removedNormativeRevisionIds).toEqual(["revision:1"]);

    const currentBegin = await workflow.beginLogic(beginRequest());
    expect(currentBegin.status).toBe("current");
    expect(currentBegin.receipt?.contextDigest).toBe(refreshed.current.contextDigest);
  });

  it("passes sidecar-owned governance links into compilation", async () => {
    const states = new InMemoryTaskContextStateProvider();
    const workflow = new TaskContextWorkflow(states, new InMemoryPreparedTaskContextStore());
    const unrelated = decision("revision:unrelated", "An unrelated subsystem keeps its policy.");
    states.set(contract.taskId, {
      ...state("commit:1", firstRevision),
      normativeRecords: [effective(firstRevision), effective(unrelated)],
      normativeRevisionCatalog: [firstRevision, unrelated],
      evidence: [firstRevision, unrelated].map((candidate) => ({
        evidenceId: `evidence:${candidate.revisionId}`,
        text: candidate.statement,
        availability: "current",
        allowedRuntimeProviders: ["codex"],
      })),
      governanceLinks: [
        {
          plannedSymbolId: "planned-symbol:execute",
          recordId: firstRevision.recordId,
          revisionId: firstRevision.revisionId,
          origin: "curated",
        },
      ],
    });
    await workflow.prepareTask({ contract, createdAt: "2026-08-28T00:00:00.000Z" });

    const compiled = await workflow.beginLogic(beginRequest());
    expect(compiled.mandatory.normativeRevisions.map((revision) => revision.revisionId)).toEqual([
      firstRevision.revisionId,
    ]);
  });
});

function beginRequest() {
  return {
    taskId: contract.taskId,
    logic: {
      workItemId: "work-item:execute",
      plannedSymbolIds: ["planned-symbol:execute"],
    },
    runtimeProvider: "codex",
    issuedAt: "2026-08-28T00:30:00.000Z",
    expiresAt: "2026-08-28T01:30:00.000Z",
    totalTokenBudget: 10_000,
    optionalEvidenceTokenBudget: 1_000,
  };
}

function state(
  codeRevision: string,
  revision: DecisionRevision,
  history: readonly DecisionRevision[] = [],
): CurrentTaskContextState {
  return {
    codeRevision,
    sourceFreshnessDigest: "freshness:1",
    effectiveScopes: [revision.scope],
    normativeRecords: [effective(revision)],
    normativeRevisionCatalog: [...history, revision],
    conflicts: [],
    evidence: [...history, revision].map((candidate) => ({
      evidenceId: `evidence:${candidate.revisionId}`,
      text: candidate.statement,
      availability: "current",
      allowedRuntimeProviders: ["codex", "claude"],
    })),
    logicPlans: [
      {
        workItemId: "work-item:execute",
        plannedSymbolIds: ["planned-symbol:execute"],
        allowedPaths: ["src/execute.ts"],
      },
    ],
  };
}

function decision(revisionId: string, statement: string): DecisionRevision {
  return {
    kind: "decision",
    organizationId: "org:acme",
    recordId: "decision:context-receipt",
    revisionId,
    scope: { kind: "workspace", workspaceId: "workspace:local" },
    evidence: [{ evidenceId: `evidence:${revisionId}` }],
    egress: {
      dataClassification: "internal",
      allowedRuntimeProviders: ["codex", "claude"],
    },
    authoredBy: "user:owner",
    authoredAt: "2026-08-28T00:00:00.000Z",
    statement,
  };
}

function effective(revision: DecisionRevision): EffectiveNormativeRecord {
  return {
    origin: "local",
    revision,
    activation: {
      organizationId: revision.organizationId,
      kind: revision.kind,
      recordId: revision.recordId,
      revisionId: revision.revisionId,
      scope: revision.scope,
      state: "accepted_local",
      acceptedBy: "user:owner",
      acceptedAt: "2026-08-28T00:01:00.000Z",
    },
  };
}
