import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  InMemoryPreparedTaskContextStore,
  InMemoryTaskContextStateProvider,
  createContextReceipt,
} from "@kontext-brain/context";
import { FileQuarantineStore, FileTaskCompletionArtifactStore } from "@kontext-brain/local";
import {
  DurableVerificationCoordinator,
  InMemoryVerificationRetryQueue,
  VerificationCoordinator,
  VerifierRegistry,
  createFastVerificationPlan,
  createFullVerificationPlan,
  createTargetedVerificationPlan,
} from "@kontext-brain/orchestrator";
import {
  type LogicWorkItem,
  type TaskContextSnapshot,
  type TaskContract,
  computeTaskContextDigest,
  createChangeBundle,
} from "@kontext-brain/spec";
import { afterEach, describe, expect, it } from "vitest";
import {
  LocalKontextCompletionOperations,
  type SidecarChangeEvidenceProvider,
} from "../src/index.js";

const temporaryDirectories: string[] = [];
const contract: TaskContract = {
  taskId: "task:local-completion",
  intent: "Drive exact completion through sidecar-owned MCP operations.",
  acceptance: [
    {
      criterionId: "acceptance:worker",
      statement: "Worker verification passes.",
      verifier: { kind: "test", ref: "worker:test" },
    },
  ],
  nonGoals: [],
  targets: ["symbol:handler"],
  risk: "low",
};
const snapshotInput = {
  taskId: contract.taskId,
  baseCodeRevision: "commit:base",
  effectiveScopes: [{ kind: "personal" as const, subjectId: "user:local" }],
  normativeRevisions: [],
  requiredEvidenceIds: ["evidence:decision"],
  sourceFreshnessDigest: "freshness:current",
};
const snapshot: TaskContextSnapshot = {
  ...snapshotInput,
  contextDigest: computeTaskContextDigest(snapshotInput),
  createdAt: "2026-08-28T13:00:00.000Z",
};
const resultRevision = "commit:result";
const workItem: LogicWorkItem = {
  workItemId: "work-item:handler",
  taskId: contract.taskId,
  plannedSymbolIds: ["symbol:handler"],
  dependsOn: [],
  allowedPaths: ["src/handler.ts"],
  requiredVerifiers: [{ kind: "test", ref: "worker:test" }],
  capabilityId: "capability:handler",
};
const receipt = createContextReceipt({
  snapshot,
  logic: workItem,
  allowedPaths: workItem.allowedPaths,
  evidenceIds: snapshot.requiredEvidenceIds,
  issuedAt: "2026-08-28T13:00:00.000Z",
  expiresAt: "2026-08-28T14:00:00.000Z",
});

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("LocalKontextCompletionOperations", () => {
  it("checks, persists, accepts a handoff, and computes done without trusting reported context", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "kontext-local-completion-"));
    temporaryDirectories.push(directory);
    const current = new InMemoryTaskContextStateProvider();
    current.set(contract.taskId, {
      codeRevision: resultRevision,
      sourceFreshnessDigest: snapshot.sourceFreshnessDigest,
      effectiveScopes: snapshot.effectiveScopes,
      normativeRecords: [],
      normativeRevisionCatalog: [],
      conflicts: [],
      evidence: [
        {
          evidenceId: "evidence:decision",
          text: "The user approved the decision.",
          availability: "current",
          allowedRuntimeProviders: ["codex", "claude"],
        },
      ],
      logicPlans: [
        {
          workItemId: workItem.workItemId,
          plannedSymbolIds: workItem.plannedSymbolIds,
          allowedPaths: workItem.allowedPaths,
          dependsOn: workItem.dependsOn,
          requiredVerifiers: workItem.requiredVerifiers,
          capabilityId: workItem.capabilityId,
        },
      ],
    });
    const prepared = new InMemoryPreparedTaskContextStore();
    await prepared.put({ contract, snapshot, additionalRequiredEvidenceIds: [] });
    const registry = passingRegistry();
    const artifacts = new FileTaskCompletionArtifactStore(directory);
    const operations = new LocalKontextCompletionOperations(
      current,
      prepared,
      artifacts,
      new FileQuarantineStore(directory),
      new DurableVerificationCoordinator(
        new VerificationCoordinator(registry),
        new InMemoryVerificationRetryQueue(),
      ),
      fixedChangeEvidence(),
    );

    for (const tier of ["fast", "targeted", "full"] as const) {
      await operations.checkChange({
        taskId: contract.taskId,
        workItemId: workItem.workItemId,
        workspacePath: directory,
        tier,
        observedAt: `2026-08-28T13:0${tier === "fast" ? 1 : tier === "targeted" ? 2 : 3}:00.000Z`,
        nextAttemptAt: "2026-08-28T13:10:00.000Z",
      });
    }
    const runs = await artifacts.listVerificationRuns(contract.taskId);
    const bundleRuns = runs.filter((run) => run.tier !== "full");
    const bundle = createChangeBundle({
      taskId: contract.taskId,
      workItemId: workItem.workItemId,
      baseRevision: snapshot.baseCodeRevision,
      resultRevision,
      taskContextDigest: snapshot.contextDigest,
      patchDigest: "sha256:patch",
      changedSymbolIds: ["symbol:handler"],
      changedPaths: ["src/handler.ts"],
      contextReceiptIds: [receipt.receiptId],
      evidenceIds: snapshot.requiredEvidenceIds,
      normativeRevisions: snapshot.normativeRevisions,
      verificationRunIds: bundleRuns.map((run) => run.verificationRunId),
      proposals: [],
      unresolved: [],
      submittedAt: "2026-08-28T13:04:00.000Z",
    });
    const { bundleId: _bundleId, ...bundleDraft } = bundle;
    const submitted = (await operations.submitChangeBundle({
      workspacePath: directory,
      bundle: bundleDraft,
    })) as {
      readonly accepted: boolean;
      readonly issues: readonly unknown[];
      readonly bundle: { readonly bundleId: string };
    };
    expect(submitted).toMatchObject({
      accepted: true,
      issues: [],
      bundle,
      observedPatch: {
        patchDigest: bundle.patchDigest,
        changedPaths: bundle.changedPaths,
        changedSymbolIds: bundle.changedSymbolIds,
      },
    });

    const transition = (await operations.proposeTransition({
      taskId: contract.taskId,
      currentState: "in_progress",
      workStarted: true,
      completionRequested: true,
      context: { status: "stale", contextDigest: "caller:wrong" },
      evidence: [
        {
          kind: "commit",
          ref: resultRevision,
          codeRevision: resultRevision,
          contextDigest: snapshot.contextDigest,
          observedAt: "2026-08-28T13:05:00.000Z",
        },
      ],
      invariantEvaluations: [],
      reviewFindings: [],
      requestedAt: "2026-08-28T13:06:00.000Z",
    })) as {
      readonly state: string;
      readonly reportedContextMatched: boolean;
      readonly accuracyManifest?: { readonly manifestId: string };
    };

    expect(transition.state, JSON.stringify(transition)).toBe("done");
    expect(transition.reportedContextMatched).toBe(false);
    expect(transition.accuracyManifest?.manifestId).toMatch(/^accuracy-manifest:/);
    expect(await artifacts.getAccuracyManifest(contract.taskId)).toEqual(
      expect.objectContaining({ manifestId: transition.accuracyManifest?.manifestId }),
    );
  });

  it("binds Verification Runs to sidecar-observed revision and symbols", async () => {
    const current = new InMemoryTaskContextStateProvider();
    current.set(contract.taskId, {
      codeRevision: resultRevision,
      sourceFreshnessDigest: snapshot.sourceFreshnessDigest,
      effectiveScopes: snapshot.effectiveScopes,
      normativeRecords: [],
      normativeRevisionCatalog: [],
      conflicts: [],
      evidence: [],
      logicPlans: [workItem],
    });
    const prepared = new InMemoryPreparedTaskContextStore();
    await prepared.put({ contract, snapshot, additionalRequiredEvidenceIds: [] });
    const directory = await mkdtemp(path.join(tmpdir(), "kontext-local-completion-"));
    temporaryDirectories.push(directory);
    const operations = new LocalKontextCompletionOperations(
      current,
      prepared,
      new FileTaskCompletionArtifactStore(directory),
      new FileQuarantineStore(directory),
      new DurableVerificationCoordinator(
        new VerificationCoordinator(passingRegistry()),
        new InMemoryVerificationRetryQueue(),
      ),
      fixedChangeEvidence(),
    );

    const result = (await operations.checkChange({
      taskId: contract.taskId,
      workItemId: workItem.workItemId,
      workspacePath: directory,
      tier: "fast",
      observedAt: "2026-08-28T13:01:00.000Z",
      nextAttemptAt: "2026-08-28T13:10:00.000Z",
    })) as { readonly executions: readonly { readonly run: { readonly codeRevision: string } }[] };

    expect(result.executions.length).toBeGreaterThan(0);
    expect(
      result.executions.every((execution) => execution.run.codeRevision === resultRevision),
    ).toBe(true);
  });

  it("refuses to verify sidecar evidence that escapes the planned symbols", async () => {
    const current = new InMemoryTaskContextStateProvider();
    current.set(contract.taskId, {
      codeRevision: resultRevision,
      sourceFreshnessDigest: snapshot.sourceFreshnessDigest,
      effectiveScopes: snapshot.effectiveScopes,
      normativeRecords: [],
      normativeRevisionCatalog: [],
      conflicts: [],
      evidence: [],
      logicPlans: [workItem],
    });
    const prepared = new InMemoryPreparedTaskContextStore();
    await prepared.put({ contract, snapshot, additionalRequiredEvidenceIds: [] });
    const directory = await mkdtemp(path.join(tmpdir(), "kontext-local-completion-"));
    temporaryDirectories.push(directory);
    const operations = new LocalKontextCompletionOperations(
      current,
      prepared,
      new FileTaskCompletionArtifactStore(directory),
      new FileQuarantineStore(directory),
      new DurableVerificationCoordinator(
        new VerificationCoordinator(passingRegistry()),
        new InMemoryVerificationRetryQueue(),
      ),
      fixedChangeEvidence({ unauthorizedChangedSymbolIds: ["symbol:unplanned"] }),
    );

    await expect(
      operations.checkChange({
        taskId: contract.taskId,
        workItemId: workItem.workItemId,
        workspacePath: directory,
        tier: "fast",
        observedAt: "2026-08-28T13:01:00.000Z",
        nextAttemptAt: "2026-08-28T13:10:00.000Z",
      }),
    ).rejects.toThrow("Cannot verify out-of-scope Code Symbols: symbol:unplanned");
  });
});

function fixedChangeEvidence(
  overrides: Partial<Awaited<ReturnType<SidecarChangeEvidenceProvider["observe"]>>> = {},
): SidecarChangeEvidenceProvider {
  return {
    async observe() {
      return {
        currentCodeRevision: resultRevision,
        observedPatch: {
          patchDigest: "sha256:patch",
          changedPaths: ["src/handler.ts"],
          changedSymbolIds: ["symbol:handler"],
        },
        receipts: [receipt],
        plannedSymbolBindings: [
          {
            plannedSymbolId: "symbol:handler",
            symbolId: "symbol:handler",
            boundBy: "existing_symbol_id",
          },
        ],
        plannedSymbolIssues: [],
        unauthorizedChangedSymbolIds: [],
        ...overrides,
      };
    },
  };
}

function passingRegistry(): VerifierRegistry {
  const registry = new VerifierRegistry();
  const requirements = [
    ...createFastVerificationPlan({ affectedSymbolIds: workItem.plannedSymbolIds }).requirements,
    ...createTargetedVerificationPlan({ workItem }).requirements,
    ...createFullVerificationPlan({ contract }).requirements,
  ];
  const seen = new Set<string>();
  for (const requirement of requirements) {
    const key = `${requirement.verifier.kind}:${requirement.verifier.ref}`;
    if (seen.has(key)) continue;
    seen.add(key);
    registry.register(requirement.verifier, {
      execute: async () => ({ result: "passed", output: { verifier: key } }),
    });
  }
  return registry;
}
