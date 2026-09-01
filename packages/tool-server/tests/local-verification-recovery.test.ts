import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  InMemoryPreparedTaskContextStore,
  InMemoryTaskContextStateProvider,
} from "@kontext-brain/context";
import { FileTaskCompletionArtifactStore } from "@kontext-brain/local";
import {
  DurableVerificationCoordinator,
  InMemoryVerificationRetryQueue,
  VerificationCoordinator,
  VerifierInfrastructureError,
  VerifierRegistry,
} from "@kontext-brain/orchestrator";
import { computeTaskContextDigest } from "@kontext-brain/spec";
import { afterEach, describe, expect, it } from "vitest";
import { LocalVerificationRecoveryService } from "../src/index.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("LocalVerificationRecoveryService", () => {
  it("automatically revalidates queued proof for the same current Task revision after recovery", async () => {
    const taskId = "task:recovery";
    const snapshotInput = {
      taskId,
      baseCodeRevision: "commit:base",
      effectiveScopes: [{ kind: "personal" as const, subjectId: "user:local" }],
      normativeRevisions: [],
      requiredEvidenceIds: [],
      sourceFreshnessDigest: "freshness:current",
    };
    const snapshot = {
      ...snapshotInput,
      contextDigest: computeTaskContextDigest(snapshotInput),
      createdAt: "2026-08-28T15:00:00.000Z",
    };
    const contract = {
      taskId,
      intent: "Revalidate after infrastructure recovery.",
      acceptance: [
        {
          criterionId: "acceptance:query",
          statement: "Semantic sync passes.",
          verifier: { kind: "query" as const, ref: "kontext:semantic-sync" },
        },
      ],
      nonGoals: [],
      targets: ["symbol:handler"],
      risk: "low" as const,
    };
    let available = false;
    const registry = new VerifierRegistry();
    registry.register(contract.acceptance[0].verifier, {
      execute: async () => {
        if (!available) throw new VerifierInfrastructureError("semantic index offline");
        return { result: "passed", output: "recovered" };
      },
    });
    const queue = new InMemoryVerificationRetryQueue();
    const durable = new DurableVerificationCoordinator(
      new VerificationCoordinator(registry),
      queue,
    );
    await durable.executePlan({
      taskId,
      workItemId: "work-item:handler",
      plan: {
        tier: "fast",
        requirements: [
          {
            tier: "fast",
            verifier: contract.acceptance[0].verifier,
            subjectIds: contract.targets,
          },
        ],
      },
      binding: {
        workspacePath: "/workspace",
        codeRevision: "commit:result",
        contextDigest: snapshot.contextDigest,
        observedAt: "2026-08-28T15:01:00.000Z",
      },
      nextAttemptAt: "2026-08-28T15:02:00.000Z",
    });
    available = true;

    const current = new InMemoryTaskContextStateProvider();
    current.set(taskId, {
      codeRevision: "commit:result",
      sourceFreshnessDigest: snapshot.sourceFreshnessDigest,
      effectiveScopes: snapshot.effectiveScopes,
      normativeRecords: [],
      normativeRevisionCatalog: [],
      conflicts: [],
      evidence: [],
      logicPlans: [],
    });
    const prepared = new InMemoryPreparedTaskContextStore();
    await prepared.put({ contract, snapshot, additionalRequiredEvidenceIds: [] });
    const directory = await mkdtemp(path.join(tmpdir(), "kontext-recovery-"));
    temporaryDirectories.push(directory);
    const artifacts = new FileTaskCompletionArtifactStore(directory);
    const recovery = new LocalVerificationRecoveryService(
      current,
      prepared,
      artifacts,
      queue,
      durable,
    );
    const executions = await recovery.recoverAvailable({
      now: "2026-08-28T15:02:00.000Z",
      nextAttemptAt: "2026-08-28T15:03:00.000Z",
      leaseExpiresAt: "2026-08-28T15:04:00.000Z",
    });

    expect(executions[0]?.run).toEqual(
      expect.objectContaining({
        result: "passed",
        codeRevision: "commit:result",
        contextDigest: snapshot.contextDigest,
      }),
    );
    expect(await artifacts.listVerificationRuns(taskId)).toEqual([
      expect.objectContaining({ result: "passed" }),
    ]);
    expect(await queue.list("completed")).toHaveLength(1);
  });
});
