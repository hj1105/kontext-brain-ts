import {
  type TaskContract,
  computeTaskContextDigest,
  createAccuracyManifest,
  createChangeBundle,
  taskContractDigest,
} from "@kontext-brain/spec";
import type { Pool, PoolClient } from "pg";
import { describe, expect, it } from "vitest";
import { PostgresTaskCompletionProjection } from "../src/index.js";

const contract: TaskContract = {
  taskId: "task:managed-completion",
  intent: "Project immutable completion proof.",
  acceptance: [
    {
      criterionId: "acceptance:test",
      statement: "Tests pass.",
      verifier: { kind: "test", ref: "workspace:test" },
    },
  ],
  nonGoals: [],
  targets: ["symbol:managed"],
  risk: "low",
};
const snapshotInput = {
  taskId: contract.taskId,
  baseCodeRevision: "commit:base",
  effectiveScopes: [{ kind: "organization" as const, organizationId: "org:acme" }],
  normativeRevisions: [],
  requiredEvidenceIds: [],
  sourceFreshnessDigest: "freshness:current",
};
const snapshot = {
  ...snapshotInput,
  contextDigest: computeTaskContextDigest(snapshotInput),
  createdAt: "2026-08-28T14:00:00.000Z",
};
const run = {
  verificationRunId: "verification:managed",
  tier: "full" as const,
  verifierKind: "test" as const,
  verifierRef: "workspace:test",
  codeRevision: "commit:result",
  contextDigest: snapshot.contextDigest,
  subjectIds: [contract.taskId],
  result: "passed" as const,
  observedAt: "2026-08-28T14:01:00.000Z",
};
const bundle = createChangeBundle({
  taskId: contract.taskId,
  workItemId: "work-item:managed",
  baseRevision: snapshot.baseCodeRevision,
  resultRevision: run.codeRevision,
  taskContextDigest: snapshot.contextDigest,
  patchDigest: "sha256:patch",
  changedSymbolIds: contract.targets,
  changedPaths: ["src/managed.ts"],
  contextReceiptIds: ["receipt:managed"],
  evidenceIds: [],
  normativeRevisions: [],
  verificationRunIds: [run.verificationRunId],
  proposals: [],
  unresolved: [],
  submittedAt: "2026-08-28T14:02:00.000Z",
});
const manifest = createAccuracyManifest({
  taskId: contract.taskId,
  taskContractDigest: taskContractDigest(contract),
  contextDigest: snapshot.contextDigest,
  baseCodeRevision: snapshot.baseCodeRevision,
  resultCodeRevision: run.codeRevision,
  normativeRevisions: [],
  evidenceIds: [],
  workItemIds: [bundle.workItemId],
  changeBundleIds: [bundle.bundleId],
  changedSymbolIds: bundle.changedSymbolIds,
  verificationRunIds: [run.verificationRunId],
  reviewFindingIds: [],
  emergencyBypassIds: [],
  createdAt: "2026-08-28T14:03:00.000Z",
});

describe("PostgresTaskCompletionProjection", () => {
  it("projects Contract and Snapshot before immutable runs, bundles, and manifest", async () => {
    const queries: string[] = [];
    const client = {
      async query(sql: string) {
        queries.push(sql);
        return { rowCount: 1, rows: [] };
      },
      release() {},
    } as unknown as PoolClient;
    const projection = new PostgresTaskCompletionProjection({
      async connect() {
        return client;
      },
    } as unknown as Pool);

    await projection.project({
      organizationId: "org:acme",
      contract,
      snapshot,
      verificationRuns: [run],
      changeBundles: [bundle],
      accuracyManifest: manifest,
      projectedAt: "2026-08-28T14:04:00.000Z",
    });

    const taskIndex = queries.findIndex((sql) => sql.includes("INSERT INTO kontext_tasks"));
    const runIndex = queries.findIndex((sql) =>
      sql.includes("INSERT INTO kontext_verification_runs"),
    );
    const bundleIndex = queries.findIndex((sql) =>
      sql.includes("INSERT INTO kontext_change_bundles"),
    );
    const manifestIndex = queries.findIndex((sql) =>
      sql.includes("INSERT INTO kontext_accuracy_manifests"),
    );
    expect(taskIndex).toBeGreaterThan(-1);
    expect(runIndex).toBeGreaterThan(taskIndex);
    expect(bundleIndex).toBeGreaterThan(runIndex);
    expect(manifestIndex).toBeGreaterThan(bundleIndex);
    expect(queries.at(-1)).toBe("COMMIT");
  });

  it("rejects a tampered immutable Change Bundle before opening a transaction", async () => {
    let connected = false;
    const projection = new PostgresTaskCompletionProjection({
      async connect() {
        connected = true;
        throw new Error("must not connect");
      },
    } as unknown as Pool);

    await expect(
      projection.project({
        organizationId: "org:acme",
        contract,
        snapshot,
        verificationRuns: [run],
        changeBundles: [{ ...bundle, resultRevision: "commit:tampered" }],
      }),
    ).rejects.toThrow("Invalid Change Bundle");
    expect(connected).toBe(false);
  });
});
