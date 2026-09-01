import {
  type AccuracyManifest,
  type ChangeBundle,
  type TaskContextSnapshot,
  type TaskContract,
  type VerificationRun,
  isAccuracyManifestValid,
  isChangeBundleValid,
  isTaskContextSnapshotValid,
  taskContractDigest,
  validateTaskContract,
} from "@kontext-brain/spec";
import type { Pool, PoolClient } from "pg";
import { withOrganizationTransaction } from "./postgres-knowledge-graph.js";

export interface ProjectTaskCompletionInput {
  readonly organizationId: string;
  readonly contract: TaskContract;
  readonly snapshot: TaskContextSnapshot;
  readonly verificationRuns: readonly VerificationRun[];
  readonly changeBundles: readonly ChangeBundle[];
  readonly accuracyManifest?: AccuracyManifest;
  readonly projectedAt?: string;
}

export class PostgresTaskCompletionProjection {
  constructor(private readonly pool: Pool) {}

  async project(input: ProjectTaskCompletionInput): Promise<void> {
    validateInput(input);
    const projectedAt = input.projectedAt ?? new Date().toISOString();
    await withOrganizationTransaction(this.pool, input.organizationId, async (client) => {
      await insertImmutable(
        client,
        `INSERT INTO kontext_tasks (
           organization_id, task_id, task_contract_digest, task_contract_data, created_at
         ) VALUES ($1,$2,$3,$4::jsonb,$5)
         ON CONFLICT (organization_id, task_id) DO UPDATE SET
           created_at = kontext_tasks.created_at
         WHERE kontext_tasks.task_contract_digest = EXCLUDED.task_contract_digest
           AND kontext_tasks.task_contract_data = EXCLUDED.task_contract_data
         RETURNING task_id`,
        [
          input.organizationId,
          input.contract.taskId,
          taskContractDigest(input.contract),
          JSON.stringify(input.contract),
          projectedAt,
        ],
        `Task Contract ${input.contract.taskId}`,
      );
      await insertImmutable(
        client,
        `INSERT INTO kontext_task_context_snapshots (
           organization_id, task_id, context_digest, base_code_revision, snapshot_data, created_at
         ) VALUES ($1,$2,$3,$4,$5::jsonb,$6)
         ON CONFLICT (organization_id, task_id, context_digest) DO UPDATE SET
           created_at = kontext_task_context_snapshots.created_at
         WHERE kontext_task_context_snapshots.snapshot_data = EXCLUDED.snapshot_data
         RETURNING context_digest`,
        [
          input.organizationId,
          input.contract.taskId,
          input.snapshot.contextDigest,
          input.snapshot.baseCodeRevision,
          JSON.stringify(input.snapshot),
          input.snapshot.createdAt,
        ],
        `Task Context Snapshot ${input.snapshot.contextDigest}`,
      );
      for (const run of input.verificationRuns) {
        await insertVerificationRun(client, input, run);
      }
      for (const bundle of input.changeBundles) {
        await insertChangeBundle(client, input.organizationId, bundle);
      }
      if (input.accuracyManifest) {
        await insertAccuracyManifest(client, input.organizationId, input.accuracyManifest);
      }
    });
  }
}

async function insertVerificationRun(
  client: PoolClient,
  input: ProjectTaskCompletionInput,
  run: VerificationRun,
): Promise<void> {
  await insertImmutable(
    client,
    `INSERT INTO kontext_verification_runs (
       organization_id, task_id, verification_run_id, tier, verifier_kind, verifier_ref,
       code_revision, context_digest, result, run_data, observed_at
     ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11)
     ON CONFLICT (organization_id, verification_run_id) DO UPDATE SET
       observed_at = kontext_verification_runs.observed_at
     WHERE kontext_verification_runs.run_data = EXCLUDED.run_data
     RETURNING verification_run_id`,
    [
      input.organizationId,
      input.contract.taskId,
      run.verificationRunId,
      run.tier,
      run.verifierKind,
      run.verifierRef,
      run.codeRevision,
      run.contextDigest,
      run.result,
      JSON.stringify(run),
      run.observedAt,
    ],
    `Verification Run ${run.verificationRunId}`,
  );
}

async function insertChangeBundle(
  client: PoolClient,
  organizationId: string,
  bundle: ChangeBundle,
): Promise<void> {
  await insertImmutable(
    client,
    `INSERT INTO kontext_change_bundles (
       organization_id, task_id, bundle_id, work_item_id, base_revision,
       result_revision, context_digest, bundle_data, submitted_at
     ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9)
     ON CONFLICT (organization_id, bundle_id) DO UPDATE SET
       submitted_at = kontext_change_bundles.submitted_at
     WHERE kontext_change_bundles.bundle_data = EXCLUDED.bundle_data
     RETURNING bundle_id`,
    [
      organizationId,
      bundle.taskId,
      bundle.bundleId,
      bundle.workItemId,
      bundle.baseRevision,
      bundle.resultRevision,
      bundle.taskContextDigest,
      JSON.stringify(bundle),
      bundle.submittedAt,
    ],
    `Change Bundle ${bundle.bundleId}`,
  );
}

async function insertAccuracyManifest(
  client: PoolClient,
  organizationId: string,
  manifest: AccuracyManifest,
): Promise<void> {
  await insertImmutable(
    client,
    `INSERT INTO kontext_accuracy_manifests (
       organization_id, task_id, manifest_id, result_code_revision,
       context_digest, manifest_data, created_at
     ) VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7)
     ON CONFLICT (organization_id, manifest_id) DO UPDATE SET
       created_at = kontext_accuracy_manifests.created_at
     WHERE kontext_accuracy_manifests.manifest_data = EXCLUDED.manifest_data
     RETURNING manifest_id`,
    [
      organizationId,
      manifest.taskId,
      manifest.manifestId,
      manifest.resultCodeRevision,
      manifest.contextDigest,
      JSON.stringify(manifest),
      manifest.createdAt,
    ],
    `Accuracy Manifest ${manifest.manifestId}`,
  );
}

async function insertImmutable(
  client: PoolClient,
  sql: string,
  values: readonly unknown[],
  label: string,
): Promise<void> {
  const result = await client.query(sql, [...values]);
  if (result.rowCount !== 1) throw new Error(`Immutable ${label} collision`);
}

function validateInput(input: ProjectTaskCompletionInput): void {
  if (!input.organizationId.trim()) throw new Error("Organization ID is required");
  const contractIssues = validateTaskContract(input.contract);
  if (contractIssues.length > 0) throw new Error("Invalid Task Contract");
  if (
    !isTaskContextSnapshotValid(input.snapshot) ||
    input.snapshot.taskId !== input.contract.taskId
  ) {
    throw new Error("Invalid Task Context Snapshot");
  }
  for (const bundle of input.changeBundles) {
    if (!isChangeBundleValid(bundle) || bundle.taskId !== input.contract.taskId) {
      throw new Error(`Invalid Change Bundle ${bundle.bundleId}`);
    }
  }
  if (
    input.accuracyManifest &&
    (!isAccuracyManifestValid(input.accuracyManifest) ||
      input.accuracyManifest.taskId !== input.contract.taskId)
  ) {
    throw new Error("Invalid Accuracy Manifest");
  }
}
