import type { ExtractionJob, ExtractionJobKey, ExtractionJobQueue } from "@kontext-brain/core";
import type { Pool, QueryResultRow } from "pg";
import { withOrganizationTransaction } from "./postgres-knowledge-graph.js";
import { toIsoString } from "./postgres-value-utils.js";

export class PostgresExtractionJobQueue implements ExtractionJobQueue {
  constructor(private readonly pool: Pool) {}

  async enqueue(key: ExtractionJobKey): Promise<boolean> {
    return withOrganizationTransaction(this.pool, key.organizationId, async (client) => {
      const result = await client.query(
        `INSERT INTO kontext_extraction_jobs (
           organization_id, resource_id, content_hash, ontology_hash, state
         ) VALUES ($1,$2,$3,$4,'pending')
         ON CONFLICT (organization_id, resource_id, content_hash, ontology_hash) DO NOTHING
         RETURNING organization_id`,
        [key.organizationId, key.resourceId, key.contentHash, key.ontologyHash],
      );
      return result.rowCount === 1;
    });
  }

  async claim(
    organizationId: string,
    workerId: string,
    limit: number,
    leaseMs: number,
  ): Promise<readonly ExtractionJob[]> {
    return withOrganizationTransaction(this.pool, organizationId, async (client) => {
      const result = await client.query(
        `WITH claimable AS (
           SELECT organization_id, resource_id, content_hash, ontology_hash
           FROM kontext_extraction_jobs
           WHERE organization_id = $1
             AND state IN ('pending', 'failed')
             AND available_at <= now()
             AND (locked_until IS NULL OR locked_until < now())
           ORDER BY available_at, created_at
           FOR UPDATE SKIP LOCKED
           LIMIT $3
         )
         UPDATE kontext_extraction_jobs jobs
         SET state = 'running',
             attempts = jobs.attempts + 1,
             locked_by = $2,
             locked_until = now() + ($4::text || ' milliseconds')::interval,
             updated_at = now()
         FROM claimable
         WHERE jobs.organization_id = claimable.organization_id
           AND jobs.resource_id = claimable.resource_id
           AND jobs.content_hash = claimable.content_hash
           AND jobs.ontology_hash = claimable.ontology_hash
         RETURNING jobs.*`,
        [organizationId, workerId, Math.max(1, limit), Math.max(1000, leaseMs)],
      );
      return result.rows.map(mapJob);
    });
  }

  async succeed(key: ExtractionJobKey, workerId: string): Promise<void> {
    await withOrganizationTransaction(this.pool, key.organizationId, async (client) => {
      const result = await client.query(
        `UPDATE kontext_extraction_jobs
         SET state = 'succeeded', locked_by = NULL, locked_until = NULL, updated_at = now()
         WHERE organization_id = $1 AND resource_id = $2 AND content_hash = $3
           AND ontology_hash = $4 AND state = 'running' AND locked_by = $5`,
        [key.organizationId, key.resourceId, key.contentHash, key.ontologyHash, workerId],
      );
      if (result.rowCount !== 1) throw new Error("Extraction job lease is not owned by worker");
    });
  }

  async fail(
    key: ExtractionJobKey,
    workerId: string,
    error: string,
    retryAt: Date | null,
    maxAttempts: number,
  ): Promise<void> {
    await withOrganizationTransaction(this.pool, key.organizationId, async (client) => {
      const result = await client.query(
        `UPDATE kontext_extraction_jobs
         SET state = CASE WHEN attempts >= $7 THEN 'dead' ELSE 'failed' END,
             available_at = COALESCE($6, now()),
             locked_by = NULL,
             locked_until = NULL,
             last_error = $5,
             updated_at = now()
         WHERE organization_id = $1 AND resource_id = $2 AND content_hash = $3
           AND ontology_hash = $4 AND state = 'running' AND locked_by = $8`,
        [
          key.organizationId,
          key.resourceId,
          key.contentHash,
          key.ontologyHash,
          error,
          retryAt?.toISOString() ?? null,
          Math.max(1, maxAttempts),
          workerId,
        ],
      );
      if (result.rowCount !== 1) throw new Error("Extraction job lease is not owned by worker");
    });
  }
}

function mapJob(row: QueryResultRow): ExtractionJob {
  return {
    organizationId: String(row.organization_id),
    resourceId: String(row.resource_id),
    contentHash: String(row.content_hash),
    ontologyHash: String(row.ontology_hash),
    state: row.state as ExtractionJob["state"],
    attempts: Number(row.attempts),
    availableAt: toIsoString(row.available_at),
    lockedBy: row.locked_by === null ? undefined : String(row.locked_by),
    lockedUntil: row.locked_until === null ? undefined : toIsoString(row.locked_until),
    lastError: row.last_error === null ? undefined : String(row.last_error),
  };
}
