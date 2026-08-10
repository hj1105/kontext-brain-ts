import type { OntologyDeployment, OntologyDeploymentRepository } from "@kontext-brain/core";
import type { Pool, QueryResultRow } from "pg";
import { withOrganizationTransaction } from "./postgres-knowledge-graph.js";

export class PostgresOntologyDeploymentRepository<TGraph>
  implements OntologyDeploymentRepository<TGraph>
{
  constructor(private readonly pool: Pool) {}

  async getActive(organizationId: string): Promise<OntologyDeployment<TGraph> | null> {
    return withOrganizationTransaction(this.pool, organizationId, async (client) => {
      const result = await client.query(
        `SELECT d.*
         FROM kontext_organization_runtime runtime
         JOIN kontext_ontology_deployments d
           ON d.organization_id = runtime.organization_id
          AND d.content_hash = runtime.active_ontology_hash
         WHERE runtime.organization_id = $1`,
        [organizationId],
      );
      return result.rows[0] ? mapDeployment<TGraph>(result.rows[0]) : null;
    });
  }

  async stage(candidate: OntologyDeployment<TGraph>): Promise<void> {
    await withOrganizationTransaction(this.pool, candidate.organizationId, async (client) => {
      await client.query(
        `INSERT INTO kontext_ontology_deployments (
           organization_id, content_hash, yaml, graph_data, git_commit, status, created_at
         ) VALUES ($1,$2,$3,$4::jsonb,$5,'staged',$6)
         ON CONFLICT (organization_id, content_hash) DO UPDATE SET
           yaml = EXCLUDED.yaml,
           graph_data = EXCLUDED.graph_data,
           git_commit = EXCLUDED.git_commit,
           status = 'staged',
           failure = NULL,
           created_at = EXCLUDED.created_at`,
        [
          candidate.organizationId,
          candidate.contentHash,
          candidate.yaml,
          JSON.stringify(candidate.graph),
          candidate.gitCommit ?? null,
          candidate.createdAt,
        ],
      );
    });
  }

  async activate(organizationId: string, contentHash: string): Promise<OntologyDeployment<TGraph>> {
    return withOrganizationTransaction(this.pool, organizationId, async (client) => {
      const locked = await client.query(
        `SELECT content_hash, status
         FROM kontext_ontology_deployments
         WHERE organization_id = $1 AND content_hash = $2
         FOR UPDATE`,
        [organizationId, contentHash],
      );
      if (locked.rows[0]?.status !== "staged") {
        throw new Error(`Ontology candidate "${contentHash}" is not staged`);
      }
      await client.query(
        `UPDATE kontext_ontology_deployments
         SET status = 'retired'
         WHERE organization_id = $1 AND status = 'active' AND content_hash <> $2`,
        [organizationId, contentHash],
      );
      const activated = await client.query(
        `UPDATE kontext_ontology_deployments
         SET status = 'active', failure = NULL
         WHERE organization_id = $1 AND content_hash = $2
         RETURNING *`,
        [organizationId, contentHash],
      );
      await client.query(
        `INSERT INTO kontext_organization_runtime (organization_id, active_ontology_hash)
         VALUES ($1,$2)
         ON CONFLICT (organization_id) DO UPDATE SET active_ontology_hash = EXCLUDED.active_ontology_hash`,
        [organizationId, contentHash],
      );
      const active = activated.rows[0];
      if (!active) throw new Error(`Ontology candidate "${contentHash}" was not activated`);
      return mapDeployment<TGraph>(active);
    });
  }

  async markFailed(organizationId: string, contentHash: string, failure: string): Promise<void> {
    await withOrganizationTransaction(this.pool, organizationId, async (client) => {
      await client.query(
        `UPDATE kontext_ontology_deployments
         SET status = 'failed', failure = $3
         WHERE organization_id = $1 AND content_hash = $2 AND status <> 'active'`,
        [organizationId, contentHash, failure],
      );
    });
  }
}

function mapDeployment<TGraph>(row: QueryResultRow): OntologyDeployment<TGraph> {
  return {
    organizationId: String(row.organization_id),
    contentHash: String(row.content_hash),
    yaml: String(row.yaml),
    graph: row.graph_data as TGraph,
    gitCommit: row.git_commit === null ? undefined : String(row.git_commit),
    status: row.status as OntologyDeployment<TGraph>["status"],
    createdAt:
      row.created_at instanceof Date
        ? row.created_at.toISOString()
        : new Date(String(row.created_at)).toISOString(),
    failure: row.failure === null ? undefined : String(row.failure),
  };
}
