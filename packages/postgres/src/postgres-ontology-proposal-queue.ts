import {
  type OntologyProposal,
  type OntologyProposalDraft,
  type OntologyProposalQueue,
  normalizeProposalKey,
} from "@kontext-brain/core";
import type { Pool, QueryResultRow } from "pg";
import { withOrganizationTransaction } from "./postgres-knowledge-graph.js";
import { toIsoString } from "./postgres-value-utils.js";

export class PostgresOntologyProposalQueue implements OntologyProposalQueue {
  constructor(private readonly pool: Pool) {}

  async enqueue(
    organizationId: string,
    proposals: readonly OntologyProposalDraft[],
  ): Promise<void> {
    await withOrganizationTransaction(this.pool, organizationId, async (client) => {
      for (const proposal of proposals) {
        const proposalKey = normalizeProposalKey(proposal.suggestedNodeId);
        if (!proposalKey) continue;
        await client.query(
          `INSERT INTO kontext_ontology_proposals (
             organization_id, proposal_key, suggested_node_id, description,
             resource_ids, occurrences, status
           ) VALUES ($1,$2,$3,$4,$5::text[],1,'open')
           ON CONFLICT (organization_id, proposal_key) DO UPDATE SET
             description = EXCLUDED.description,
             resource_ids = ARRAY(
               SELECT DISTINCT value
               FROM unnest(
                 kontext_ontology_proposals.resource_ids || EXCLUDED.resource_ids
               ) value
               ORDER BY value
             ),
             occurrences = kontext_ontology_proposals.occurrences + 1,
             -- Preserve the existing lifecycle status. Re-observing an unmapped node
             -- must not resurrect a proposal that is already published or decided.
             updated_at = now()`,
          [
            organizationId,
            proposalKey,
            proposal.suggestedNodeId,
            proposal.description,
            [...new Set(proposal.resourceIds)],
          ],
        );
      }
    });
  }

  async listOpen(organizationId: string): Promise<readonly OntologyProposal[]> {
    return withOrganizationTransaction(
      this.pool,
      organizationId,
      async (client) => {
        const result = await client.query(
          `SELECT * FROM kontext_ontology_proposals
           WHERE organization_id = $1 AND status = 'open'
           ORDER BY occurrences DESC, proposal_key`,
          [organizationId],
        );
        return result.rows.map(mapProposal);
      },
      { readOnly: true },
    );
  }

  async listPending(organizationId: string): Promise<readonly OntologyProposal[]> {
    return withOrganizationTransaction(
      this.pool,
      organizationId,
      async (client) => {
        const result = await client.query(
          `SELECT * FROM kontext_ontology_proposals
           WHERE organization_id = $1 AND status IN ('open', 'published')
           ORDER BY occurrences DESC, proposal_key`,
          [organizationId],
        );
        return result.rows.map(mapProposal);
      },
      { readOnly: true },
    );
  }

  async markPublished(organizationId: string, proposalKeys: readonly string[]): Promise<void> {
    if (proposalKeys.length === 0) return;
    await withOrganizationTransaction(this.pool, organizationId, async (client) => {
      await client.query(
        `UPDATE kontext_ontology_proposals
         SET status = 'published', updated_at = now()
         WHERE organization_id = $1 AND proposal_key = ANY($2::text[]) AND status = 'open'`,
        [organizationId, [...proposalKeys]],
      );
    });
  }
}

function mapProposal(row: QueryResultRow): OntologyProposal {
  return {
    organizationId: String(row.organization_id),
    proposalKey: String(row.proposal_key),
    suggestedNodeId: String(row.suggested_node_id),
    description: String(row.description),
    resourceIds: row.resource_ids as string[],
    occurrences: Number(row.occurrences),
    status: row.status as OntologyProposal["status"],
    updatedAt: toIsoString(row.updated_at),
  };
}
