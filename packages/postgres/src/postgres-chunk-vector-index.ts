import type { Principal, SearchSeed } from "@kontext-brain/core";
import type { Pool } from "pg";
import { withOrganizationTransaction } from "./postgres-knowledge-graph.js";
import type { SearchSeedProvider } from "./postgres-knowledge-search-graph.js";

export interface QuestionEmbeddingProvider {
  embed(text: string): Promise<readonly number[] | Float32Array>;
}

export class PostgresChunkVectorIndex implements SearchSeedProvider {
  constructor(
    private readonly pool: Pool,
    private readonly embeddingProvider: QuestionEmbeddingProvider,
    private readonly dimensions = 1536,
    private readonly limit = 20,
  ) {}

  async upsert(
    organizationId: string,
    chunkId: string,
    embedding: readonly number[],
  ): Promise<void> {
    this.assertDimensions(embedding);
    await withOrganizationTransaction(this.pool, organizationId, async (client) => {
      const result = await client.query(
        `UPDATE kontext_chunks SET embedding = $3::vector
         WHERE organization_id = $1 AND chunk_id = $2 AND status = 'active'`,
        [organizationId, chunkId, vectorLiteral(embedding)],
      );
      if (result.rowCount !== 1) throw new Error(`Active Chunk "${chunkId}" was not found`);
    });
  }

  async seed(question: string, principal: Principal): Promise<readonly SearchSeed[]> {
    const embedding = Array.from(await this.embeddingProvider.embed(question));
    this.assertDimensions(embedding);
    return withOrganizationTransaction(this.pool, principal.organizationId, async (client) => {
      const result = await client.query(
        `SELECT c.chunk_id, 1 - (c.embedding <=> $4::vector) AS score
         FROM kontext_chunks c
         JOIN kontext_resources r
           ON r.organization_id = c.organization_id AND r.resource_id = c.resource_id
         WHERE c.organization_id = $1 AND c.status = 'active' AND r.status = 'active'
           AND c.embedding IS NOT NULL
           AND ${aclPredicate("c")} AND ${aclPredicate("r")}
         ORDER BY c.embedding <=> $4::vector
         LIMIT $5`,
        [
          principal.organizationId,
          principal.subjectId,
          [...principal.groupIds],
          vectorLiteral(embedding),
          this.limit,
        ],
      );
      return result.rows.map((row) => ({
        node: { kind: "chunk" as const, id: String(row.chunk_id) },
        score: Math.max(0, Math.min(1, Number(row.score))),
      }));
    });
  }

  private assertDimensions(embedding: readonly number[]): void {
    if (embedding.length !== this.dimensions) {
      throw new Error(
        `Expected ${this.dimensions} embedding dimensions, received ${embedding.length}`,
      );
    }
    if (embedding.some((value) => !Number.isFinite(value))) {
      throw new Error("Embedding contains a non-finite value");
    }
  }
}

function aclPredicate(alias: string): string {
  return `(COALESCE((${alias}.acl->>'organizationWide')::boolean, false)
    OR COALESCE(${alias}.acl->'subjectIds', '[]'::jsonb) ? $2
    OR EXISTS (
      SELECT 1 FROM jsonb_array_elements_text(COALESCE(${alias}.acl->'groupIds', '[]'::jsonb)) g(id)
      WHERE g.id = ANY($3::text[])
    ))`;
}

function vectorLiteral(values: readonly number[]): string {
  return `[${values.join(",")}]`;
}
