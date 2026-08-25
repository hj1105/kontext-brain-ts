import type { Principal, SearchGraphSession, SearchSeed } from "@kontext-brain/core";
import type { Pool, PoolClient } from "pg";
import { withOrganizationTransaction } from "./postgres-knowledge-graph.js";
import type { SearchSeedProvider } from "./postgres-knowledge-search-graph.js";
import { runPostgresSearchRead } from "./postgres-search-session.js";
import { aclPredicate } from "./postgres-value-utils.js";

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

  async seed(
    question: string,
    principal: Principal,
    session?: SearchGraphSession,
  ): Promise<readonly SearchSeed[]> {
    const embedding = Array.from(await this.embeddingProvider.embed(question));
    this.assertDimensions(embedding);
    const search = async (client: PoolClient): Promise<readonly SearchSeed[]> => {
      const result = await client.query(
        `WITH matches AS (
           SELECT c.chunk_id, 1 - (c.embedding <=> $4::vector) AS similarity
           FROM kontext_chunks c
           JOIN kontext_resources r
             ON r.organization_id = c.organization_id AND r.resource_id = c.resource_id
           WHERE c.organization_id = $1 AND c.status = 'active' AND r.status = 'active'
             AND c.embedding IS NOT NULL
             AND ${aclPredicate("c")} AND ${aclPredicate("r")}
         )
         SELECT chunk_id, similarity,
                row_number() OVER (ORDER BY similarity DESC, chunk_id) AS retrieval_rank,
                count(*) OVER () AS candidate_count
         FROM matches
         ORDER BY similarity DESC, chunk_id
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
        observations: {
          providers: ["postgres-chunk-vector"],
          query: {
            vector: {
              rank: finitePositiveInteger(row.retrieval_rank),
              candidateCount: finitePositiveInteger(row.candidate_count),
              normalizedScore: normalizedSimilarity(row.similarity),
            },
          },
        },
      }));
    };
    return runPostgresSearchRead(this.pool, principal.organizationId, session, search);
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

function vectorLiteral(values: readonly number[]): string {
  return `[${values.join(",")}]`;
}

function finitePositiveInteger(value: unknown): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.max(1, Math.floor(numeric)) : 1;
}

function normalizedSimilarity(value: unknown): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(0, Math.min(1, numeric));
}
