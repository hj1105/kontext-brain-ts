import type {
  AccessControlList,
  AuthorizedEvidenceMetadata,
  AuthorizedKnowledgeGraphReader,
  ChunkRecord,
  EntityMentionRecord,
  EntityRecord,
  EvidenceRecord,
  FactEvent,
  FactObject,
  FactRecord,
  KnowledgeGraphRepository,
  KnowledgeGraphUnitOfWork,
  OntologyLinkRecord,
  Principal,
  ResourceRecord,
  ResourceSource,
} from "@kontext-brain/core";
import type { Pool, PoolClient, QueryResultRow } from "pg";
import { aclPredicate, toConnectionError, toIsoString } from "./postgres-value-utils.js";

type Queryable = Pick<PoolClient, "query">;

export class PostgresKnowledgeGraphRepository implements KnowledgeGraphRepository {
  constructor(private readonly pool: Pool) {}

  async transaction<T>(
    organizationId: string,
    work: (unitOfWork: KnowledgeGraphUnitOfWork) => Promise<T>,
  ): Promise<T> {
    return withOrganizationTransaction(this.pool, organizationId, async (client) => {
      return work(new PostgresKnowledgeGraphUnitOfWork(client, organizationId));
    });
  }

  async getResourceBySource(
    organizationId: string,
    source: ResourceSource,
  ): Promise<ResourceRecord | null> {
    return this.read(organizationId, (unit) => unit.getResourceBySource(source));
  }

  async getResource(organizationId: string, resourceId: string): Promise<ResourceRecord | null> {
    return this.read(organizationId, (unit) => unit.getResource(resourceId));
  }

  async listChunks(organizationId: string, resourceId: string): Promise<readonly ChunkRecord[]> {
    return this.read(organizationId, (unit) => unit.listChunks(resourceId));
  }

  async listEntitiesForResource(
    organizationId: string,
    resourceId: string,
  ): Promise<readonly EntityRecord[]> {
    return this.read(organizationId, (unit) => unit.listEntities(resourceId));
  }

  async listEntityMentions(
    organizationId: string,
    resourceId: string,
  ): Promise<readonly EntityMentionRecord[]> {
    return this.read(organizationId, (unit) => unit.listEntityMentions(resourceId));
  }

  async getFact(organizationId: string, factKey: string): Promise<FactRecord | null> {
    return this.read(organizationId, (unit) => unit.getFact(factKey));
  }

  async listFacts(organizationId: string): Promise<readonly FactRecord[]> {
    return this.read(organizationId, (unit) => unit.listFacts());
  }

  async listEvidenceForFact(
    organizationId: string,
    factKey: string,
  ): Promise<readonly EvidenceRecord[]> {
    return this.read(organizationId, (unit) => unit.listEvidenceForFact(factKey));
  }

  async listFactEvents(organizationId: string, factKey: string): Promise<readonly FactEvent[]> {
    return withOrganizationTransaction(
      this.pool,
      organizationId,
      async (client) => {
        const result = await client.query(
          `SELECT organization_id, fact_key, event_type, occurred_at, resource_id
           FROM kontext_fact_events
           WHERE organization_id = $1 AND fact_key = $2
           ORDER BY event_id`,
          [organizationId, factKey],
        );
        return result.rows.map(mapFactEvent);
      },
      { readOnly: true },
    );
  }

  private async read<T>(
    organizationId: string,
    work: (unitOfWork: KnowledgeGraphUnitOfWork) => Promise<T>,
  ): Promise<T> {
    return withOrganizationTransaction(
      this.pool,
      organizationId,
      (client) => work(new PostgresKnowledgeGraphUnitOfWork(client, organizationId)),
      { readOnly: true },
    );
  }
}

class PostgresKnowledgeGraphUnitOfWork implements KnowledgeGraphUnitOfWork {
  constructor(
    private readonly client: Queryable,
    private readonly organizationId: string,
  ) {}

  async getResourceBySource(source: ResourceSource): Promise<ResourceRecord | null> {
    const result = await this.client.query(
      `${resourceSelectSql()}
       WHERE r.organization_id = $1 AND r.connector_id = $2 AND r.external_id = $3`,
      [this.organizationId, source.connectorId, source.externalId],
    );
    return result.rows[0] ? mapResource(result.rows[0]) : null;
  }

  async getResource(resourceId: string): Promise<ResourceRecord | null> {
    const result = await this.client.query(
      `${resourceSelectSql()}
       WHERE r.organization_id = $1 AND r.resource_id = $2`,
      [this.organizationId, resourceId],
    );
    return result.rows[0] ? mapResource(result.rows[0]) : null;
  }

  async saveResource(resource: ResourceRecord): Promise<void> {
    assertOrganization(this.organizationId, resource.organizationId);
    await this.client.query(
      `INSERT INTO kontext_resources (
         organization_id, resource_id, connector_id, external_id, source_type, title,
         content_hash, content_object_key, acl, status, updated_at
       ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$11)
       ON CONFLICT (organization_id, resource_id) DO UPDATE SET
         connector_id = EXCLUDED.connector_id,
         external_id = EXCLUDED.external_id,
         source_type = EXCLUDED.source_type,
         title = EXCLUDED.title,
         content_hash = EXCLUDED.content_hash,
         content_object_key = EXCLUDED.content_object_key,
         acl = EXCLUDED.acl,
         status = EXCLUDED.status,
         updated_at = EXCLUDED.updated_at`,
      [
        resource.organizationId,
        resource.resourceId,
        resource.source.connectorId,
        resource.source.externalId,
        resource.source.type,
        resource.title,
        resource.contentHash,
        resource.contentObjectKey,
        JSON.stringify(resource.acl),
        resource.status,
        resource.updatedAt,
      ],
    );
    await this.client.query(
      `DELETE FROM kontext_resource_ontology_links
       WHERE organization_id = $1 AND resource_id = $2`,
      [resource.organizationId, resource.resourceId],
    );
    await insertOntologyLinks(
      this.client,
      "kontext_resource_ontology_links",
      "resource_id",
      resource.organizationId,
      resource.resourceId,
      resource.ontologyNodeIds,
      resource.ontologyLinks,
    );
  }

  async listChunks(resourceId: string): Promise<readonly ChunkRecord[]> {
    const result = await this.client.query(
      `${chunkSelectSql()}
       WHERE c.organization_id = $1 AND c.resource_id = $2
       ORDER BY c.position`,
      [this.organizationId, resourceId],
    );
    return result.rows.map(mapChunk);
  }

  async saveChunk(chunk: ChunkRecord): Promise<void> {
    assertOrganization(this.organizationId, chunk.organizationId);
    await this.client.query(
      `INSERT INTO kontext_chunks (
         organization_id, resource_id, chunk_id, source_chunk_id, content_hash,
         content_object_key, position, acl, status
       ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9)
       ON CONFLICT (organization_id, chunk_id) DO UPDATE SET
         resource_id = EXCLUDED.resource_id,
         source_chunk_id = EXCLUDED.source_chunk_id,
         content_hash = EXCLUDED.content_hash,
         content_object_key = EXCLUDED.content_object_key,
         position = EXCLUDED.position,
         acl = EXCLUDED.acl,
         status = EXCLUDED.status`,
      [
        chunk.organizationId,
        chunk.resourceId,
        chunk.chunkId,
        chunk.sourceChunkId,
        chunk.contentHash,
        chunk.contentObjectKey,
        chunk.position,
        JSON.stringify(chunk.acl),
        chunk.status,
      ],
    );
    await this.client.query(
      `DELETE FROM kontext_chunk_ontology_links
       WHERE organization_id = $1 AND chunk_id = $2`,
      [chunk.organizationId, chunk.chunkId],
    );
    await insertOntologyLinks(
      this.client,
      "kontext_chunk_ontology_links",
      "chunk_id",
      chunk.organizationId,
      chunk.chunkId,
      chunk.ontologyNodeIds,
      chunk.ontologyLinks,
    );
  }

  async saveEntity(entity: EntityRecord): Promise<void> {
    assertOrganization(this.organizationId, entity.organizationId);
    await this.client.query(
      `INSERT INTO kontext_entities (
         organization_id, entity_id, scope, resource_id, name, entity_type, status
       ) VALUES ($1,$2,$3,$4,$5,$6,$7)
       ON CONFLICT (organization_id, entity_id) DO UPDATE SET
         scope = EXCLUDED.scope,
         resource_id = EXCLUDED.resource_id,
         name = EXCLUDED.name,
         entity_type = EXCLUDED.entity_type,
         status = EXCLUDED.status`,
      [
        entity.organizationId,
        entity.entityId,
        entity.scope,
        entity.resourceId ?? null,
        entity.name,
        entity.type ?? null,
        entity.status,
      ],
    );
  }

  async listEntities(resourceId: string): Promise<readonly EntityRecord[]> {
    const result = await this.client.query(
      `SELECT organization_id, entity_id, scope, resource_id, name, entity_type, status
       FROM kontext_entities
       WHERE organization_id = $1 AND resource_id = $2`,
      [this.organizationId, resourceId],
    );
    return result.rows.map(mapEntity);
  }

  async listEntityMentions(resourceId: string): Promise<readonly EntityMentionRecord[]> {
    const result = await this.client.query(
      `SELECT organization_id, entity_id, resource_id, chunk_id, status,
              extraction_confidence, extractor_version, origin, observed_at
       FROM kontext_entity_mentions
       WHERE organization_id = $1 AND resource_id = $2`,
      [this.organizationId, resourceId],
    );
    return result.rows.map(mapEntityMention);
  }

  async saveEntityMention(mention: EntityMentionRecord): Promise<void> {
    assertOrganization(this.organizationId, mention.organizationId);
    await this.client.query(
      `INSERT INTO kontext_entity_mentions (
         organization_id, entity_id, resource_id, chunk_id, status,
         extraction_confidence, extractor_version, origin, observed_at
       ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
       ON CONFLICT (organization_id, entity_id, chunk_id) DO UPDATE SET
         resource_id = EXCLUDED.resource_id,
         status = EXCLUDED.status,
         extraction_confidence = EXCLUDED.extraction_confidence,
         extractor_version = EXCLUDED.extractor_version,
         origin = EXCLUDED.origin,
         observed_at = EXCLUDED.observed_at`,
      [
        mention.organizationId,
        mention.entityId,
        mention.resourceId,
        mention.chunkId,
        mention.status,
        mention.extractionConfidence ?? null,
        mention.extractorVersion ?? null,
        mention.origin ?? null,
        mention.observedAt ?? null,
      ],
    );
  }

  async getFact(factKey: string): Promise<FactRecord | null> {
    const result = await this.client.query(
      `SELECT organization_id, fact_key, subject, predicate, object, single_value, status,
              updated_at, extraction_confidence, extractor_version, origin, observed_at, verified_at
       FROM kontext_facts
       WHERE organization_id = $1 AND fact_key = $2`,
      [this.organizationId, factKey],
    );
    return result.rows[0] ? mapFact(result.rows[0]) : null;
  }

  async listFacts(): Promise<readonly FactRecord[]> {
    const result = await this.client.query(
      `SELECT organization_id, fact_key, subject, predicate, object, single_value, status,
              updated_at, extraction_confidence, extractor_version, origin, observed_at, verified_at
       FROM kontext_facts
       WHERE organization_id = $1`,
      [this.organizationId],
    );
    return result.rows.map(mapFact);
  }

  async saveFact(fact: FactRecord): Promise<void> {
    assertOrganization(this.organizationId, fact.organizationId);
    await this.client.query(
      `INSERT INTO kontext_facts (
         organization_id, fact_key, subject, predicate, object, single_value, status, updated_at,
         extraction_confidence, extractor_version, origin, observed_at, verified_at
       ) VALUES ($1,$2,$3::jsonb,$4,$5::jsonb,$6,$7,$8,$9,$10,$11,$12,$13)
       ON CONFLICT (organization_id, fact_key) DO UPDATE SET
         subject = EXCLUDED.subject,
         predicate = EXCLUDED.predicate,
         object = EXCLUDED.object,
         single_value = EXCLUDED.single_value,
         status = EXCLUDED.status,
         updated_at = EXCLUDED.updated_at,
         extraction_confidence = EXCLUDED.extraction_confidence,
         extractor_version = EXCLUDED.extractor_version,
         origin = EXCLUDED.origin,
         observed_at = EXCLUDED.observed_at,
         verified_at = EXCLUDED.verified_at`,
      [
        fact.organizationId,
        fact.factKey,
        JSON.stringify(fact.subject),
        fact.predicate,
        JSON.stringify(fact.object),
        fact.singleValue,
        fact.status,
        fact.updatedAt,
        fact.extractionConfidence ?? null,
        fact.extractorVersion ?? null,
        fact.origin ?? null,
        fact.observedAt ?? null,
        fact.verifiedAt ?? null,
      ],
    );
  }

  async listEvidenceForResource(resourceId: string): Promise<readonly EvidenceRecord[]> {
    const result = await this.client.query(
      `SELECT organization_id, evidence_id, fact_key, resource_id, chunk_id, acl, origin, status,
              confidence, observed_at, verified_at
       FROM kontext_evidence
       WHERE organization_id = $1 AND resource_id = $2`,
      [this.organizationId, resourceId],
    );
    return result.rows.map(mapEvidence);
  }

  async listEvidenceForFact(factKey: string): Promise<readonly EvidenceRecord[]> {
    const result = await this.client.query(
      `SELECT organization_id, evidence_id, fact_key, resource_id, chunk_id, acl, origin, status,
              confidence, observed_at, verified_at
       FROM kontext_evidence
       WHERE organization_id = $1 AND fact_key = $2`,
      [this.organizationId, factKey],
    );
    return result.rows.map(mapEvidence);
  }

  async saveEvidence(evidence: EvidenceRecord): Promise<void> {
    assertOrganization(this.organizationId, evidence.organizationId);
    await this.client.query(
      `INSERT INTO kontext_evidence (
         organization_id, evidence_id, fact_key, resource_id, chunk_id, acl, origin, status,
         confidence, observed_at, verified_at
       ) VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9,$10,$11)
       ON CONFLICT (organization_id, evidence_id) DO UPDATE SET
         fact_key = EXCLUDED.fact_key,
         resource_id = EXCLUDED.resource_id,
         chunk_id = EXCLUDED.chunk_id,
         acl = EXCLUDED.acl,
         origin = EXCLUDED.origin,
         status = EXCLUDED.status,
         confidence = EXCLUDED.confidence,
         observed_at = EXCLUDED.observed_at,
         verified_at = EXCLUDED.verified_at`,
      [
        evidence.organizationId,
        evidence.evidenceId,
        evidence.factKey,
        evidence.resourceId,
        evidence.chunkId,
        JSON.stringify(evidence.acl),
        evidence.origin,
        evidence.status,
        evidence.confidence ?? null,
        evidence.observedAt ?? null,
        evidence.verifiedAt ?? null,
      ],
    );
  }

  async appendFactEvent(event: FactEvent): Promise<void> {
    assertOrganization(this.organizationId, event.organizationId);
    await this.client.query(
      `INSERT INTO kontext_fact_events (
         organization_id, fact_key, event_type, occurred_at, resource_id
       ) VALUES ($1,$2,$3,$4,$5)`,
      [event.organizationId, event.factKey, event.type, event.occurredAt, event.resourceId ?? null],
    );
  }
}

export class PostgresAuthorizedKnowledgeGraphReader implements AuthorizedKnowledgeGraphReader {
  constructor(private readonly pool: Pool) {}

  async listAccessibleFactEvidence(
    principal: Principal,
    factKeys?: readonly string[],
  ): Promise<readonly AuthorizedEvidenceMetadata[]> {
    return withOrganizationTransaction(
      this.pool,
      principal.organizationId,
      async (client) => {
        const result = await client.query(
          `SELECT
           row_to_json(f) AS fact_record,
           row_to_json(e) AS evidence_record,
           row_to_json(r) AS resource_record,
           row_to_json(c) AS chunk_record,
           ARRAY(SELECT ontology_node_id FROM kontext_resource_ontology_links rl
                 WHERE rl.organization_id = r.organization_id AND rl.resource_id = r.resource_id
                 ORDER BY ontology_node_id) AS resource_ontology_node_ids,
           ARRAY(SELECT ontology_node_id FROM kontext_chunk_ontology_links cl
                 WHERE cl.organization_id = c.organization_id AND cl.chunk_id = c.chunk_id
                 ORDER BY ontology_node_id) AS chunk_ontology_node_ids
         FROM kontext_facts f
         JOIN kontext_evidence e
           ON e.organization_id = f.organization_id AND e.fact_key = f.fact_key
         JOIN kontext_resources r
           ON r.organization_id = e.organization_id AND r.resource_id = e.resource_id
         JOIN kontext_chunks c
           ON c.organization_id = e.organization_id AND c.chunk_id = e.chunk_id
         WHERE f.organization_id = $1
           AND f.status IN ('active', 'conflict')
           AND e.status = 'active' AND r.status = 'active' AND c.status = 'active'
           AND ($4::text[] IS NULL OR f.fact_key = ANY($4::text[]))
           AND ${aclPredicate("e")}
           AND ${aclPredicate("r")}
           AND ${aclPredicate("c")}
         ORDER BY f.fact_key, e.evidence_id`,
          [
            principal.organizationId,
            principal.subjectId,
            [...principal.groupIds],
            factKeys ? [...factKeys] : null,
          ],
        );
        return result.rows.map((row) => ({
          fact: mapFact(row.fact_record),
          evidence: mapEvidence(row.evidence_record),
          resource: mapResource({
            ...row.resource_record,
            ontology_node_ids: row.resource_ontology_node_ids,
          }),
          chunk: mapChunk({
            ...row.chunk_record,
            ontology_node_ids: row.chunk_ontology_node_ids,
          }),
        }));
      },
      { readOnly: true },
    );
  }
}

export interface OrganizationTransactionOptions {
  /**
   * Read-only work must not provision the organization row. Skipping the write
   * keeps retrieval usable on read replicas and prevents a mistyped/unknown
   * organizationId from being silently created by a mere lookup.
   */
  readonly readOnly?: boolean;
}

export async function withOrganizationTransaction<T>(
  pool: Pool,
  organizationId: string,
  work: (client: PoolClient) => Promise<T>,
  options: OrganizationTransactionOptions = {},
): Promise<T> {
  const client = await pool.connect();
  let released = false;
  try {
    await client.query(options.readOnly ? "BEGIN READ ONLY" : "BEGIN");
    if (!options.readOnly) {
      await client.query(
        `INSERT INTO kontext_organizations (organization_id) VALUES ($1)
         ON CONFLICT (organization_id) DO NOTHING`,
        [organizationId],
      );
    }
    await client.query("SELECT set_config('kontext.organization_id', $1, true)", [organizationId]);
    const result = await work(client);
    await client.query("COMMIT");
    return result;
  } catch (error) {
    try {
      await client.query("ROLLBACK");
    } catch (rollbackError) {
      released = true;
      client.release(toConnectionError(rollbackError));
      throw new AggregateError(
        [error, rollbackError],
        "PostgreSQL transaction and rollback both failed",
      );
    }
    throw error;
  } finally {
    if (!released) client.release();
  }
}

function resourceSelectSql(): string {
  return `SELECT r.*,
    ARRAY(SELECT ontology_node_id FROM kontext_resource_ontology_links l
          WHERE l.organization_id = r.organization_id AND l.resource_id = r.resource_id
          ORDER BY ontology_node_id) AS ontology_node_ids,
    COALESCE((SELECT jsonb_agg(jsonb_build_object(
      'ontologyNodeId', ontology_node_id,
      'origin', origin,
      'confidence', confidence,
      'createdAt', created_at
    ) ORDER BY ontology_node_id)
      FROM kontext_resource_ontology_links l
      WHERE l.organization_id = r.organization_id AND l.resource_id = r.resource_id), '[]'::jsonb)
      AS ontology_links
    FROM kontext_resources r`;
}

function chunkSelectSql(): string {
  return `SELECT c.*,
    ARRAY(SELECT ontology_node_id FROM kontext_chunk_ontology_links l
          WHERE l.organization_id = c.organization_id AND l.chunk_id = c.chunk_id
          ORDER BY ontology_node_id) AS ontology_node_ids,
    COALESCE((SELECT jsonb_agg(jsonb_build_object(
      'ontologyNodeId', ontology_node_id,
      'origin', origin,
      'confidence', confidence,
      'createdAt', created_at
    ) ORDER BY ontology_node_id)
      FROM kontext_chunk_ontology_links l
      WHERE l.organization_id = c.organization_id AND l.chunk_id = c.chunk_id), '[]'::jsonb)
      AS ontology_links
    FROM kontext_chunks c`;
}

async function insertOntologyLinks(
  client: Queryable,
  table: "kontext_resource_ontology_links" | "kontext_chunk_ontology_links",
  ownerColumn: "resource_id" | "chunk_id",
  organizationId: string,
  ownerId: string,
  ontologyNodeIds: readonly string[],
  ontologyLinks: ResourceRecord["ontologyLinks"] | ChunkRecord["ontologyLinks"],
): Promise<void> {
  const metadata = new Map((ontologyLinks ?? []).map((link) => [link.ontologyNodeId, link]));
  const links = [...new Set(ontologyNodeIds)].map((ontologyNodeId) => ({
    ontologyNodeId,
    origin: metadata.get(ontologyNodeId)?.origin ?? null,
    confidence: metadata.get(ontologyNodeId)?.confidence ?? null,
    createdAt: metadata.get(ontologyNodeId)?.createdAt ?? null,
  }));
  if (links.length === 0) return;
  await client.query(
    `INSERT INTO ${table} (
       organization_id, ${ownerColumn}, ontology_node_id, origin, confidence, created_at
     )
     SELECT $1, $2, item.ontology_node_id, item.origin, item.confidence, item.created_at
     FROM jsonb_to_recordset($3::jsonb) AS item(
       ontology_node_id text, origin text, confidence real, created_at timestamptz
     )
     ON CONFLICT (organization_id, ${ownerColumn}, ontology_node_id) DO UPDATE SET
       origin = EXCLUDED.origin,
       confidence = EXCLUDED.confidence,
       created_at = EXCLUDED.created_at`,
    [
      organizationId,
      ownerId,
      JSON.stringify(
        links.map((link) => ({
          ontology_node_id: link.ontologyNodeId,
          origin: link.origin,
          confidence: link.confidence,
          created_at: link.createdAt,
        })),
      ),
    ],
  );
}

function mapResource(row: QueryResultRow): ResourceRecord {
  return {
    organizationId: String(row.organization_id),
    resourceId: String(row.resource_id),
    source: {
      connectorId: String(row.connector_id),
      externalId: String(row.external_id),
      type: String(row.source_type),
    },
    title: String(row.title),
    contentHash: String(row.content_hash),
    contentObjectKey: String(row.content_object_key),
    acl: row.acl as AccessControlList,
    ontologyNodeIds: (row.ontology_node_ids ?? []) as string[],
    ontologyLinks: mapOntologyLinks(row.ontology_links),
    status: row.status as ResourceRecord["status"],
    updatedAt: toIsoString(row.updated_at),
  };
}

function mapChunk(row: QueryResultRow): ChunkRecord {
  return {
    organizationId: String(row.organization_id),
    resourceId: String(row.resource_id),
    chunkId: String(row.chunk_id),
    sourceChunkId: String(row.source_chunk_id),
    contentHash: String(row.content_hash),
    contentObjectKey: String(row.content_object_key),
    position: Number(row.position),
    acl: row.acl as AccessControlList,
    ontologyNodeIds: (row.ontology_node_ids ?? []) as string[],
    ontologyLinks: mapOntologyLinks(row.ontology_links),
    status: row.status as ChunkRecord["status"],
  };
}

function mapEntityMention(row: QueryResultRow): EntityMentionRecord {
  return {
    organizationId: String(row.organization_id),
    entityId: String(row.entity_id),
    resourceId: String(row.resource_id),
    chunkId: String(row.chunk_id),
    status: row.status as EntityMentionRecord["status"],
    extractionConfidence:
      row.extraction_confidence === null || row.extraction_confidence === undefined
        ? undefined
        : Number(row.extraction_confidence),
    extractorVersion:
      row.extractor_version === null || row.extractor_version === undefined
        ? undefined
        : String(row.extractor_version),
    origin:
      row.origin === null || row.origin === undefined
        ? undefined
        : (row.origin as EntityMentionRecord["origin"]),
    observedAt:
      row.observed_at === null || row.observed_at === undefined
        ? undefined
        : toIsoString(row.observed_at),
  };
}

function mapEntity(row: QueryResultRow): EntityRecord {
  return {
    organizationId: String(row.organization_id),
    entityId: String(row.entity_id),
    scope: row.scope as EntityRecord["scope"],
    resourceId: row.resource_id === null ? undefined : String(row.resource_id),
    name: String(row.name),
    type: row.entity_type === null ? undefined : String(row.entity_type),
    status: row.status as EntityRecord["status"],
  };
}

function mapFact(row: QueryResultRow): FactRecord {
  return {
    organizationId: String(row.organization_id),
    factKey: String(row.fact_key),
    subject: row.subject as FactRecord["subject"],
    predicate: String(row.predicate),
    object: row.object as FactObject,
    singleValue: Boolean(row.single_value),
    status: row.status as FactRecord["status"],
    updatedAt: toIsoString(row.updated_at),
    extractionConfidence:
      row.extraction_confidence === null || row.extraction_confidence === undefined
        ? undefined
        : Number(row.extraction_confidence),
    extractorVersion:
      row.extractor_version === null || row.extractor_version === undefined
        ? undefined
        : String(row.extractor_version),
    origin:
      row.origin === null || row.origin === undefined
        ? undefined
        : (row.origin as FactRecord["origin"]),
    observedAt:
      row.observed_at === null || row.observed_at === undefined
        ? undefined
        : toIsoString(row.observed_at),
    verifiedAt:
      row.verified_at === null || row.verified_at === undefined
        ? undefined
        : toIsoString(row.verified_at),
  };
}

function mapEvidence(row: QueryResultRow): EvidenceRecord {
  return {
    organizationId: String(row.organization_id),
    evidenceId: String(row.evidence_id),
    factKey: row.fact_key === null ? undefined : String(row.fact_key),
    resourceId: String(row.resource_id),
    chunkId: String(row.chunk_id),
    acl: row.acl as AccessControlList,
    origin: row.origin as EvidenceRecord["origin"],
    status: row.status as EvidenceRecord["status"],
    confidence:
      row.confidence === null || row.confidence === undefined ? undefined : Number(row.confidence),
    observedAt:
      row.observed_at === null || row.observed_at === undefined
        ? undefined
        : toIsoString(row.observed_at),
    verifiedAt:
      row.verified_at === null || row.verified_at === undefined
        ? undefined
        : toIsoString(row.verified_at),
  };
}

function mapOntologyLinks(value: unknown): OntologyLinkRecord[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const row = item as Record<string, unknown>;
    if (typeof row.ontologyNodeId !== "string") return [];
    return [
      {
        ontologyNodeId: row.ontologyNodeId,
        ...(row.origin === null || row.origin === undefined
          ? {}
          : { origin: row.origin as OntologyLinkRecord["origin"] }),
        ...(row.confidence === null || row.confidence === undefined
          ? {}
          : { confidence: Number(row.confidence) }),
        ...(row.createdAt === null || row.createdAt === undefined
          ? {}
          : { createdAt: toIsoString(row.createdAt) }),
      },
    ];
  });
}

function mapFactEvent(row: QueryResultRow): FactEvent {
  return {
    organizationId: String(row.organization_id),
    factKey: String(row.fact_key),
    type: row.event_type as FactEvent["type"],
    occurredAt: toIsoString(row.occurred_at),
    resourceId: row.resource_id === null ? undefined : String(row.resource_id),
  };
}

function assertOrganization(expected: string, received: string): void {
  if (expected !== received) {
    throw new Error(`Organization mismatch: expected "${expected}", received "${received}"`);
  }
}
