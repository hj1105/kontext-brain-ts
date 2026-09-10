import type { DatabaseSync } from "node:sqlite";
import type {
  ChunkRecord,
  EntityMentionRecord,
  EntityRecord,
  EvidenceRecord,
  FactEvent,
  FactRecord,
  ResourceRecord,
  ResourceSource,
} from "./domain.js";
import type { KnowledgeGraphUnitOfWork } from "./ports.js";
import {
  type SqliteKnowledgeRecords,
  sqliteFactEventSchema,
  sqliteKnowledgeSchemas,
} from "./sqlite-knowledge-records.js";

export class SqliteKnowledgeUnit implements KnowledgeGraphUnitOfWork {
  private active = true;
  constructor(
    private readonly db: DatabaseSync,
    private readonly organizationId: string,
  ) {}
  close(): void {
    this.active = false;
  }
  private assertActive(): void {
    if (!this.active) throw new Error("Knowledge transaction is closed");
  }
  private list<K extends keyof SqliteKnowledgeRecords>(
    kind: K,
    field?: "resource_id" | "fact_key" | "record_key",
    value?: string,
  ): SqliteKnowledgeRecords[K][] {
    this.assertActive();
    const sql = `SELECT payload FROM knowledge_records WHERE organization_id = ? AND kind = ?${field ? ` AND ${field} = ?` : ""} ORDER BY record_key`;
    const rows = this.db
      .prepare(sql)
      .all(...(field ? [this.organizationId, kind, value ?? ""] : [this.organizationId, kind]));
    return this.decode(kind, rows);
  }
  private decode<K extends keyof SqliteKnowledgeRecords>(
    kind: K,
    rows: readonly Record<string, unknown>[],
  ): SqliteKnowledgeRecords[K][] {
    return rows.map((row) => {
      if (typeof row.payload !== "string") throw new Error("Invalid persisted knowledge record");
      const record = sqliteKnowledgeSchemas[kind].parse(JSON.parse(row.payload));
      if (record.organizationId !== this.organizationId)
        throw new Error("Knowledge record belongs to another Organization");
      return record;
    });
  }
  private save<K extends keyof SqliteKnowledgeRecords>(
    kind: K,
    key: string,
    input: SqliteKnowledgeRecords[K],
  ): void {
    this.assertActive();
    const record = sqliteKnowledgeSchemas[kind].parse(input);
    if (record.organizationId !== this.organizationId)
      throw new Error("Knowledge record belongs to another Organization");
    const resourceId = "resourceId" in record ? (record.resourceId ?? null) : null;
    const factKey = "factKey" in record ? (record.factKey ?? null) : null;
    this.db
      .prepare(
        "INSERT INTO knowledge_records (organization_id, kind, record_key, resource_id, fact_key, payload) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(organization_id, kind, record_key) DO UPDATE SET resource_id=excluded.resource_id, fact_key=excluded.fact_key, payload=excluded.payload",
      )
      .run(this.organizationId, kind, key, resourceId, factKey, JSON.stringify(record));
  }
  async getResource(id: string): Promise<ResourceRecord | null> {
    return this.list("resource", "record_key", id)[0] ?? null;
  }
  async getResourceBySource(source: ResourceSource): Promise<ResourceRecord | null> {
    this.assertActive();
    const rows = this.db
      .prepare(
        "SELECT payload FROM knowledge_records WHERE organization_id = ? AND kind = 'resource' AND json_extract(payload, '$.source.connectorId') = ? AND json_extract(payload, '$.source.externalId') = ? ORDER BY record_key LIMIT 1",
      )
      .all(this.organizationId, source.connectorId, source.externalId);
    return this.decode("resource", rows)[0] ?? null;
  }
  async listResourcesByOntologyNode(id: string): Promise<readonly ResourceRecord[]> {
    this.assertActive();
    const rows = this.db
      .prepare(
        "SELECT payload FROM knowledge_records WHERE organization_id = ? AND kind = 'resource' AND EXISTS (SELECT 1 FROM json_each(payload, '$.ontologyNodeIds') WHERE value = ?) ORDER BY record_key",
      )
      .all(this.organizationId, id);
    return this.decode("resource", rows);
  }
  async saveResource(value: ResourceRecord): Promise<void> {
    this.save("resource", value.resourceId, value);
  }
  /** Every resource of the organization; the local search scans them all. */
  async listResources(): Promise<readonly ResourceRecord[]> {
    return this.list("resource");
  }
  async listChunks(id: string): Promise<readonly ChunkRecord[]> {
    return this.list("chunk", "resource_id", id).sort((a, b) => a.position - b.position);
  }
  async saveChunk(value: ChunkRecord): Promise<void> {
    this.save("chunk", value.chunkId, value);
  }
  async listEntities(id: string): Promise<readonly EntityRecord[]> {
    return this.list("entity", "resource_id", id);
  }
  async saveEntity(value: EntityRecord): Promise<void> {
    this.save("entity", value.entityId, value);
  }
  async listEntityMentions(id: string): Promise<readonly EntityMentionRecord[]> {
    return this.list("mention", "resource_id", id);
  }
  async saveEntityMention(value: EntityMentionRecord): Promise<void> {
    this.save("mention", JSON.stringify([value.entityId, value.chunkId]), value);
  }
  async getFact(id: string): Promise<FactRecord | null> {
    return this.list("fact", "record_key", id)[0] ?? null;
  }
  async listFacts(): Promise<readonly FactRecord[]> {
    return this.list("fact");
  }
  async saveFact(value: FactRecord): Promise<void> {
    this.save("fact", value.factKey, value);
  }
  async listEvidenceForResource(id: string): Promise<readonly EvidenceRecord[]> {
    return this.list("evidence", "resource_id", id);
  }
  async listEvidenceForFact(id: string): Promise<readonly EvidenceRecord[]> {
    return this.list("evidence", "fact_key", id);
  }
  async saveEvidence(value: EvidenceRecord): Promise<void> {
    this.save("evidence", value.evidenceId, value);
  }
  async appendFactEvent(value: FactEvent): Promise<void> {
    this.assertActive();
    const event = sqliteFactEventSchema.parse(value);
    if (event.organizationId !== this.organizationId)
      throw new Error("Knowledge event belongs to another Organization");
    this.db
      .prepare("INSERT INTO knowledge_events (organization_id, fact_key, payload) VALUES (?, ?, ?)")
      .run(this.organizationId, event.factKey, JSON.stringify(event));
  }
  async listFactEvents(factKey: string): Promise<readonly FactEvent[]> {
    this.assertActive();
    return this.db
      .prepare(
        "SELECT payload FROM knowledge_events WHERE organization_id = ? AND fact_key = ? ORDER BY sequence",
      )
      .all(this.organizationId, factKey)
      .map((row) => {
        if (typeof row.payload !== "string") throw new Error("Invalid persisted knowledge event");
        const value = sqliteFactEventSchema.parse(JSON.parse(row.payload));
        if (value.organizationId !== this.organizationId || value.factKey !== factKey)
          throw new Error("Knowledge event identity mismatch");
        return value;
      });
  }
}
