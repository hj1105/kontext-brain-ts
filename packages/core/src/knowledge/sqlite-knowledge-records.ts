import { z } from "zod";
import type {
  ChunkRecord,
  EntityMentionRecord,
  EntityRecord,
  EvidenceRecord,
  FactEvent,
  FactRecord,
  ResourceRecord,
} from "./domain.js";

export interface SqliteKnowledgeRecords {
  resource: ResourceRecord;
  chunk: ChunkRecord;
  entity: EntityRecord;
  mention: EntityMentionRecord;
  fact: FactRecord;
  evidence: EvidenceRecord;
}
const id = z.string().min(1);
const status = z.enum(["active", "stale", "purged"]);
const origin = z.enum(["derived", "curated"]);
const acl = z
  .object({
    organizationWide: z.boolean().optional(),
    subjectIds: z.array(id).optional(),
    groupIds: z.array(id).optional(),
  })
  .strict();
const entityRef = z.object({ entityId: id, scope: z.enum(["resource", "global"]) }).strict();
const extraction = {
  extractionConfidence: z.number().optional(),
  extractorVersion: id.optional(),
  origin: origin.optional(),
  observedAt: id.optional(),
};
const ontologyLinks = z
  .array(
    z
      .object({
        ontologyNodeId: id,
        origin: z.enum(["manual", "automatic", "deterministic"]).optional(),
        confidence: z.number().optional(),
        createdAt: id.optional(),
      })
      .passthrough(),
  )
  .optional();
export const sqliteKnowledgeSchemas: {
  [K in keyof SqliteKnowledgeRecords]: z.ZodType<SqliteKnowledgeRecords[K]>;
} = {
  resource: z
    .object({
      organizationId: id,
      resourceId: id,
      source: z.object({ connectorId: id, externalId: id, type: id }).strict(),
      title: z.string(),
      contentHash: id,
      contentObjectKey: id,
      acl,
      ontologyNodeIds: z.array(id),
      ontologyLinks,
      status,
      updatedAt: id,
    })
    .passthrough(),
  chunk: z
    .object({
      organizationId: id,
      resourceId: id,
      chunkId: id,
      sourceChunkId: id,
      contentHash: id,
      contentObjectKey: id,
      position: z.number().int().nonnegative(),
      acl,
      ontologyNodeIds: z.array(id),
      ontologyLinks,
      status,
    })
    .passthrough(),
  entity: z
    .object({
      organizationId: id,
      resourceId: id.optional(),
      ...entityRef.shape,
      name: z.string(),
      type: id.optional(),
      status,
    })
    .passthrough(),
  mention: z
    .object({
      organizationId: id,
      resourceId: id,
      entityId: id,
      chunkId: id,
      status,
      ...extraction,
    })
    .passthrough(),
  fact: z
    .object({
      organizationId: id,
      factKey: id,
      subject: entityRef,
      predicate: id,
      object: z.discriminatedUnion("kind", [
        z.object({ kind: z.literal("entity"), entity: entityRef }).strict(),
        z
          .object({
            kind: z.literal("literal"),
            value: z.union([z.string(), z.number(), z.boolean()]),
          })
          .strict(),
      ]),
      singleValue: z.boolean(),
      status: z.enum(["active", "inactive", "conflict"]),
      updatedAt: id,
      ...extraction,
      verifiedAt: id.optional(),
    })
    .passthrough(),
  evidence: z
    .object({
      organizationId: id,
      evidenceId: id,
      resourceId: id,
      chunkId: id,
      factKey: id.optional(),
      acl,
      origin,
      status,
      confidence: z.number().optional(),
      observedAt: id.optional(),
      verifiedAt: id.optional(),
    })
    .passthrough(),
};
export const sqliteFactEventSchema: z.ZodType<FactEvent> = z
  .object({
    organizationId: id,
    factKey: id,
    type: z.enum(["created", "invalidated", "restored", "conflict_detected", "conflict_resolved"]),
    occurredAt: id,
    resourceId: id.optional(),
  })
  .passthrough();
