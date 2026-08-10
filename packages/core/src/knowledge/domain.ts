export type OrganizationId = string;

export interface Principal {
  readonly organizationId: OrganizationId;
  readonly subjectId: string;
  readonly groupIds: readonly string[];
}

export interface AccessControlList {
  readonly organizationWide?: boolean;
  readonly subjectIds?: readonly string[];
  readonly groupIds?: readonly string[];
}

export interface ResourceSource {
  readonly connectorId: string;
  readonly externalId: string;
  readonly type: string;
}

export type RecordStatus = "active" | "stale" | "purged";
export type FactStatus = "active" | "inactive" | "conflict";
export type EvidenceOrigin = "derived" | "curated";
export type EntityScope = "resource" | "global";

export interface ResourceRecord {
  readonly organizationId: OrganizationId;
  readonly resourceId: string;
  readonly source: ResourceSource;
  readonly title: string;
  readonly contentHash: string;
  readonly contentObjectKey: string;
  readonly acl: AccessControlList;
  readonly ontologyNodeIds: readonly string[];
  readonly status: RecordStatus;
  readonly updatedAt: string;
}

export interface ChunkRecord {
  readonly organizationId: OrganizationId;
  readonly resourceId: string;
  readonly chunkId: string;
  readonly sourceChunkId: string;
  readonly contentHash: string;
  readonly contentObjectKey: string;
  readonly position: number;
  readonly acl: AccessControlList;
  readonly ontologyNodeIds: readonly string[];
  readonly status: RecordStatus;
}

export interface EntityRef {
  readonly entityId: string;
  readonly scope: EntityScope;
}

export interface ExtractedEntity extends EntityRef {
  readonly name: string;
  readonly type?: string;
  readonly mentionChunkIds: readonly string[];
  readonly promotionEvidence?: "deterministic" | "manual" | "resolved";
}

export interface EntityRecord extends EntityRef {
  readonly organizationId: OrganizationId;
  readonly resourceId?: string;
  readonly name: string;
  readonly type?: string;
  readonly status: RecordStatus;
}

export interface EntityMentionRecord {
  readonly organizationId: OrganizationId;
  readonly entityId: string;
  readonly resourceId: string;
  readonly chunkId: string;
  readonly status: RecordStatus;
}

export type FactObject =
  | { readonly kind: "entity"; readonly entity: EntityRef }
  | { readonly kind: "literal"; readonly value: string | number | boolean };

export interface ExtractedFact {
  readonly factKey: string;
  readonly subject: EntityRef;
  readonly predicate: string;
  readonly object: FactObject;
  readonly evidenceChunkIds: readonly string[];
  readonly singleValue?: boolean;
}

export interface FactRecord {
  readonly organizationId: OrganizationId;
  readonly factKey: string;
  readonly subject: EntityRef;
  readonly predicate: string;
  readonly object: FactObject;
  readonly singleValue: boolean;
  readonly status: FactStatus;
  readonly updatedAt: string;
}

export type FactEventType =
  | "created"
  | "invalidated"
  | "restored"
  | "conflict_detected"
  | "conflict_resolved";

export interface FactEvent {
  readonly organizationId: OrganizationId;
  readonly factKey: string;
  readonly type: FactEventType;
  readonly occurredAt: string;
  readonly resourceId?: string;
}

export interface EvidenceRecord {
  readonly organizationId: OrganizationId;
  readonly evidenceId: string;
  readonly factKey?: string;
  readonly resourceId: string;
  readonly chunkId: string;
  readonly acl: AccessControlList;
  readonly origin: EvidenceOrigin;
  readonly status: RecordStatus;
}

export interface ResourceChunkSnapshot {
  readonly id: string;
  readonly contentHash: string;
  readonly text: string;
  readonly position: number;
  readonly ontologyNodeIds?: readonly string[];
  readonly acl?: AccessControlList;
}

export interface ResourceSnapshot {
  readonly organizationId: OrganizationId;
  readonly source: ResourceSource;
  readonly title: string;
  readonly contentHash: string;
  readonly body: string;
  readonly acl: AccessControlList;
  readonly ontologyNodeIds?: readonly string[];
  readonly chunks: readonly ResourceChunkSnapshot[];
  readonly entities?: readonly ExtractedEntity[];
  readonly facts?: readonly ExtractedFact[];
}

export function resourceIdentity(source: ResourceSource): string {
  return `${encodeURIComponent(source.connectorId)}:${encodeURIComponent(source.externalId)}`;
}

export function chunkIdentity(resourceId: string, sourceChunkId: string): string {
  return `${resourceId}#${encodeURIComponent(sourceChunkId)}`;
}

export function factSlotKey(fact: Pick<FactRecord, "subject" | "predicate">): string {
  return `${fact.subject.scope}:${fact.subject.entityId}:${fact.predicate}`;
}

export function factObjectKey(object: FactObject): string {
  return object.kind === "entity"
    ? `entity:${object.entity.scope}:${object.entity.entityId}`
    : `literal:${typeof object.value}:${String(object.value)}`;
}
