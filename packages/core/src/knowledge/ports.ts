import type { ResourceSnapshotEnricher } from "./adaptive-knowledge-enricher.js";
import type {
  ChunkRecord,
  EntityMentionRecord,
  EntityRecord,
  EvidenceRecord,
  FactEvent,
  FactRecord,
  OrganizationId,
  Principal,
  ResourceRecord,
  ResourceSnapshot,
  ResourceSource,
} from "./domain.js";

export interface KnowledgeGraphUnitOfWork {
  getResourceBySource(source: ResourceSource): Promise<ResourceRecord | null>;
  getResource(resourceId: string): Promise<ResourceRecord | null>;
  saveResource(resource: ResourceRecord): Promise<void>;
  listChunks(resourceId: string): Promise<readonly ChunkRecord[]>;
  saveChunk(chunk: ChunkRecord): Promise<void>;
  saveEntity(entity: EntityRecord): Promise<void>;
  listEntities(resourceId: string): Promise<readonly EntityRecord[]>;
  listEntityMentions(resourceId: string): Promise<readonly EntityMentionRecord[]>;
  saveEntityMention(mention: EntityMentionRecord): Promise<void>;
  getFact(factKey: string): Promise<FactRecord | null>;
  listFacts(): Promise<readonly FactRecord[]>;
  saveFact(fact: FactRecord): Promise<void>;
  listEvidenceForResource(resourceId: string): Promise<readonly EvidenceRecord[]>;
  listEvidenceForFact(factKey: string): Promise<readonly EvidenceRecord[]>;
  saveEvidence(evidence: EvidenceRecord): Promise<void>;
  appendFactEvent(event: FactEvent): Promise<void>;
}

export interface KnowledgeGraphRepository {
  transaction<T>(
    organizationId: OrganizationId,
    work: (unitOfWork: KnowledgeGraphUnitOfWork) => Promise<T>,
  ): Promise<T>;
  getResourceBySource(
    organizationId: OrganizationId,
    source: ResourceSource,
  ): Promise<ResourceRecord | null>;
  getResource(organizationId: OrganizationId, resourceId: string): Promise<ResourceRecord | null>;
  listChunks(organizationId: OrganizationId, resourceId: string): Promise<readonly ChunkRecord[]>;
  listEntitiesForResource(
    organizationId: OrganizationId,
    resourceId: string,
  ): Promise<readonly EntityRecord[]>;
  listEntityMentions(
    organizationId: OrganizationId,
    resourceId: string,
  ): Promise<readonly EntityMentionRecord[]>;
  getFact(organizationId: OrganizationId, factKey: string): Promise<FactRecord | null>;
  listFacts(organizationId: OrganizationId): Promise<readonly FactRecord[]>;
  listEvidenceForFact(
    organizationId: OrganizationId,
    factKey: string,
  ): Promise<readonly EvidenceRecord[]>;
  listFactEvents(organizationId: OrganizationId, factKey: string): Promise<readonly FactEvent[]>;
}

export interface StoredResourceContent {
  readonly organizationId: OrganizationId;
  readonly resourceId: string;
  readonly contentHash: string;
  readonly body: string;
  readonly chunks: Readonly<Record<string, string>>;
}

export interface ResourceContentStore {
  put(content: StoredResourceContent): Promise<string>;
  get(objectKey: string): Promise<StoredResourceContent | null>;
  purge(objectKey: string): Promise<void>;
}

export interface AuthorizedEvidenceMetadata {
  readonly fact: FactRecord;
  readonly evidence: EvidenceRecord;
  readonly resource: ResourceRecord;
  readonly chunk: ChunkRecord;
}

/** Implementations must apply organization and ACL predicates before returning rows. */
export interface AuthorizedKnowledgeGraphReader {
  listAccessibleFactEvidence(
    principal: Principal,
    factKeys?: readonly string[],
  ): Promise<readonly AuthorizedEvidenceMetadata[]>;
}

export interface Clock {
  now(): Date;
}

export const SystemClock: Clock = {
  now: () => new Date(),
};

export interface ResourceSyncResult {
  readonly resourceId: string;
  readonly changed: boolean;
  readonly affectedFactKeys: readonly string[];
}

export interface ResourceSyncOptions {
  /** Re-run an enricher even when the source bytes have not changed. */
  readonly forceReenrich?: boolean;
}

export interface ResourceSyncUseCase {
  execute(
    snapshot: ResourceSnapshot,
    snapshotEnricher?: ResourceSnapshotEnricher,
    options?: ResourceSyncOptions,
  ): Promise<ResourceSyncResult>;
  remove(organizationId: OrganizationId, source: ResourceSource): Promise<boolean>;
}
