import type { AccessPolicy } from "./access-policy.js";
import type {
  ChunkRecord,
  EntityMentionRecord,
  EntityRecord,
  EvidenceRecord,
  FactEvent,
  FactRecord,
  OrganizationId,
  ResourceRecord,
  ResourceSource,
} from "./domain.js";
import { resourceIdentity } from "./domain.js";
import type { Principal } from "./domain.js";
import type {
  AuthorizedEvidenceMetadata,
  AuthorizedKnowledgeGraphReader,
  KnowledgeGraphRepository,
  KnowledgeGraphUnitOfWork,
  ResourceContentStore,
  StoredResourceContent,
} from "./ports.js";

interface OrganizationState {
  readonly resources: Map<string, ResourceRecord>;
  readonly chunks: Map<string, ChunkRecord>;
  readonly entities: Map<string, EntityRecord>;
  readonly mentions: Map<string, EntityMentionRecord>;
  readonly facts: Map<string, FactRecord>;
  readonly evidence: Map<string, EvidenceRecord>;
  readonly events: FactEvent[];
}

function emptyState(): OrganizationState {
  return {
    resources: new Map(),
    chunks: new Map(),
    entities: new Map(),
    mentions: new Map(),
    facts: new Map(),
    evidence: new Map(),
    events: [],
  };
}

function copyState(state: OrganizationState): OrganizationState {
  return {
    resources: new Map(state.resources),
    chunks: new Map(state.chunks),
    entities: new Map(state.entities),
    mentions: new Map(state.mentions),
    facts: new Map(state.facts),
    evidence: new Map(state.evidence),
    events: [...state.events],
  };
}

function sourceKey(source: ResourceSource): string {
  return resourceIdentity(source);
}

class InMemoryUnitOfWork implements KnowledgeGraphUnitOfWork {
  constructor(private readonly state: OrganizationState) {}

  async getResourceBySource(source: ResourceSource): Promise<ResourceRecord | null> {
    const key = sourceKey(source);
    return (
      Array.from(this.state.resources.values()).find((item) => sourceKey(item.source) === key) ??
      null
    );
  }

  async getResource(resourceId: string): Promise<ResourceRecord | null> {
    return this.state.resources.get(resourceId) ?? null;
  }

  async saveResource(resource: ResourceRecord): Promise<void> {
    this.state.resources.set(resource.resourceId, resource);
  }

  async listChunks(resourceId: string): Promise<readonly ChunkRecord[]> {
    return Array.from(this.state.chunks.values()).filter((item) => item.resourceId === resourceId);
  }

  async saveChunk(chunk: ChunkRecord): Promise<void> {
    this.state.chunks.set(chunk.chunkId, chunk);
  }

  async saveEntity(entity: EntityRecord): Promise<void> {
    this.state.entities.set(entity.entityId, entity);
  }

  async listEntityMentions(resourceId: string): Promise<readonly EntityMentionRecord[]> {
    return Array.from(this.state.mentions.values()).filter(
      (item) => item.resourceId === resourceId,
    );
  }

  async saveEntityMention(mention: EntityMentionRecord): Promise<void> {
    this.state.mentions.set(`${mention.entityId}:${mention.chunkId}`, mention);
  }

  async getFact(factKey: string): Promise<FactRecord | null> {
    return this.state.facts.get(factKey) ?? null;
  }

  async listFacts(): Promise<readonly FactRecord[]> {
    return Array.from(this.state.facts.values());
  }

  async saveFact(fact: FactRecord): Promise<void> {
    this.state.facts.set(fact.factKey, fact);
  }

  async listEvidenceForResource(resourceId: string): Promise<readonly EvidenceRecord[]> {
    return Array.from(this.state.evidence.values()).filter(
      (item) => item.resourceId === resourceId,
    );
  }

  async listEvidenceForFact(factKey: string): Promise<readonly EvidenceRecord[]> {
    return Array.from(this.state.evidence.values()).filter((item) => item.factKey === factKey);
  }

  async saveEvidence(evidence: EvidenceRecord): Promise<void> {
    this.state.evidence.set(evidence.evidenceId, evidence);
  }

  async appendFactEvent(event: FactEvent): Promise<void> {
    this.state.events.push(event);
  }
}

export class InMemoryKnowledgeGraphRepository implements KnowledgeGraphRepository {
  private readonly organizations = new Map<OrganizationId, OrganizationState>();
  private queues = new Map<OrganizationId, Promise<void>>();

  async transaction<T>(
    organizationId: OrganizationId,
    work: (unitOfWork: KnowledgeGraphUnitOfWork) => Promise<T>,
  ): Promise<T> {
    const previous = this.queues.get(organizationId) ?? Promise.resolve();
    let release: () => void = () => undefined;
    const current = new Promise<void>((resolve) => {
      release = resolve;
    });
    const queued = previous.then(() => current);
    this.queues.set(organizationId, queued);
    await previous;
    try {
      const candidate = copyState(this.stateFor(organizationId));
      const result = await work(new InMemoryUnitOfWork(candidate));
      this.organizations.set(organizationId, candidate);
      return result;
    } finally {
      release();
      if (this.queues.get(organizationId) === queued) this.queues.delete(organizationId);
    }
  }

  async getResourceBySource(
    organizationId: OrganizationId,
    source: ResourceSource,
  ): Promise<ResourceRecord | null> {
    return new InMemoryUnitOfWork(this.stateFor(organizationId)).getResourceBySource(source);
  }

  async getResource(
    organizationId: OrganizationId,
    resourceId: string,
  ): Promise<ResourceRecord | null> {
    return this.stateFor(organizationId).resources.get(resourceId) ?? null;
  }

  async listChunks(
    organizationId: OrganizationId,
    resourceId: string,
  ): Promise<readonly ChunkRecord[]> {
    return Array.from(this.stateFor(organizationId).chunks.values())
      .filter((item) => item.resourceId === resourceId)
      .sort((left, right) => left.position - right.position);
  }

  async getFact(organizationId: OrganizationId, factKey: string): Promise<FactRecord | null> {
    return this.stateFor(organizationId).facts.get(factKey) ?? null;
  }

  async listFacts(organizationId: OrganizationId): Promise<readonly FactRecord[]> {
    return Array.from(this.stateFor(organizationId).facts.values());
  }

  async listEvidenceForFact(
    organizationId: OrganizationId,
    factKey: string,
  ): Promise<readonly EvidenceRecord[]> {
    return Array.from(this.stateFor(organizationId).evidence.values()).filter(
      (item) => item.factKey === factKey,
    );
  }

  async listFactEvents(
    organizationId: OrganizationId,
    factKey: string,
  ): Promise<readonly FactEvent[]> {
    return this.stateFor(organizationId).events.filter((item) => item.factKey === factKey);
  }

  private stateFor(organizationId: OrganizationId): OrganizationState {
    let state = this.organizations.get(organizationId);
    if (!state) {
      state = emptyState();
      this.organizations.set(organizationId, state);
    }
    return state;
  }
}

export class InMemoryResourceContentStore implements ResourceContentStore {
  private readonly objects = new Map<string, StoredResourceContent>();
  putCount = 0;
  getCount = 0;

  async put(content: StoredResourceContent): Promise<string> {
    const objectKey = `${encodeURIComponent(content.organizationId)}/${content.resourceId}/${content.contentHash}.json`;
    this.objects.set(objectKey, content);
    this.putCount++;
    return objectKey;
  }

  async get(objectKey: string): Promise<StoredResourceContent | null> {
    this.getCount++;
    return this.objects.get(objectKey) ?? null;
  }

  async purge(objectKey: string): Promise<void> {
    this.objects.delete(objectKey);
  }
}

export class InMemoryAuthorizedKnowledgeGraphReader implements AuthorizedKnowledgeGraphReader {
  constructor(
    private readonly repository: KnowledgeGraphRepository,
    private readonly accessPolicy: AccessPolicy,
  ) {}

  async listAccessibleFactEvidence(
    principal: Principal,
    factKeys?: readonly string[],
  ): Promise<readonly AuthorizedEvidenceMetadata[]> {
    const requested = factKeys ? new Set(factKeys) : null;
    const output: AuthorizedEvidenceMetadata[] = [];
    const facts = await this.repository.listFacts(principal.organizationId);
    for (const fact of facts) {
      if (fact.status === "inactive" || (requested && !requested.has(fact.factKey))) continue;
      const evidenceItems = await this.repository.listEvidenceForFact(
        principal.organizationId,
        fact.factKey,
      );
      for (const evidence of evidenceItems) {
        if (evidence.status !== "active" || !this.accessPolicy.canAccess(principal, evidence.acl)) {
          continue;
        }
        const resource = await this.repository.getResource(
          principal.organizationId,
          evidence.resourceId,
        );
        if (
          !resource ||
          resource.status !== "active" ||
          !this.accessPolicy.canAccess(principal, resource.acl)
        ) {
          continue;
        }
        const chunk = (
          await this.repository.listChunks(principal.organizationId, evidence.resourceId)
        ).find((item) => item.chunkId === evidence.chunkId);
        if (
          !chunk ||
          chunk.status !== "active" ||
          !this.accessPolicy.canAccess(principal, chunk.acl)
        ) {
          continue;
        }
        output.push({ fact, evidence, resource, chunk });
      }
    }
    return output;
  }
}
