import type { ResourceSnapshotEnricher } from "./adaptive-knowledge-enricher.js";
import type {
  EntityRecord,
  EvidenceRecord,
  ExtractedEntity,
  FactEventType,
  FactRecord,
  FactStatus,
  ResourceSnapshot,
} from "./domain.js";
import { chunkIdentity, factObjectKey, factSlotKey, resourceIdentity } from "./domain.js";
import type {
  Clock,
  KnowledgeGraphRepository,
  KnowledgeGraphUnitOfWork,
  ResourceContentStore,
  ResourceSyncResult,
  ResourceSyncUseCase,
} from "./ports.js";
import { SystemClock } from "./ports.js";

export class SyncResourceUseCase implements ResourceSyncUseCase {
  constructor(
    private readonly repository: KnowledgeGraphRepository,
    private readonly contentStore: ResourceContentStore,
    private readonly clock: Clock = SystemClock,
  ) {}

  async execute(
    incomingSnapshot: ResourceSnapshot,
    snapshotEnricher?: ResourceSnapshotEnricher,
  ): Promise<ResourceSyncResult> {
    const existing = await this.repository.getResourceBySource(
      incomingSnapshot.organizationId,
      incomingSnapshot.source,
    );
    const resourceId = existing?.resourceId ?? resourceIdentity(incomingSnapshot.source);
    if (existing?.contentHash === incomingSnapshot.contentHash && existing.status === "active") {
      return { resourceId, changed: false, affectedFactKeys: [] };
    }
    const priorEntities = existing
      ? await this.loadPriorResourceEntities(incomingSnapshot.organizationId, resourceId)
      : [];
    const snapshot = snapshotEnricher
      ? (await snapshotEnricher.enrich(incomingSnapshot, priorEntities)).snapshot
      : incomingSnapshot;

    const objectKey = await this.contentStore.put({
      organizationId: snapshot.organizationId,
      resourceId,
      contentHash: snapshot.contentHash,
      body: snapshot.body,
      chunks: Object.fromEntries(snapshot.chunks.map((chunk) => [chunk.id, chunk.text])),
    });

    try {
      return await this.repository.transaction(snapshot.organizationId, async (unitOfWork) => {
        const now = this.clock.now().toISOString();
        const affectedFactKeys = new Set<string>();

        for (const chunk of await unitOfWork.listChunks(resourceId)) {
          if (chunk.status === "active") await unitOfWork.saveChunk({ ...chunk, status: "stale" });
        }
        for (const mention of await unitOfWork.listEntityMentions(resourceId)) {
          if (mention.status === "active") {
            await unitOfWork.saveEntityMention({ ...mention, status: "stale" });
          }
        }
        for (const evidence of await unitOfWork.listEvidenceForResource(resourceId)) {
          if (evidence.factKey) affectedFactKeys.add(evidence.factKey);
          if (evidence.origin === "derived" && evidence.status === "active") {
            await unitOfWork.saveEvidence({ ...evidence, status: "stale" });
          }
        }

        await unitOfWork.saveResource({
          organizationId: snapshot.organizationId,
          resourceId,
          source: snapshot.source,
          title: snapshot.title,
          contentHash: snapshot.contentHash,
          contentObjectKey: objectKey,
          acl: snapshot.acl,
          ontologyNodeIds: unique(snapshot.ontologyNodeIds ?? []),
          status: "active",
          updatedAt: now,
        });

        const chunkIds = new Map<string, string>();
        for (const chunk of snapshot.chunks) {
          const chunkId = chunkIdentity(resourceId, chunk.id);
          chunkIds.set(chunk.id, chunkId);
          await unitOfWork.saveChunk({
            organizationId: snapshot.organizationId,
            resourceId,
            chunkId,
            sourceChunkId: chunk.id,
            contentHash: chunk.contentHash,
            contentObjectKey: objectKey,
            position: chunk.position,
            acl: chunk.acl ?? snapshot.acl,
            ontologyNodeIds: unique(chunk.ontologyNodeIds ?? []),
            status: "active",
          });
          await unitOfWork.saveEvidence({
            organizationId: snapshot.organizationId,
            evidenceId: `${resourceId}|source|${chunkId}`,
            resourceId,
            chunkId,
            acl: chunk.acl ?? snapshot.acl,
            origin: "derived",
            status: "active",
          });
        }

        await this.replaceEntityMentions(unitOfWork, snapshot, resourceId, chunkIds);

        for (const extracted of snapshot.facts ?? []) {
          if (extracted.evidenceChunkIds.length === 0) {
            throw new Error(`Fact "${extracted.factKey}" requires explicit Evidence`);
          }
          affectedFactKeys.add(extracted.factKey);
          const previous = await unitOfWork.getFact(extracted.factKey);
          const subject = normalizeEntityRef(extracted.subject, resourceId);
          const object =
            extracted.object.kind === "entity"
              ? {
                  kind: "entity" as const,
                  entity: normalizeEntityRef(extracted.object.entity, resourceId),
                }
              : extracted.object;
          const next: FactRecord = {
            organizationId: snapshot.organizationId,
            factKey: extracted.factKey,
            subject,
            predicate: extracted.predicate,
            object,
            singleValue: extracted.singleValue ?? false,
            status: previous?.status ?? "active",
            updatedAt: now,
          };
          await unitOfWork.saveFact(next);
          if (!previous) {
            await unitOfWork.appendFactEvent({
              organizationId: snapshot.organizationId,
              factKey: extracted.factKey,
              type: "created",
              occurredAt: now,
              resourceId,
            });
          }

          for (const sourceChunkId of extracted.evidenceChunkIds) {
            const chunkId = chunkIds.get(sourceChunkId);
            if (!chunkId) {
              throw new Error(
                `Fact "${extracted.factKey}" references unknown chunk "${sourceChunkId}"`,
              );
            }
            const evidenceId = evidenceIdentity(resourceId, extracted.factKey, chunkId);
            const evidence: EvidenceRecord = {
              organizationId: snapshot.organizationId,
              evidenceId,
              factKey: extracted.factKey,
              resourceId,
              chunkId,
              acl: snapshot.chunks.find((chunk) => chunk.id === sourceChunkId)?.acl ?? snapshot.acl,
              origin: "derived",
              status: "active",
            };
            await unitOfWork.saveEvidence(evidence);
          }
        }

        await this.reconcileFacts(unitOfWork, affectedFactKeys, now, resourceId);
        return {
          resourceId,
          changed: true,
          affectedFactKeys: Array.from(affectedFactKeys),
        };
      });
    } catch (error) {
      await this.contentStore.purge(objectKey).catch(() => undefined);
      throw error;
    }
  }

  private async loadPriorResourceEntities(
    organizationId: string,
    resourceId: string,
  ): Promise<readonly ExtractedEntity[]> {
    const [entities, mentions, chunks] = await Promise.all([
      this.repository.listEntitiesForResource(organizationId, resourceId),
      this.repository.listEntityMentions(organizationId, resourceId),
      this.repository.listChunks(organizationId, resourceId),
    ]);
    const sourceChunkByStoredId = new Map(
      chunks
        .filter((chunk) => chunk.status === "active")
        .map((chunk) => [chunk.chunkId, chunk.sourceChunkId]),
    );
    const mentionsByEntity = new Map<string, string[]>();
    for (const mention of mentions) {
      if (mention.status !== "active") continue;
      const sourceChunkId = sourceChunkByStoredId.get(mention.chunkId);
      if (!sourceChunkId) continue;
      const ids = mentionsByEntity.get(mention.entityId) ?? [];
      ids.push(sourceChunkId);
      mentionsByEntity.set(mention.entityId, ids);
    }
    const prefix = `${resourceId}:`;
    return entities
      .filter((entity) => entity.scope === "resource" && entity.status === "active")
      .map((entity) => ({
        entityId: entity.entityId.startsWith(prefix)
          ? entity.entityId.slice(prefix.length)
          : entity.entityId,
        scope: "resource" as const,
        name: entity.name,
        type: entity.type,
        mentionChunkIds: unique(mentionsByEntity.get(entity.entityId) ?? []),
      }));
  }

  async remove(organizationId: string, source: ResourceSnapshot["source"]): Promise<boolean> {
    const existing = await this.repository.getResourceBySource(organizationId, source);
    if (!existing || existing.status !== "active") return false;
    await this.repository.transaction(organizationId, async (unitOfWork) => {
      const now = this.clock.now().toISOString();
      const affectedFactKeys = new Set<string>();
      await unitOfWork.saveResource({ ...existing, status: "stale", updatedAt: now });
      for (const chunk of await unitOfWork.listChunks(existing.resourceId)) {
        if (chunk.status === "active") await unitOfWork.saveChunk({ ...chunk, status: "stale" });
      }
      for (const mention of await unitOfWork.listEntityMentions(existing.resourceId)) {
        if (mention.status === "active") {
          await unitOfWork.saveEntityMention({ ...mention, status: "stale" });
        }
      }
      for (const evidence of await unitOfWork.listEvidenceForResource(existing.resourceId)) {
        if (evidence.factKey) affectedFactKeys.add(evidence.factKey);
        if (evidence.origin === "derived" && evidence.status === "active") {
          await unitOfWork.saveEvidence({ ...evidence, status: "stale" });
        }
      }
      await this.reconcileFacts(unitOfWork, affectedFactKeys, now, existing.resourceId);
    });
    return true;
  }

  private async replaceEntityMentions(
    unitOfWork: KnowledgeGraphUnitOfWork,
    snapshot: ResourceSnapshot,
    resourceId: string,
    chunkIds: ReadonlyMap<string, string>,
  ): Promise<void> {
    for (const extracted of snapshot.entities ?? []) {
      const mayBeGlobal = extracted.scope === "global" && extracted.promotionEvidence !== undefined;
      const scope = mayBeGlobal ? "global" : "resource";
      const entityId =
        scope === "global" ? extracted.entityId : `${resourceId}:${extracted.entityId}`;
      const entity: EntityRecord = {
        organizationId: snapshot.organizationId,
        entityId,
        scope,
        resourceId: scope === "resource" ? resourceId : undefined,
        name: extracted.name,
        type: extracted.type,
        status: "active",
      };
      await unitOfWork.saveEntity(entity);
      for (const sourceChunkId of extracted.mentionChunkIds) {
        const chunkId = chunkIds.get(sourceChunkId);
        if (!chunkId) {
          throw new Error(
            `Entity "${extracted.entityId}" references unknown chunk "${sourceChunkId}"`,
          );
        }
        await unitOfWork.saveEntityMention({
          organizationId: snapshot.organizationId,
          entityId,
          resourceId,
          chunkId,
          status: "active",
        });
      }
    }
  }

  private async reconcileFacts(
    unitOfWork: KnowledgeGraphUnitOfWork,
    affectedFactKeys: ReadonlySet<string>,
    now: string,
    resourceId: string,
  ): Promise<void> {
    const allFacts = await unitOfWork.listFacts();
    const affectedSlots = new Set(
      allFacts
        .filter((fact) => affectedFactKeys.has(fact.factKey))
        .map((fact) => factSlotKey(fact)),
    );
    const baseStatuses = new Map<string, FactStatus>();
    for (const fact of allFacts) {
      if (!affectedSlots.has(factSlotKey(fact))) continue;
      const evidence = await unitOfWork.listEvidenceForFact(fact.factKey);
      baseStatuses.set(
        fact.factKey,
        evidence.some((item) => item.status === "active") ? "active" : "inactive",
      );
    }

    for (const slot of affectedSlots) {
      const facts = allFacts.filter((fact) => factSlotKey(fact) === slot);
      const active = facts.filter((fact) => baseStatuses.get(fact.factKey) === "active");
      const values = new Set(active.map((fact) => factObjectKey(fact.object)));
      const hasConflict = active.some((fact) => fact.singleValue) && values.size > 1;
      for (const fact of facts) {
        const base = baseStatuses.get(fact.factKey) ?? "inactive";
        const nextStatus: FactStatus = hasConflict && base === "active" ? "conflict" : base;
        if (fact.status === nextStatus) continue;
        const eventType = eventForTransition(fact.status, nextStatus);
        await unitOfWork.saveFact({ ...fact, status: nextStatus, updatedAt: now });
        if (eventType) {
          await unitOfWork.appendFactEvent({
            organizationId: fact.organizationId,
            factKey: fact.factKey,
            type: eventType,
            occurredAt: now,
            resourceId,
          });
        }
      }
    }
  }
}

function eventForTransition(previous: FactStatus, next: FactStatus): FactEventType | null {
  if ((previous === "active" || previous === "conflict") && next === "inactive") {
    return "invalidated";
  }
  if (previous === "inactive" && next === "active") return "restored";
  if (next === "conflict" && previous !== "conflict") return "conflict_detected";
  if (previous === "conflict" && next === "active") return "conflict_resolved";
  return null;
}

function evidenceIdentity(resourceId: string, factKey: string, chunkId: string): string {
  return `${resourceId}|${encodeURIComponent(factKey)}|${chunkId}`;
}

function unique(values: readonly string[]): readonly string[] {
  return Array.from(new Set(values));
}

function normalizeEntityRef(ref: FactRecord["subject"], resourceId: string): FactRecord["subject"] {
  if (ref.scope === "global" || ref.entityId.startsWith(`${resourceId}:`)) return ref;
  return { ...ref, entityId: `${resourceId}:${ref.entityId}` };
}
