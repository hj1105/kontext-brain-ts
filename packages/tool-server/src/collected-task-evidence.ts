import type { ContextEvidenceItem } from "@kontext-brain/context";
import {
  type ChunkRecord,
  DefaultAccessPolicy,
  type EvidenceRecord,
  type KnowledgeGraphRepository,
  type Principal,
  type ResourceContentStore,
  type ResourceRecord,
} from "@kontext-brain/core";

export interface CollectedEvidenceRef {
  readonly resourceId: string;
  readonly evidenceId: string;
}
export interface CollectedEvidenceMetadata {
  readonly resource: ResourceRecord;
  readonly chunk: ChunkRecord;
  readonly evidence: EvidenceRecord;
}
export type CollectedEvidenceEgressPolicy = (
  principal: Principal,
  metadata: CollectedEvidenceMetadata,
) => Promise<readonly string[]>;

/** Resolves source-owned bytes and provenance; never creates Facts or normative approvals. */
export class CollectedTaskEvidence {
  private readonly access = new DefaultAccessPolicy();

  constructor(
    private readonly graph: KnowledgeGraphRepository,
    private readonly content: ResourceContentStore,
    private readonly egress: CollectedEvidenceEgressPolicy,
  ) {}

  async collect(
    principal: Principal,
    references: readonly CollectedEvidenceRef[],
  ): Promise<readonly ContextEvidenceItem[]> {
    if (!principal.organizationId.trim() || !principal.subjectId.trim())
      throw new Error("An authenticated principal is required");
    if (references.length > 256) throw new Error("Too many collected Evidence references");
    const selected = new Map<string, CollectedEvidenceRef>();
    for (const ref of references) {
      if (!ref.resourceId.trim() || !ref.evidenceId.trim())
        throw new Error("Resource and Evidence IDs are required");
      const previous = selected.get(ref.evidenceId);
      if (previous && previous.resourceId !== ref.resourceId)
        throw new Error("An Evidence ID cannot name different Resources");
      selected.set(ref.evidenceId, { ...ref });
    }
    const result: ContextEvidenceItem[] = [];
    let remainingCharacters = 2_000_000;
    for (const ref of selected.values()) {
      const item = await this.resolve(principal, ref);
      if (item.text.length > remainingCharacters) {
        result.push(unavailable(ref));
      } else {
        result.push(item);
        remainingCharacters -= item.text.length;
      }
    }
    return result;
  }

  private async resolve(
    principal: Principal,
    ref: CollectedEvidenceRef,
  ): Promise<ContextEvidenceItem> {
    try {
      const before = await this.readMetadata(principal, ref);
      const blocked = this.blocked(principal, ref, before);
      if (blocked || !before) return blocked ?? unavailable(ref);
      const initiallyAllowedProviders = new Set(await this.egress(principal, before));
      const stored = await this.content.get(before.chunk.contentObjectKey);
      const allowedRuntimeProviders = [...new Set(await this.egress(principal, before))]
        .filter((provider) => initiallyAllowedProviders.has(provider))
        .sort();
      // Recheck after hydration: synchronization or ACL changes must not disclose the old snapshot.
      const after = await this.readMetadata(principal, ref);
      const changedAccess = this.blocked(principal, ref, after);
      if (changedAccess || !after) return changedAccess ?? unavailable(ref);
      if (JSON.stringify(before) !== JSON.stringify(after)) return unavailable(ref, "stale");
      if (
        !stored ||
        stored.organizationId !== principal.organizationId ||
        stored.resourceId !== ref.resourceId ||
        stored.contentHash !== before.resource.contentHash
      )
        return unavailable(ref);
      const text = Object.hasOwn(stored.chunks, before.chunk.sourceChunkId)
        ? stored.chunks[before.chunk.sourceChunkId]
        : undefined;
      if (typeof text !== "string" || text.length > 1_000_000) return unavailable(ref);
      const observedAt = before.evidence.observedAt ?? before.resource.updatedAt;
      if (!Number.isFinite(Date.parse(observedAt))) return unavailable(ref);
      return {
        evidenceId: ref.evidenceId,
        text,
        sourceSpan: before.chunk.sourceChunkId,
        availability: "current",
        allowedRuntimeProviders,
        provenance: {
          resourceId: before.resource.resourceId,
          chunkId: before.chunk.chunkId,
          resourceTitle: before.resource.title,
          source: { ...before.resource.source },
          observedAt: new Date(observedAt).toISOString(),
          contentHash: before.resource.contentHash,
          ontologyNodeIds: [
            ...new Set([...before.resource.ontologyNodeIds, ...before.chunk.ontologyNodeIds]),
          ].sort(),
        },
      };
    } catch {
      return unavailable(ref);
    }
  }

  private readMetadata(
    principal: Principal,
    ref: CollectedEvidenceRef,
  ): Promise<CollectedEvidenceMetadata | null> {
    return this.graph.transaction(principal.organizationId, async (tx) => {
      const resource = await tx.getResource(ref.resourceId);
      if (!resource) return null;
      const evidence = (await tx.listEvidenceForResource(ref.resourceId)).find(
        (item) => item.evidenceId === ref.evidenceId,
      );
      if (!evidence) return null;
      const chunk = (await tx.listChunks(ref.resourceId)).find(
        (item) => item.chunkId === evidence.chunkId,
      );
      return chunk ? { resource, chunk, evidence } : null;
    });
  }

  private blocked(
    principal: Principal,
    ref: CollectedEvidenceRef,
    metadata: CollectedEvidenceMetadata | null,
  ): ContextEvidenceItem | null {
    if (!metadata) return unavailable(ref);
    const { resource, chunk, evidence } = metadata;
    const records = [resource, chunk, evidence];
    if (
      records.some(
        (record) =>
          record.organizationId !== principal.organizationId ||
          record.resourceId !== ref.resourceId,
      )
    )
      return unavailable(ref);
    if (records.some((record) => !this.access.canAccess(principal, record.acl)))
      return unavailable(ref, "inaccessible");
    if (records.some((record) => record.status === "purged")) return unavailable(ref);
    if (records.some((record) => record.status !== "active")) return unavailable(ref, "stale");
    if (chunk.contentObjectKey !== resource.contentObjectKey) return unavailable(ref, "stale");
    return null;
  }
}

function unavailable(
  ref: CollectedEvidenceRef,
  availability: "unavailable" | "inaccessible" | "stale" = "unavailable",
): ContextEvidenceItem {
  return { evidenceId: ref.evidenceId, text: "", availability, allowedRuntimeProviders: [] };
}
