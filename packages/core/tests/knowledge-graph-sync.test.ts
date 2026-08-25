import { describe, expect, it } from "vitest";
import {
  InMemoryKnowledgeGraphRepository,
  InMemoryResourceContentStore,
  type ResourceSnapshot,
  type ResourceSnapshotEnricher,
  SyncResourceUseCase,
} from "../src/index.js";

const organizationId = "acme";

function orderSnapshot(
  contentHash: string,
  status?: "paid" | "cancelled",
  externalId = "orders-42",
): ResourceSnapshot {
  return {
    organizationId,
    source: { connectorId: "notion", externalId, type: "notion" },
    title: "Order 42",
    contentHash,
    body: status ? `Order 42 is ${status}` : "Order 42 status was removed",
    acl: { organizationWide: true },
    ontologyNodeIds: ["order", "payment"],
    chunks: [
      {
        id: "status-block",
        contentHash: `${contentHash}:block`,
        text: status ? `Status: ${status}` : "No status",
        position: 0,
        ontologyNodeIds: ["order", "payment"],
      },
    ],
    facts: status
      ? [
          {
            factKey: `order:42:status:${status}`,
            subject: { entityId: "order:42", scope: "global" },
            predicate: "status",
            object: { kind: "literal", value: status },
            evidenceChunkIds: ["status-block"],
            singleValue: true,
          },
        ]
      : [],
  };
}

describe("SyncResourceUseCase", () => {
  it("invalidates a fact when its source evidence disappears and restores the same fact key", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const contentStore = new InMemoryResourceContentStore();
    const sync = new SyncResourceUseCase(repository, contentStore);

    await sync.execute(orderSnapshot("v1", "paid"));
    expect((await repository.getFact(organizationId, "order:42:status:paid"))?.status).toBe(
      "active",
    );

    await sync.execute(orderSnapshot("v2"));
    expect((await repository.getFact(organizationId, "order:42:status:paid"))?.status).toBe(
      "inactive",
    );

    await sync.execute(orderSnapshot("v3", "paid"));
    const restored = await repository.getFact(organizationId, "order:42:status:paid");
    expect(restored?.status).toBe("active");
    if (!restored) throw new Error("Expected restored Fact");
    expect(await repository.listFactEvents(organizationId, restored.factKey)).toMatchObject([
      { type: "created" },
      { type: "invalidated" },
      { type: "restored" },
    ]);
  });

  it("keeps multiple ontology links for a resource and its chunks", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const sync = new SyncResourceUseCase(repository, new InMemoryResourceContentStore());

    const base = orderSnapshot("v1", "paid");
    const result = await sync.execute({
      ...base,
      ontologyLinks: [
        { ontologyNodeId: "order", origin: "manual", confidence: 1 },
        { ontologyNodeId: "payment", origin: "automatic", confidence: 0.7 },
      ],
      chunks: base.chunks.map((chunk) => ({
        ...chunk,
        ontologyLinks: [{ ontologyNodeId: "order", origin: "deterministic", confidence: 1 }],
      })),
    });
    const resource = await repository.getResource(organizationId, result.resourceId);
    const chunks = await repository.listChunks(organizationId, result.resourceId);

    expect(resource?.ontologyNodeIds).toEqual(["order", "payment"]);
    expect(chunks[0]?.ontologyNodeIds).toEqual(["order", "payment"]);
    expect(resource?.ontologyLinks).toMatchObject([
      { ontologyNodeId: "order", origin: "manual", confidence: 1 },
      { ontologyNodeId: "payment", origin: "automatic", confidence: 0.7 },
    ]);
    expect(chunks[0]?.ontologyLinks).toMatchObject([
      { ontologyNodeId: "order", origin: "deterministic", confidence: 1 },
    ]);
  });

  it("marks competing active values as conflicts instead of choosing one", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const sync = new SyncResourceUseCase(repository, new InMemoryResourceContentStore());

    await sync.execute(orderSnapshot("v1", "paid", "finance-system"));
    await sync.execute(orderSnapshot("v1", "cancelled", "support-system"));

    expect((await repository.getFact(organizationId, "order:42:status:paid"))?.status).toBe(
      "conflict",
    );
    expect((await repository.getFact(organizationId, "order:42:status:cancelled"))?.status).toBe(
      "conflict",
    );
  });

  it("is idempotent for the same source and content hash", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const contentStore = new InMemoryResourceContentStore();
    const sync = new SyncResourceUseCase(repository, contentStore);

    const first = await sync.execute(orderSnapshot("v1", "paid"));
    const second = await sync.execute(orderSnapshot("v1", "paid"));

    expect(first.changed).toBe(true);
    expect(second.changed).toBe(false);
    expect(contentStore.putCount).toBe(1);
    expect(await repository.listFactEvents(organizationId, "order:42:status:paid")).toHaveLength(1);
  });

  it("re-syncs unchanged content when the resource ACL changes", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const contentStore = new InMemoryResourceContentStore();
    const sync = new SyncResourceUseCase(repository, contentStore);

    const first = await sync.execute(orderSnapshot("v1", "paid"));
    const restricted = {
      ...orderSnapshot("v1", "paid"),
      acl: { subjectIds: ["finance-user"] },
    } satisfies ResourceSnapshot;
    const repeated = await sync.execute(restricted);

    expect(repeated.changed).toBe(true);
    expect(contentStore.putCount).toBe(2);
    expect((await repository.getResource(organizationId, first.resourceId))?.acl).toEqual(
      restricted.acl,
    );
    expect((await repository.listChunks(organizationId, first.resourceId))[0]?.acl).toEqual(
      restricted.acl,
    );
    expect(
      (await repository.listEvidenceForFact(organizationId, "order:42:status:paid"))[0]?.acl,
    ).toEqual(restricted.acl);
  });

  it("re-syncs unchanged content when a chunk ACL changes", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const contentStore = new InMemoryResourceContentStore();
    const sync = new SyncResourceUseCase(repository, contentStore);
    const withChunkAcl = (groupId: string): ResourceSnapshot => {
      const snapshot = orderSnapshot("v1", "paid");
      return {
        ...snapshot,
        chunks: snapshot.chunks.map((chunk) => ({
          ...chunk,
          acl: { groupIds: [groupId] },
        })),
      };
    };

    const first = await sync.execute(withChunkAcl("finance"));
    const repeated = await sync.execute(withChunkAcl("support"));

    expect(repeated.changed).toBe(true);
    expect(contentStore.putCount).toBe(2);
    expect((await repository.listChunks(organizationId, first.resourceId))[0]?.acl).toEqual({
      groupIds: ["support"],
    });
    expect(
      (await repository.listEvidenceForFact(organizationId, "order:42:status:paid"))[0]?.acl,
    ).toEqual({ groupIds: ["support"] });
  });

  it("keeps semantically equivalent resource and chunk ACLs unchanged", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const contentStore = new InMemoryResourceContentStore();
    const sync = new SyncResourceUseCase(repository, contentStore);
    const first = orderSnapshot("v1", "paid");
    const initial = {
      ...first,
      acl: { subjectIds: ["alice", "bob", "alice"], groupIds: ["finance", "support"] },
      chunks: first.chunks.map((chunk) => ({
        ...chunk,
        acl: { subjectIds: ["carol", "carol"], groupIds: ["editors", "readers"] },
      })),
    } satisfies ResourceSnapshot;
    const equivalent = {
      ...initial,
      acl: { subjectIds: ["bob", "alice"], groupIds: ["support", "finance", "support"] },
      chunks: initial.chunks.map((chunk) => ({
        ...chunk,
        acl: { subjectIds: ["carol"], groupIds: ["readers", "editors", "readers"] },
      })),
    } satisfies ResourceSnapshot;

    await sync.execute(initial);
    const repeated = await sync.execute(equivalent);

    expect(repeated.changed).toBe(false);
    expect(contentStore.putCount).toBe(1);
  });

  it("can force re-enrichment when source content is unchanged", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const contentStore = new InMemoryResourceContentStore();
    const sync = new SyncResourceUseCase(repository, contentStore);
    let enrichmentCount = 0;
    const enricher: ResourceSnapshotEnricher = {
      async enrich(snapshot) {
        enrichmentCount += 1;
        return {
          snapshot,
          capabilities: [],
          processedWindows: 1,
          hypothesisCount: 0,
        };
      },
    };

    await sync.execute(orderSnapshot("v1", "paid"));
    const repeated = await sync.execute(orderSnapshot("v1", "paid"), enricher, {
      forceReenrich: true,
    });

    expect(repeated.changed).toBe(true);
    expect(enrichmentCount).toBe(1);
    expect(contentStore.putCount).toBe(2);
  });

  it("passes prior identity separately without reactivating a disappeared Mention", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const sync = new SyncResourceUseCase(repository, new InMemoryResourceContentStore());
    const priorCounts: number[] = [];
    const enricher: ResourceSnapshotEnricher = {
      async enrich(snapshot, priorEntities = []) {
        priorCounts.push(priorEntities.length);
        return {
          snapshot:
            priorCounts.length === 1
              ? {
                  ...snapshot,
                  entities: [
                    {
                      entityId: "adaptive-order-42",
                      scope: "resource",
                      name: "Order 42",
                      type: "other",
                      mentionChunkIds: ["status-block"],
                    },
                  ],
                }
              : snapshot,
          capabilities: [],
          processedWindows: 1,
          hypothesisCount: 0,
        };
      },
    };

    const first = await sync.execute(orderSnapshot("v1", "paid"), enricher);
    await sync.execute(orderSnapshot("v2"), enricher);

    expect(priorCounts).toEqual([0, 1]);
    expect(
      (await repository.listEntityMentions(organizationId, first.resourceId)).filter(
        (mention) => mention.status === "active",
      ),
    ).toEqual([]);
  });

  it("soft-removes a missing source Resource and restores it when the source reappears", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const sync = new SyncResourceUseCase(repository, new InMemoryResourceContentStore());
    const source = orderSnapshot("v1", "paid").source;
    await sync.execute(orderSnapshot("v1", "paid"));

    expect(await sync.remove(organizationId, source)).toBe(true);
    expect((await repository.getResourceBySource(organizationId, source))?.status).toBe("stale");
    expect((await repository.getFact(organizationId, "order:42:status:paid"))?.status).toBe(
      "inactive",
    );

    await sync.execute(orderSnapshot("v1", "paid"));
    expect((await repository.getFact(organizationId, "order:42:status:paid"))?.status).toBe(
      "active",
    );
  });
});
