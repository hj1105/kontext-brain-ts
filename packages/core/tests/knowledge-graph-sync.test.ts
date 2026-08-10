import { describe, expect, it } from "vitest";
import {
  InMemoryKnowledgeGraphRepository,
  InMemoryResourceContentStore,
  type ResourceSnapshot,
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

    const result = await sync.execute(orderSnapshot("v1", "paid"));
    const resource = await repository.getResource(organizationId, result.resourceId);
    const chunks = await repository.listChunks(organizationId, result.resourceId);

    expect(resource?.ontologyNodeIds).toEqual(["order", "payment"]);
    expect(chunks[0]?.ontologyNodeIds).toEqual(["order", "payment"]);
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
