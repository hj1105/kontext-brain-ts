import {
  InMemoryKnowledgeGraphRepository,
  InMemoryResourceContentStore,
  type ResourceSnapshotEnricher,
  SyncResourceUseCase,
} from "@kontext-brain/core";
import { describe, expect, it } from "vitest";
import type { MCPConnector } from "../src/index.js";
import { GenericMCPResourceSnapshotAdapter, MCPKnowledgeSynchronizer } from "../src/index.js";

describe("MCPKnowledgeSynchronizer", () => {
  it("normalizes a generic MCP body and persists Resource, Chunk, and ontology links", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const synchronizer = new MCPKnowledgeSynchronizer(
      new SyncResourceUseCase(repository, new InMemoryResourceContentStore()),
      [new GenericMCPResourceSnapshotAdapter("notion", "notion", { organizationWide: true })],
    );
    let body = "Order 42 was paid";
    let metadata: Record<string, string> = { channel: "orders", team: "acme" };
    const connector: MCPConnector = {
      name: "notion",
      async listResources() {
        return [];
      },
      async fetchResource(resourceId) {
        return { resourceId, content: body, metadata, fetchedAt: new Date() };
      },
      async search() {
        return [];
      },
    };

    const created = await synchronizer.sync(
      "acme",
      connector,
      { id: "page-1", name: "Order page", description: "" },
      ["order", "payment"],
    );
    expect(created.changed).toBe(true);

    metadata = { team: "acme", channel: "orders" };
    const unchanged = await synchronizer.sync(
      "acme",
      connector,
      { id: "page-1", name: "Order page", description: "" },
      ["order", "payment"],
    );
    expect(unchanged.changed).toBe(false);

    body = "Order 42 was refunded";
    const updated = await synchronizer.sync(
      "acme",
      connector,
      { id: "page-1", name: "Order page", description: "" },
      ["order", "payment"],
    );
    expect(updated.changed).toBe(true);
    expect(updated.contentHash).not.toBe(created.contentHash);

    const resource = await repository.getResourceBySource("acme", {
      connectorId: "notion",
      externalId: "page-1",
      type: "notion",
    });
    expect(resource?.ontologyNodeIds).toEqual(["order", "payment"]);
    if (!resource) throw new Error("Expected synchronized Resource");
    expect(await repository.listChunks("acme", resource.resourceId)).toMatchObject([
      { sourceChunkId: "chunk-0", ontologyNodeIds: ["order", "payment"] },
    ]);
  });

  it("enriches a normalized snapshot before evidence-backed synchronization", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const enricher: ResourceSnapshotEnricher = {
      async enrich(snapshot) {
        return {
          snapshot: {
            ...snapshot,
            entities: [
              {
                entityId: "order-42",
                scope: "resource",
                name: "Order 42",
                type: "order",
                mentionChunkIds: ["chunk-0"],
              },
            ],
            facts: [
              {
                factKey: "adaptive-order-42-paid",
                subject: { entityId: "order-42", scope: "resource" },
                predicate: "status",
                object: { kind: "literal", value: "paid" },
                evidenceChunkIds: ["chunk-0"],
                singleValue: true,
              },
            ],
          },
          capabilities: [],
          processedWindows: 1,
          hypothesisCount: 0,
        };
      },
    };
    const synchronizer = new MCPKnowledgeSynchronizer(
      new SyncResourceUseCase(repository, new InMemoryResourceContentStore()),
      [new GenericMCPResourceSnapshotAdapter("notion", "notion", { organizationWide: true })],
      enricher,
    );
    const connector: MCPConnector = {
      name: "notion",
      async listResources() {
        return [];
      },
      async fetchResource(resourceId) {
        return { resourceId, content: "Order 42 was paid", metadata: {}, fetchedAt: new Date() };
      },
      async search() {
        return [];
      },
    };

    await synchronizer.sync(
      "acme",
      connector,
      { id: "page-1", name: "Order page", description: "" },
      ["order", "payment"],
    );

    expect(await repository.getFact("acme", "adaptive-order-42-paid")).toMatchObject({
      predicate: "status",
      object: { kind: "literal", value: "paid" },
      status: "active",
    });
    expect(await repository.listEvidenceForFact("acme", "adaptive-order-42-paid")).toHaveLength(1);
  });
});
