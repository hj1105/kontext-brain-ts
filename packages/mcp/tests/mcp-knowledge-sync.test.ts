import {
  InMemoryKnowledgeGraphRepository,
  InMemoryResourceContentStore,
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
});
