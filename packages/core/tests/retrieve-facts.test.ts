import { describe, expect, it } from "vitest";
import {
  DefaultAccessPolicy,
  InMemoryAuthorizedKnowledgeGraphReader,
  InMemoryKnowledgeGraphRepository,
  InMemoryResourceContentStore,
  type ResourceSnapshot,
  RetrieveFactsUseCase,
  SyncResourceUseCase,
} from "../src/index.js";

function privateSnapshot(externalId: string, groupId: string): ResourceSnapshot {
  return {
    organizationId: "acme",
    source: { connectorId: "slack", externalId, type: "slack" },
    title: "Private order thread",
    contentHash: `hash:${externalId}`,
    body: "Order 42 has enterprise pricing",
    acl: { groupIds: [groupId] },
    chunks: [
      {
        id: "message-1",
        contentHash: `chunk:${externalId}`,
        text: "Enterprise price is 100",
        position: 0,
      },
    ],
    facts: [
      {
        factKey: "order:42:price:100",
        subject: { entityId: "order:42", scope: "global" },
        predicate: "price",
        object: { kind: "literal", value: 100 },
        evidenceChunkIds: ["message-1"],
        singleValue: true,
      },
    ],
  };
}

describe("RetrieveFactsUseCase", () => {
  it("loads content only for active Evidence accessible to the principal", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const contentStore = new InMemoryResourceContentStore();
    const sync = new SyncResourceUseCase(repository, contentStore);
    await sync.execute(privateSnapshot("finance-thread", "finance"));
    const retrieve = new RetrieveFactsUseCase(
      new InMemoryAuthorizedKnowledgeGraphReader(repository, new DefaultAccessPolicy()),
      contentStore,
    );

    const allowed = await retrieve.execute({
      principal: { organizationId: "acme", subjectId: "u1", groupIds: ["finance"] },
    });
    const denied = await retrieve.execute({
      principal: { organizationId: "acme", subjectId: "u2", groupIds: ["engineering"] },
    });

    expect(allowed).toHaveLength(1);
    expect(allowed[0]?.evidence[0]?.text).toBe("Enterprise price is 100");
    expect(denied).toEqual([]);
  });

  it("keeps a fact visible when at least one of several active Evidence items is accessible", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const contentStore = new InMemoryResourceContentStore();
    const sync = new SyncResourceUseCase(repository, contentStore);
    await sync.execute(privateSnapshot("finance-thread", "finance"));
    await sync.execute(privateSnapshot("sales-thread", "sales"));
    const retrieve = new RetrieveFactsUseCase(
      new InMemoryAuthorizedKnowledgeGraphReader(repository, new DefaultAccessPolicy()),
      contentStore,
    );

    const result = await retrieve.execute({
      principal: { organizationId: "acme", subjectId: "u1", groupIds: ["sales"] },
    });

    expect(result).toHaveLength(1);
    expect(result[0]?.evidence).toHaveLength(1);
    expect(result[0]?.evidence[0]?.resourceId).toContain("sales-thread");
  });

  it("never crosses the principal organization boundary", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const contentStore = new InMemoryResourceContentStore();
    await new SyncResourceUseCase(repository, contentStore).execute(
      privateSnapshot("finance-thread", "finance"),
    );
    const retrieve = new RetrieveFactsUseCase(
      new InMemoryAuthorizedKnowledgeGraphReader(repository, new DefaultAccessPolicy()),
      contentStore,
    );

    expect(
      await retrieve.execute({
        principal: { organizationId: "other", subjectId: "u1", groupIds: ["finance"] },
      }),
    ).toEqual([]);
  });
});
