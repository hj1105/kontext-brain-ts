import { describe, expect, it } from "vitest";
import {
  InMemoryKnowledgeGraphRepository,
  InMemoryResourceContentStore,
  SyncResourceUseCase,
} from "../src/index.js";

const organizationId = "org:acme";

function snapshot(input: {
  readonly externalId: string;
  readonly title: string;
  readonly body: string;
  readonly ontologyNodeIds: readonly string[];
}) {
  return {
    organizationId,
    source: { connectorId: "docs", externalId: input.externalId, type: "markdown" },
    title: input.title,
    contentHash: `sha256:${input.externalId}`,
    body: input.body,
    acl: { organizationWide: true },
    ontologyNodeIds: input.ontologyNodeIds,
    chunks: [{ chunkId: `${input.externalId}:0`, body: input.body }],
  };
}

/**
 * Code files and documents are both synchronized with ontologyNodeIds, but until
 * now nothing could read that edge back. Without it a Code Symbol cannot reach
 * the approved decisions covering its area, which is the whole point of linking
 * them in the first place.
 */
describe("listResourcesByOntologyNode", () => {
  it("returns the resources attached to one node and nothing else", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const sync = new SyncResourceUseCase(repository, new InMemoryResourceContentStore());

    await sync.execute(
      snapshot({
        externalId: "spec:billing-retry",
        title: "Billing retry specification",
        body: "Billing retry delays use a factor of 3.",
        ontologyNodeIds: ["billing"],
      }),
    );
    await sync.execute(
      snapshot({
        externalId: "spec:notify-retry",
        title: "Notify retry specification",
        body: "Notify retry delays use a factor of 2.",
        ontologyNodeIds: ["notify"],
      }),
    );
    await sync.execute(
      snapshot({
        externalId: "runbook:shared",
        title: "Shared on-call runbook",
        body: "Escalate before changing an approved value.",
        ontologyNodeIds: ["billing", "notify"],
      }),
    );

    const billing = await repository.listResourcesByOntologyNode(organizationId, "billing");
    expect(billing.map((resource) => resource.source.externalId)).toEqual([
      "runbook:shared",
      "spec:billing-retry",
    ]);

    const notify = await repository.listResourcesByOntologyNode(organizationId, "notify");
    expect(notify.map((resource) => resource.source.externalId)).toEqual([
      "runbook:shared",
      "spec:notify-retry",
    ]);
  });

  it("returns nothing for a node with no resources", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const sync = new SyncResourceUseCase(repository, new InMemoryResourceContentStore());
    await sync.execute(
      snapshot({
        externalId: "spec:billing-retry",
        title: "Billing retry specification",
        body: "Billing retry delays use a factor of 3.",
        ontologyNodeIds: ["billing"],
      }),
    );
    expect(await repository.listResourcesByOntologyNode(organizationId, "media")).toEqual([]);
  });

  it("keeps one organization's resources out of another's node", async () => {
    const repository = new InMemoryKnowledgeGraphRepository();
    const sync = new SyncResourceUseCase(repository, new InMemoryResourceContentStore());
    await sync.execute(
      snapshot({
        externalId: "spec:billing-retry",
        title: "Billing retry specification",
        body: "Billing retry delays use a factor of 3.",
        ontologyNodeIds: ["billing"],
      }),
    );
    expect(await repository.listResourcesByOntologyNode("org:other", "billing")).toEqual([]);
  });
});
