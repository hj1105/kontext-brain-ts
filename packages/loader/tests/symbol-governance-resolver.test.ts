import {
  InMemoryKnowledgeGraphRepository,
  InMemoryResourceContentStore,
  SyncResourceUseCase,
} from "@kontext-brain/core";
import { describe, expect, it } from "vitest";
import {
  ExternalIdNormativeResourceReader,
  SymbolGovernanceResolver,
} from "../src/symbol-governance-resolver.js";

const organizationId = "org:acme";
const codebaseId = "codebase:shop";

/** Every record in this fixture is at revision 1. */
const reader = new ExternalIdNormativeResourceReader(() => "revision:1");

async function seed() {
  const repository = new InMemoryKnowledgeGraphRepository();
  const sync = new SyncResourceUseCase(repository, new InMemoryResourceContentStore());
  const put = (input: {
    connectorId: string;
    externalId: string;
    type: string;
    title: string;
    ontologyNodeIds: readonly string[];
  }) =>
    sync.execute({
      organizationId,
      source: {
        connectorId: input.connectorId,
        externalId: input.externalId,
        type: input.type,
      },
      title: input.title,
      contentHash: `sha256:${input.externalId}`,
      body: input.title,
      acl: { organizationWide: true },
      ontologyNodeIds: input.ontologyNodeIds,
      chunks: [{ chunkId: `${input.externalId}:0`, body: input.title }],
    });

  // Two code files that read alike, on different nodes.
  await put({
    connectorId: "code",
    externalId: `${codebaseId}:src/billing/charge.js`,
    type: "code-module",
    title: "chargeBillingRetryDelay",
    ontologyNodeIds: ["billing"],
  });
  await put({
    connectorId: "code",
    externalId: `${codebaseId}:src/notify/push.js`,
    type: "code-module",
    title: "pushNotifyRetryDelay",
    ontologyNodeIds: ["notify"],
  });

  // The governing decision, and a sibling that a similarity search ranks just
  // as highly because the two documents are nearly identical prose.
  await put({
    connectorId: "sidecar",
    externalId: "decision:billing-retry-ceiling",
    type: "normative",
    title: "Billing retry ceiling",
    ontologyNodeIds: ["billing"],
  });
  await put({
    connectorId: "sidecar",
    externalId: "invariant:billing-retry-bounded",
    type: "normative",
    title: "Billing retry stays bounded",
    ontologyNodeIds: ["billing"],
  });
  await put({
    connectorId: "sidecar",
    externalId: "decision:notify-retry-ceiling",
    type: "normative",
    title: "Notify retry ceiling",
    ontologyNodeIds: ["notify"],
  });
  // A runbook is on the node but is not a normative record.
  await put({
    connectorId: "docs",
    externalId: "runbook:billing-oncall",
    type: "markdown",
    title: "Billing on-call runbook",
    ontologyNodeIds: ["billing"],
  });

  return new SymbolGovernanceResolver(repository, reader);
}

describe("SymbolGovernanceResolver", () => {
  it("reaches the governing records through the symbol's own ontology node", async () => {
    const resolver = await seed();
    const result = await resolver.resolve({
      organizationId,
      codebaseId,
      relativePath: "src/billing/charge.js",
      plannedSymbolId: "planned-symbol:charge",
    });
    expect(result.ontologyNodeIds).toEqual(["billing"]);
    expect(result.records.map((record) => record.recordId)).toEqual([
      "decision:billing-retry-ceiling",
      "invariant:billing-retry-bounded",
    ]);
  });

  it("excludes a neighbouring subsystem's decision even though the text is alike", async () => {
    const resolver = await seed();
    const result = await resolver.resolve({
      organizationId,
      codebaseId,
      relativePath: "src/billing/charge.js",
      plannedSymbolId: "planned-symbol:charge",
    });
    // This is the discrimination similarity search cannot make.
    expect(result.records.map((record) => record.recordId)).not.toContain(
      "decision:notify-retry-ceiling",
    );
  });

  it("returns the neighbouring subsystem's own decision for its own symbol", async () => {
    const resolver = await seed();
    const result = await resolver.resolve({
      organizationId,
      codebaseId,
      relativePath: "src/notify/push.js",
      plannedSymbolId: "planned-symbol:push",
    });
    expect(result.records.map((record) => record.recordId)).toEqual([
      "decision:notify-retry-ceiling",
    ]);
  });

  it("skips resources on the node that are not normative records", async () => {
    const resolver = await seed();
    const result = await resolver.resolve({
      organizationId,
      codebaseId,
      relativePath: "src/billing/charge.js",
      plannedSymbolId: "planned-symbol:charge",
    });
    expect(result.records.map((record) => record.recordId)).not.toContain("runbook:billing-oncall");
  });

  it("reports nothing for a file that was never synchronized", async () => {
    const resolver = await seed();
    const result = await resolver.resolve({
      organizationId,
      codebaseId,
      relativePath: "src/media/upload.js",
      plannedSymbolId: "planned-symbol:upload",
    });
    expect(result.ontologyNodeIds).toEqual([]);
    expect(result.records).toEqual([]);
  });
});
