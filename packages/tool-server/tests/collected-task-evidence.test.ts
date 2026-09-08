import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { TaskContextWorkflow } from "@kontext-brain/context";
import {
  FileResourceContentStore,
  type ResourceSnapshot,
  SqliteKnowledgeGraphRepository,
  SyncResourceUseCase,
} from "@kontext-brain/core";
import { FileTaskContextRepository, assembleCurrentTaskContextState } from "@kontext-brain/local";
import { afterEach, expect, it, vi } from "vitest";
import { CollectedTaskEvidence } from "../src/collected-task-evidence.js";

const dirs: string[] = [];
const principal = { organizationId: "personal:one", subjectId: "user:one", groupIds: [] };
afterEach(async () => {
  await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true })));
});
async function fixture(connectorId = "markdown") {
  const dir = await mkdtemp(path.join(tmpdir(), "kontext-collected-evidence-"));
  dirs.push(dir);
  const graph = await SqliteKnowledgeGraphRepository.open(dir);
  const content = new FileResourceContentStore(dir);
  const snapshot: ResourceSnapshot = {
    organizationId: principal.organizationId,
    source: { connectorId, externalId: "document:one", type: "document" },
    title: "Original source title",
    contentHash: "resource-hash:one",
    body: "Original source body",
    acl: { subjectIds: [principal.subjectId] },
    ontologyNodeIds: ["domain:orders"],
    chunks: [
      {
        id: "section:one",
        contentHash: "chunk-hash:one",
        position: 0,
        text: "Original source body",
      },
    ],
    entities: [],
    facts: [],
  };
  const sync = new SyncResourceUseCase(graph, content);
  const { resourceId } = await sync.execute(snapshot);
  const evidence = await graph.transaction(
    principal.organizationId,
    async (tx) => (await tx.listEvidenceForResource(resourceId))[0],
  );
  if (!evidence) throw new Error("Fixture source Evidence was not synchronized");
  const ref = { resourceId, evidenceId: evidence.evidenceId };
  const policy = vi.fn(async () => ["codex"]);
  const collector = new CollectedTaskEvidence(graph, content, policy);
  return { dir, graph, content, snapshot, sync, ref, evidence, policy, collector };
}
it.each(["markdown", "session", "notion", "slack"])(
  "resolves %s Evidence from actual synchronized Resource/Chunk records",
  async (connector) => {
    const h = await fixture(connector);
    const [item] = await h.collector.collect(principal, [h.ref]);
    expect(item).toMatchObject({
      evidenceId: h.ref.evidenceId,
      text: h.snapshot.body,
      availability: "current",
      allowedRuntimeProviders: ["codex"],
      provenance: {
        resourceId: h.ref.resourceId,
        chunkId: h.evidence.chunkId,
        resourceTitle: h.snapshot.title,
        source: h.snapshot.source,
        contentHash: h.snapshot.contentHash,
        ontologyNodeIds: ["domain:orders"],
      },
    });
    expect(item?.provenance?.observedAt).toBe(h.evidence.observedAt);
  },
);
it("checks all ACLs before loading source content", async () => {
  const h = await fixture();
  await h.graph.transaction(principal.organizationId, async (tx) =>
    tx.saveEvidence({ ...h.evidence, acl: { subjectIds: ["someone:else"] } }),
  );
  const get = vi.spyOn(h.content, "get");
  expect(await h.collector.collect(principal, [h.ref])).toEqual([
    {
      evidenceId: h.ref.evidenceId,
      text: "",
      availability: "inaccessible",
      allowedRuntimeProviders: [],
    },
  ]);
  expect(get).not.toHaveBeenCalled();
  expect(h.policy).not.toHaveBeenCalled();
});
it("does not reveal another Organization's source metadata", async () => {
  const h = await fixture();
  const result = await h.collector.collect({ ...principal, organizationId: "other" }, [h.ref]);
  expect(result[0]).toEqual({
    evidenceId: h.ref.evidenceId,
    text: "",
    availability: "unavailable",
    allowedRuntimeProviders: [],
  });
});
it("returns stale without source text when the indexed chunk is stale", async () => {
  const h = await fixture();
  await h.graph.transaction(principal.organizationId, async (tx) => {
    const chunk = (await tx.listChunks(h.ref.resourceId))[0];
    if (!chunk) throw new Error("Fixture Chunk was not synchronized");
    await tx.saveChunk({ ...chunk, status: "stale" });
  });
  expect((await h.collector.collect(principal, [h.ref]))[0]).toMatchObject({
    availability: "stale",
    text: "",
  });
});
it("does not disclose source text after access is revoked during content hydration", async () => {
  const h = await fixture();
  const get = h.content.get.bind(h.content);
  vi.spyOn(h.content, "get").mockImplementation(async (key) => {
    const value = await get(key);
    await h.graph.transaction(principal.organizationId, async (tx) =>
      tx.saveEvidence({ ...h.evidence, acl: {} }),
    );
    return value;
  });
  expect((await h.collector.collect(principal, [h.ref]))[0]).toMatchObject({
    availability: "inaccessible",
    text: "",
  });
});
it("marks concurrently replaced source content stale instead of attaching old bytes to new provenance", async () => {
  const h = await fixture();
  const get = h.content.get.bind(h.content);
  vi.spyOn(h.content, "get").mockImplementation(async (key) => {
    const value = await get(key);
    await h.sync.execute({
      ...h.snapshot,
      contentHash: "resource-hash:two",
      body: "New source",
      chunks: [
        { id: "section:one", position: 0, contentHash: "chunk-hash:two", text: "New source" },
      ],
    });
    return value;
  });
  expect((await h.collector.collect(principal, [h.ref]))[0]).toMatchObject({
    availability: "stale",
    text: "",
  });
});
it("does not accept content-store identity mismatches", async () => {
  const h = await fixture();
  vi.spyOn(h.content, "get").mockResolvedValue({
    organizationId: "other",
    resourceId: h.ref.resourceId,
    contentHash: h.snapshot.contentHash,
    body: "Wrong body",
    chunks: { "section:one": "Wrong body" },
  });
  expect((await h.collector.collect(principal, [h.ref]))[0]).toMatchObject({
    availability: "unavailable",
    text: "",
  });
});
it("preserves policy denial and never grants provider access from source text", async () => {
  const h = await fixture();
  h.policy.mockResolvedValue([]);
  expect((await h.collector.collect(principal, [h.ref]))[0]?.allowedRuntimeProviders).toEqual([]);
});
it("keeps missing or offline Evidence explicit", async () => {
  const h = await fixture();
  vi.spyOn(h.content, "get").mockRejectedValue(new Error("Infrastructure offline"));
  expect((await h.collector.collect(principal, [h.ref]))[0]).toMatchObject({
    availability: "unavailable",
    text: "",
  });
});

it("rechecks provider policy after hydration without expanding the initial grant", async () => {
  const h = await fixture();
  h.policy.mockResolvedValueOnce(["codex"]).mockResolvedValueOnce(["claude"]);
  expect((await h.collector.collect(principal, [h.ref]))[0]?.allowedRuntimeProviders).toEqual([]);
});

it("feeds source-owned provenance through persistent Task preparation and blocks stale refresh", async () => {
  const h = await fixture("session");
  const repository = new FileTaskContextRepository(h.dir);
  const workflow = new TaskContextWorkflow(repository, repository);
  const contract = {
    taskId: "task:collected",
    intent: "Implement the supported behavior",
    risk: "low" as const,
    acceptance: [
      {
        criterionId: "criterion:one",
        statement: "Pass behavior test",
        verifier: { kind: "test" as const, ref: "fixture:test" },
      },
    ],
    targets: ["planned:one"],
    nonGoals: [],
  };
  const publish = async () => {
    const state = assembleCurrentTaskContextState({
      taskId: contract.taskId,
      organizationId: principal.organizationId,
      codeRevision: "revision:one",
      evidence: await h.collector.collect(principal, [h.ref]),
      logicPlans: [
        {
          workItemId: "logic:one",
          plannedSymbolIds: ["planned:one"],
          allowedPaths: ["src/one.ts"],
        },
      ],
    });
    await repository.publishCurrent(contract.taskId, state);
    return state;
  };
  const state = await publish();
  expect(state.normativeRecords).toEqual([]);
  expect(await h.graph.listFacts(principal.organizationId)).toEqual([]);
  await workflow.prepareTask({
    contract,
    additionalRequiredEvidenceIds: [h.ref.evidenceId],
    createdAt: new Date().toISOString(),
  });
  const request = {
    taskId: contract.taskId,
    logic: { workItemId: "logic:one", plannedSymbolIds: ["planned:one"] },
    runtimeProvider: "codex",
    issuedAt: new Date().toISOString(),
    expiresAt: new Date(Date.now() + 300_000).toISOString(),
    totalTokenBudget: 10000,
    optionalEvidenceTokenBudget: 1000,
  };
  const compiled = await workflow.beginLogic(request);
  expect(compiled.editingAllowed).toBe(true);
  expect(compiled.mandatory.evidence[0]?.provenance?.source.connectorId).toBe("session");
  expect(compiled.receipt?.evidenceIds).toContain(h.ref.evidenceId);
  await h.sync.remove(principal.organizationId, h.snapshot.source);
  await publish();
  const stale = await workflow.beginLogic(request);
  expect(stale.editingAllowed).toBe(false);
  expect(stale.receipt).toBeUndefined();
  await workflow.refreshTaskContext({
    taskId: contract.taskId,
    createdAt: new Date().toISOString(),
  });
  expect((await workflow.beginLogic(request)).editingAllowed).toBe(false);
});
