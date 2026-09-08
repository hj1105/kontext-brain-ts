import { createHash, randomUUID } from "node:crypto";
import { mkdtemp, readFile, rename, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { FileTaskContextRepository } from "@kontext-brain/local";
import { afterEach, expect, it, vi } from "vitest";
import { FileRuntimeScheduleJobStore } from "../src/file-runtime-schedule-job-store.js";
import { loadLocalKnowledgePrincipal } from "../src/local-knowledge-principal.js";
import { LocalTaskInventoryOperations } from "../src/local-task-inventory.js";

const roots: string[] = [];
afterEach(async () => {
  vi.restoreAllMocks();
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true })));
});
async function fixture() {
  const root = await mkdtemp(path.join(tmpdir(), "kontext-task-inventory-"));
  roots.push(root);
  const principal = await loadLocalKnowledgePrincipal(root);
  const repository = new FileTaskContextRepository(root);
  async function create(taskId: string, workspaceId = "folder:one", foreign = false) {
    return repository.initializeTask({
      owner: {
        organizationId: principal.organizationId,
        subjectId: foreign ? randomUUID() : principal.subjectId,
        workspacePath: path.join(root, "missing-workspace-must-not-be-read"),
        requestId: randomUUID(),
        requestDigest: `sha256:${"a".repeat(64)}`,
        contextSelection: { workspaceId, sourceResourceIds: [] },
      },
      contract: {
        taskId,
        intent: `Implement ${taskId}`,
        risk: "low",
        nonGoals: [],
        targets: ["symbol:one"],
        acceptance: [
          { criterionId: "one", statement: "Works", verifier: { kind: "test", ref: "test:one" } },
        ],
      },
      state: {
        codeRevision: "revision:one",
        sourceFreshnessDigest: "fresh:one",
        effectiveScopes: [],
        normativeRecords: [],
        normativeRevisionCatalog: [],
        conflicts: [],
        evidence: [
          {
            evidenceId: "private:one",
            text: "PRIVATE_SOURCE_BODY",
            availability: "current",
            allowedRuntimeProviders: [],
          },
        ],
        logicPlans: [],
      },
      createdAt: "2026-09-06T00:00:00.000Z",
    });
  }
  return {
    root,
    principal,
    repository,
    create,
    operations: new LocalTaskInventoryOperations(root),
  };
}
it("lists only registered owned metadata and never treats saved running state as current completion", async () => {
  const h = await fixture();
  await h.create("task:one");
  await h.create("task:foreign", "folder:one", true);
  const store = new FileRuntimeScheduleJobStore(h.root);
  await store.create({
    jobId: "job:one",
    taskId: "task:one",
    status: "queued",
    codeRevision: "revision:one",
    contextDigest: "context:one",
    requestedAt: "2026-09-06T00:00:00.000Z",
    ownerInstanceId: "old-host",
    ownerProcessId: 424242,
    request: {
      taskId: "task:one",
      repositoryPath: "/private/path",
      work: [{ workItemId: "work:one", prompt: "PRIVATE_PROMPT", eligibleProviders: ["codex"] }],
    },
  });
  await store.update("job:one", ["queued"], (job) => ({
    ...job,
    status: "running",
    ownerInstanceId: "old-host",
    ownerProcessId: 424242,
    startedAt: "2026-09-06T00:00:01.000Z",
  }));
  const before = await store.get("job:one");
  const result = await h.operations.list({});
  expect(result.tasks).toHaveLength(1);
  expect(result).toMatchObject({
    observation: "saved_metadata_only",
    currentEvidence: "not_revalidated",
    tasks: [
      {
        taskId: "task:one",
        scheduleCount: 1,
        unsettledScheduleCount: 1,
        latestSchedule: { jobId: "job:one", status: "running" },
        integration: null,
        finalization: null,
      },
    ],
  });
  expect(JSON.stringify(result)).not.toMatch(/PRIVATE_|private:path|ownerProcessId|task:foreign/);
  expect(await store.get("job:one")).toEqual(before);
});
it("paginates a stable owned inventory and refuses changed inventories or filter reuse", async () => {
  const h = await fixture();
  await h.create("task:one");
  await h.create("task:two");
  const first = await h.operations.list({ limit: 1 });
  expect(first.nextCursor).not.toBeNull();
  const second = await h.operations.list({ limit: 1, cursor: first.nextCursor });
  expect(second.tasks[0]?.taskId).not.toBe(first.tasks[0]?.taskId);
  expect(second.nextCursor).toBeNull();
  await expect(
    h.operations.list({ limit: 1, workspaceId: "folder:other", cursor: first.nextCursor }),
  ).rejects.toThrow("inventory changed");
  await h.create("task:three");
  await expect(h.operations.list({ limit: 1, cursor: first.nextCursor })).rejects.toThrow(
    "inventory changed",
  );
});
it("returns an explicit empty registration inventory without inventing Tasks", async () => {
  const h = await fixture();
  expect(await h.operations.list({})).toMatchObject({
    tasks: [],
    nextCursor: null,
    currentEvidence: "not_revalidated",
  });
});
it("filters by the registered workspace and excludes unregistered prepared context", async () => {
  const h = await fixture();
  await h.create("task:one");
  await h.create("task:two", "folder:two");
  const prepared = await h.repository.get("task:one");
  if (!prepared) throw new Error("Missing fixture context");
  await h.repository.put({
    ...prepared,
    contract: { ...prepared.contract, taskId: "task:legacy" },
    snapshot: { ...prepared.snapshot, taskId: "task:legacy" },
  });
  expect(
    (await h.operations.list({ workspaceId: "folder:two" })).tasks.map((row) => row.taskId),
  ).toEqual(["task:two"]);
  expect((await h.operations.list({})).tasks.map((row) => row.taskId)).toEqual([
    "task:one",
    "task:two",
  ]);
});
it("refuses a changing saved schedule observation instead of mixing states", async () => {
  const h = await fixture();
  await h.create("task:one");
  vi.spyOn(FileRuntimeScheduleJobStore.prototype, "listTaskSummaries")
    .mockResolvedValueOnce([])
    .mockResolvedValueOnce([
      {
        taskId: "task:one",
        jobId: "job:new",
        status: "queued",
        codeRevision: "revision:one",
        contextDigest: "context:one",
        requestedAt: "2026-09-06T00:00:01.000Z",
      },
    ]);
  await expect(h.operations.list({})).rejects.toThrow("changed during observation");
});
it("refuses renamed or corrupted registration files instead of returning an incomplete list", async () => {
  const h = await fixture();
  await h.create("task:one");
  const filename = path.join(
    h.root,
    "task-context",
    "initial",
    `${createHash("sha256").update("task:one").digest("hex")}.json`,
  );
  const text = await readFile(filename, "utf8");
  await writeFile(filename, text.replace("Implement task:one", "Tampered intent"));
  await expect(h.operations.list({})).rejects.toThrow("digest mismatch");
  await writeFile(filename, text);
  await rename(filename, path.join(path.dirname(filename), `${"e".repeat(64)}.json`));
  await expect(h.operations.list({})).rejects.toThrow("storage identity mismatch");
});
it("refuses an owner change during observation", async () => {
  const h = await fixture();
  await h.create("task:one");
  const original = FileRuntimeScheduleJobStore.prototype.listTaskSummaries;
  vi.spyOn(FileRuntimeScheduleJobStore.prototype, "listTaskSummaries").mockImplementationOnce(
    async function (this: FileRuntimeScheduleJobStore, ids) {
      const result = await original.call(this, ids);
      await writeFile(
        path.join(h.root, "knowledge", "local-principal.json"),
        JSON.stringify({
          schemaVersion: 1,
          organizationId: h.principal.organizationId,
          subjectId: randomUUID(),
        }),
      );
      return result;
    },
  );
  await expect(h.operations.list({})).rejects.toThrow("changed during observation");
});
