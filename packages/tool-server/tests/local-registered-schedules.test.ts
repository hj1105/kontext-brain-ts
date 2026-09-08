import { randomUUID } from "node:crypto";
import { mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { FileTaskContextRepository } from "@kontext-brain/local";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { afterEach, expect, it, vi } from "vitest";
import { FileIntegratedTaskStateStore } from "../src/file-integrated-task-state-store.js";
import {
  FileRuntimeScheduleJobStore,
  RuntimeScheduleJobManager,
} from "../src/file-runtime-schedule-job-store.js";
import { loadLocalKnowledgePrincipal } from "../src/local-knowledge-principal.js";
import { LocalRegisteredIntegrationOperations } from "../src/local-registered-integration.js";
import { LocalRegisteredScheduleOperations } from "../src/local-registered-schedules.js";

const roots: string[] = [];
afterEach(async () => {
  vi.restoreAllMocks();
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true })));
});
async function fixture(readableWorkspace = false) {
  const root = await mkdtemp(path.join(tmpdir(), "kontext-registered-schedule-"));
  roots.push(root);
  const workspacePath = readableWorkspace ? root : "/unread-project";
  const principal = await loadLocalKnowledgePrincipal(root);
  await new FileTaskContextRepository(root).initializeTask({
    owner: {
      organizationId: principal.organizationId,
      subjectId: principal.subjectId,
      workspacePath,
      requestId: randomUUID(),
      requestDigest: `sha256:${"a".repeat(64)}`,
      contextSelection: { workspaceId: "folder:one", sourceResourceIds: [] },
    },
    contract: {
      taskId: "task:one",
      intent: "Test registered execution",
      risk: "low",
      targets: ["symbol:one"],
      nonGoals: [],
      acceptance: [
        { criterionId: "test", statement: "works", verifier: { kind: "test", ref: "test:one" } },
      ],
    },
    state: {
      codeRevision: "revision:one",
      sourceFreshnessDigest: "fresh:one",
      effectiveScopes: [],
      normativeRecords: [],
      normativeRevisionCatalog: [],
      conflicts: [],
      evidence: [],
      logicPlans: [],
    },
    createdAt: "2026-09-06T00:00:00.000Z",
  });
  const store = new FileRuntimeScheduleJobStore(root);
  await store.create({
    jobId: "job:one",
    taskId: "task:one",
    status: "queued",
    codeRevision: "revision:one",
    contextDigest: "context:one",
    requestedAt: "2026-09-06T00:00:00.000Z",
    ownerInstanceId: "stopped-fixture",
    ownerProcessId: process.pid,
    request: {
      taskId: "task:one",
      repositoryPath: workspacePath,
      work: [{ workItemId: "work:one", prompt: "PRIVATE_PROMPT", eligibleProviders: ["codex"] }],
    },
  });
  await store.update("job:one", "queued", (job) => ({
    ...job,
    status: "interrupted",
    finishedAt: "2026-09-06T00:00:01.000Z",
    diagnostic: "PRIVATE_DIAGNOSTIC",
  }));
  const manager = new RuntimeScheduleJobManager(
    store,
    () => new Date("2026-09-06T00:00:02.000Z"),
    () => "unused",
    424242,
    () => false,
  );
  const runtime = {
    getSchedule: vi.fn(async ({ jobId }: { jobId: string }) => manager.get(jobId)),
    cancelSchedule: vi.fn(async ({ jobId }: { jobId: string }) => manager.cancel(jobId)),
  };
  const operations = new LocalRegisteredScheduleOperations(root, runtime);
  const selector = { taskId: "task:one", jobId: "job:one" };
  return { root, principal, store, manager, runtime, operations, selector };
}

it("reads an integration record for another execution without running or claiming completion", async () => {
  const h = await fixture();
  const state = integrationState(h.root);
  await new FileIntegratedTaskStateStore(h.root).put(state);
  const runtime = { integrateSchedule: vi.fn() };
  const operations = new LocalRegisteredIntegrationOperations(h.root, h.operations, runtime);
  const result = await operations.inspect(h.selector);
  expect(result).toMatchObject({
    integration: state,
    canRequestIntegration: false,
    currentEvidence: "not_revalidated",
  });
  expect(h.runtime.getSchedule).not.toHaveBeenCalled();
  expect(runtime.integrateSchedule).not.toHaveBeenCalled();
});
it("requires fresh identity, integration basis and consent, then recovers a lost response by read only", async () => {
  const h = await fixture(true);
  await h.store.update("job:one", "interrupted", (job) => ({
    ...job,
    status: "queued",
    finishedAt: undefined,
    diagnostic: undefined,
  }));
  await h.store.update("job:one", "queued", (job) => ({
    ...job,
    status: "running",
    startedAt: job.requestedAt,
  }));
  await h.store.update("job:one", "running", (job) => ({
    ...job,
    status: "completed",
    finishedAt: "2026-09-06T00:00:03.000Z",
    result: {
      capabilities: [],
      results: [
        {
          workItemId: "work:one",
          status: "completed",
          attempts: 1,
          provider: "codex",
          diagnostics: [],
          checkpoints: [],
        },
      ],
    },
  }));
  const runtime = {
    integrateSchedule: vi.fn(async () => {
      await new FileIntegratedTaskStateStore(h.root).put({
        ...integrationState(h.root),
        scheduleJobId: "job:one",
      });
      throw new Error("lost response");
    }),
  };
  const operations = new LocalRegisteredIntegrationOperations(h.root, h.operations, runtime);
  const reviewed = await operations.inspect(h.selector);
  const request = {
    ...h.selector,
    expectedJobIdentityDigest: reviewed.jobIdentityDigest,
    expectedIntegrationDigest: null,
    allowSubscriptionExecution: true,
  };
  for (const input of [
    { ...request, allowSubscriptionExecution: false },
    { ...request, expectedJobIdentityDigest: `sha256:${"e".repeat(64)}` },
    { ...request, expectedIntegrationDigest: `sha256:${"e".repeat(64)}` },
  ]) {
    await expect(operations.integrate(input)).rejects.toThrow();
  }
  expect(runtime.integrateSchedule).not.toHaveBeenCalled();
  await expect(operations.integrate(request)).rejects.toThrow("lost response");
  const recovered = await operations.inspect(h.selector);
  expect(recovered.integration?.scheduleJobId).toBe("job:one");
  expect(recovered.currentEvidence).toBe("not_revalidated");
  expect(runtime.integrateSchedule).toHaveBeenCalledTimes(1);
  await expect(operations.integrate(request)).rejects.toThrow("changed");
  expect(runtime.integrateSchedule).toHaveBeenCalledTimes(1);
  await writeFile(
    path.join(h.root, "knowledge", "local-principal.json"),
    JSON.stringify({
      schemaVersion: 1,
      organizationId: h.principal.organizationId,
      subjectId: randomUUID(),
    }),
  );
  await expect(operations.inspect(h.selector)).rejects.toThrow("current local principal");
});

function integrationState(root: string) {
  return {
    taskId: "task:one",
    scheduleJobId: "job:other",
    repositoryPath: root,
    workspacePath: root,
    baseRevision: "revision:one",
    gitCommit: "commit:one",
    resultRevision: "revision:result",
    contextDigest: "context:one",
    changeBundleIds: ["bundle:one"],
    workItemIds: ["work:one"],
    changedPaths: ["index.ts"],
    changedSymbolIds: ["symbol:one"],
    authorProviders: ["codex" as const],
    createdAt: "2026-09-06T00:00:04.000Z",
  };
}

it("reads registered saved metadata without consulting runtime or exposing private bodies", async () => {
  const h = await fixture();
  const before = await h.store.get(h.selector.jobId);
  const result = await h.operations.inspect(h.selector);
  expect(result).toMatchObject({
    observation: "saved_metadata_only",
    currentEvidence: "not_revalidated",
    diagnosticPresent: true,
    job: { status: "interrupted" },
    workItems: [{ workItemId: "work:one", result: null }],
  });
  expect(JSON.stringify(result)).not.toMatch(/PRIVATE_|ownerProcessId|prompt/);
  expect(h.runtime.getSchedule).not.toHaveBeenCalled();
  expect(h.runtime.cancelSchedule).not.toHaveBeenCalled();
  expect(await h.store.get(h.selector.jobId)).toEqual(before);
});
it("pages all owned executions deterministically without reading runtime or changing records", async () => {
  const h = await fixture();
  const original = await h.store.get("job:one");
  if (!original) throw new Error("Missing fixture job");
  for (const [jobId, taskId] of [
    ["job:new-b", "task:one"],
    ["job:new-a", "task:one"],
    ["job:foreign", "task:other"],
  ] as const) {
    await h.store.create({
      ...original,
      jobId,
      taskId,
      request: { ...original.request, taskId },
      status: "queued",
      finishedAt: undefined,
      diagnostic: undefined,
      requestedAt: "2026-09-07T00:00:00.000Z",
    });
  }
  const first = await h.operations.list({ taskId: "task:one", limit: 2 });
  expect(first.schedules.map((job) => job.jobId)).toEqual(["job:new-a", "job:new-b"]);
  expect(first).toMatchObject({
    observation: "saved_metadata_only",
    currentEvidence: "not_revalidated",
    nextCursor: { digest: first.inventoryDigest, offset: 2 },
  });
  const second = await h.operations.list({
    taskId: "task:one",
    limit: 2,
    cursor: first.nextCursor,
  });
  expect(second.schedules.map((job) => job.jobId)).toEqual(["job:one"]);
  expect(second.nextCursor).toBeNull();
  expect(JSON.stringify([first, second])).not.toMatch(
    /PRIVATE_|prompt|diagnostic|ownerProcessId|job:foreign/,
  );
  expect(await h.store.get("job:one")).toEqual(original);
  expect((await h.store.get("job:new-a"))?.status).toBe("queued");
  expect(h.runtime.getSchedule).not.toHaveBeenCalled();
  expect(h.runtime.cancelSchedule).not.toHaveBeenCalled();
  await h.store.update("job:new-a", "queued", (job) => ({
    ...job,
    status: "running",
    startedAt: "2026-09-07T00:00:01.000Z",
  }));
  await expect(
    h.operations.list({ taskId: "task:one", limit: 2, cursor: first.nextCursor }),
  ).rejects.toThrow("history changed");
  const older = await h.operations.inspect({ taskId: "task:one", jobId: "job:one" });
  const cancelled = await h.operations.cancel({
    taskId: "task:one",
    jobId: "job:one",
    expectedJobIdentityDigest: older.jobIdentityDigest,
  });
  expect(cancelled.job).toMatchObject({
    jobId: "job:one",
    status: "interrupted",
    cancellationRequestedAt: expect.any(String),
  });
  expect((await h.store.get("job:new-a"))?.status).toBe("running");
  expect((await h.store.get("job:new-a"))?.cancellationRequestedAt).toBeUndefined();
  expect(h.runtime.cancelSchedule).toHaveBeenCalledTimes(1);
  expect(h.runtime.cancelSchedule).toHaveBeenCalledWith({ jobId: "job:one" });
});
it("returns an empty history for an owned Task with no schedule store", async () => {
  const h = await fixture();
  await rename(
    path.join(h.root, "runtime-schedules"),
    path.join(h.root, "fixture-archived-schedules"),
  );
  expect(await h.operations.list({ taskId: "task:one" })).toMatchObject({
    taskId: "task:one",
    schedules: [],
    nextCursor: null,
    currentEvidence: "not_revalidated",
  });
  expect(h.runtime.getSchedule).not.toHaveBeenCalled();
});
it("refuses stale, out-of-range and cross-Task history cursors and unauthorized Tasks", async () => {
  const h = await fixture();
  const first = await h.operations.list({ taskId: "task:one" });
  for (const input of [
    { taskId: "task:other" },
    { taskId: "task:one", limit: 101 },
    { taskId: "task:one", cursor: { digest: first.inventoryDigest, offset: 1 } },
    { taskId: "task:one", cursor: { digest: `sha256:${"e".repeat(64)}`, offset: 1 } },
  ])
    await expect(h.operations.list(input)).rejects.toThrow();
  await h.store.requestCancellation("job:one", "2026-09-07T00:00:00.000Z");
  // Cancellation intent is private detail, not a new list status or an exit claim.
  expect((await h.operations.list({ taskId: "task:one" })).schedules[0]?.status).toBe(
    "interrupted",
  );
});
it("refuses saved status changes between history observations", async () => {
  const h = await fixture();
  const read = FileRuntimeScheduleJobStore.prototype.listTaskSummaries;
  let calls = 0;
  vi.spyOn(FileRuntimeScheduleJobStore.prototype, "listTaskSummaries").mockImplementation(
    async function (this: FileRuntimeScheduleJobStore, ids) {
      const rows = await read.call(this, ids);
      if (++calls === 1)
        await h.store.update("job:one", "interrupted", (job) => ({
          ...job,
          status: "queued",
          finishedAt: undefined,
          diagnostic: undefined,
        }));
      return rows;
    },
  );
  await expect(h.operations.list({ taskId: "task:one" })).rejects.toThrow("history changed");
  expect(h.runtime.getSchedule).not.toHaveBeenCalled();
});
it("refuses ownership replacement during history capture", async () => {
  const h = await fixture();
  const read = FileRuntimeScheduleJobStore.prototype.listTaskSummaries;
  vi.spyOn(FileRuntimeScheduleJobStore.prototype, "listTaskSummaries").mockImplementation(
    async function (this: FileRuntimeScheduleJobStore, ids) {
      const rows = await read.call(this, ids);
      await writeFile(
        path.join(h.root, "knowledge", "local-principal.json"),
        JSON.stringify({
          schemaVersion: 1,
          organizationId: h.principal.organizationId,
          subjectId: randomUUID(),
        }),
      );
      return rows;
    },
  );
  await expect(h.operations.list({ taskId: "task:one" })).rejects.toThrow(
    "current local principal",
  );
});
it("requires exact registered owner, Task and reviewed identity before any command", async () => {
  const h = await fixture();
  await expect(h.operations.inspect({ ...h.selector, taskId: "task:other" })).rejects.toThrow();
  await expect(
    h.operations.cancel({ ...h.selector, expectedJobIdentityDigest: `sha256:${"e".repeat(64)}` }),
  ).rejects.toThrow("identity changed");
  await expect(
    h.operations.resume({ ...h.selector, expectedJobIdentityDigest: `sha256:${"e".repeat(64)}` }),
  ).rejects.toThrow();
  await writeFile(
    path.join(h.root, "knowledge", "local-principal.json"),
    JSON.stringify({
      schemaVersion: 1,
      organizationId: h.principal.organizationId,
      subjectId: randomUUID(),
    }),
  );
  await expect(h.operations.inspect(h.selector)).rejects.toThrow("current local principal");
  expect(h.runtime.getSchedule).not.toHaveBeenCalled();
  expect(h.runtime.cancelSchedule).not.toHaveBeenCalled();
});
it("delegates explicit resume once without creating or reconstructing an enqueue request", async () => {
  const h = await fixture();
  const inspected = await h.operations.inspect(h.selector);
  const result = await h.operations.resume({
    ...h.selector,
    expectedJobIdentityDigest: inspected.jobIdentityDigest,
    allowSubscriptionExecution: true,
  });
  expect(result.command).toEqual({ action: "resume", resumeBlocked: false });
  expect(h.runtime.getSchedule).toHaveBeenCalledTimes(1);
  expect(h.runtime.getSchedule).toHaveBeenCalledWith({ jobId: h.selector.jobId });
  h.runtime.getSchedule.mockRejectedValueOnce(new Error("lost response"));
  await expect(
    h.operations.resume({
      ...h.selector,
      expectedJobIdentityDigest: inspected.jobIdentityDigest,
      allowSubscriptionExecution: true,
    }),
  ).rejects.toThrow("lost response");
  expect(h.runtime.getSchedule).toHaveBeenCalledTimes(2);
});
it("persists cancellation for an interrupted schedule and suppresses later resume after restart", async () => {
  const h = await fixture();
  const inspected = await h.operations.inspect(h.selector);
  const result = await h.operations.cancel({
    ...h.selector,
    expectedJobIdentityDigest: inspected.jobIdentityDigest,
  });
  expect(result.job).toMatchObject({
    status: "interrupted",
    cancellationRequestedAt: "2026-09-06T00:00:02.000Z",
  });
  const prepare = vi.fn();
  const restarted = new RuntimeScheduleJobManager(new FileRuntimeScheduleJobStore(h.root));
  expect(await restarted.resume(h.selector.jobId, prepare)).toMatchObject({
    status: "interrupted",
    cancellationRequestedAt: expect.any(String),
  });
  expect(prepare).not.toHaveBeenCalled();
  expect((await h.operations.inspect(h.selector)).jobIdentityDigest).toBe(
    inspected.jobIdentityDigest,
  );
});
it("does not start prepared execution when cancellation races an interrupted resume", async () => {
  const h = await fixture();
  const execute = vi.fn();
  const result = await h.manager.resume(h.selector.jobId, async () => {
    await h.manager.cancel(h.selector.jobId);
    return execute;
  });
  expect(result).toMatchObject({
    status: "interrupted",
    cancellationRequestedAt: expect.any(String),
  });
  expect(execute).not.toHaveBeenCalled();
});
it("refuses a changed owner after a command without replaying it", async () => {
  const h = await fixture();
  const inspected = await h.operations.inspect(h.selector);
  h.runtime.getSchedule.mockImplementationOnce(async ({ jobId }) => {
    const result = await h.manager.get(jobId);
    await writeFile(
      path.join(h.root, "knowledge", "local-principal.json"),
      JSON.stringify({
        schemaVersion: 1,
        organizationId: h.principal.organizationId,
        subjectId: randomUUID(),
      }),
    );
    return result;
  });
  await expect(
    h.operations.resume({
      ...h.selector,
      expectedJobIdentityDigest: inspected.jobIdentityDigest,
      allowSubscriptionExecution: true,
    }),
  ).rejects.toThrow("current local principal");
  expect(h.runtime.getSchedule).toHaveBeenCalledTimes(1);
});
it("refuses unbound saved progress rather than presenting it as this Task's work", async () => {
  const h = await fixture();
  const original = FileRuntimeScheduleJobStore.prototype.get;
  vi.spyOn(FileRuntimeScheduleJobStore.prototype, "get").mockImplementation(async function (
    this: FileRuntimeScheduleJobStore,
    id,
  ) {
    const job = await original.call(this, id);
    return (
      job && {
        ...job,
        progress: {
          capabilities: [],
          results: [
            {
              workItemId: "work:foreign",
              status: "completed",
              attempts: 1,
              diagnostics: [],
              checkpoints: [],
            },
          ],
        },
      }
    );
  });
  await expect(h.operations.inspect(h.selector)).rejects.toThrow("progress identity mismatch");
});
it("routes inspection and cancellation through the actual host MCP without any provider execution", async () => {
  const h = await fixture();
  const client = new Client({ name: "registered-schedule-test", version: "1" });
  const hostToken = "d".repeat(64);
  try {
    await client.connect(
      new StdioClientTransport({
        command: process.env.KONTEXT_TEST_NODE_EXECUTABLE ?? process.execPath,
        args: [path.resolve("plugins/kontext-brain/server.mjs")],
        cwd: h.root,
        env: {
          KONTEXT_PLUGIN_DATA: h.root,
          KONTEXT_HOST_MANAGEMENT_TOKEN: hostToken,
          CODEX_HOME: h.root,
          CLAUDE_CONFIG_DIR: h.root,
          PATH: "",
          ELECTRON_RUN_AS_NODE: "1",
        },
        stderr: "pipe",
      }),
    );
    const listed = await client.callTool({
      name: "kontext_list_registered_schedules",
      arguments: { taskId: h.selector.taskId, hostToken },
    });
    expect(listed.isError).not.toBe(true);
    expect(listed.structuredContent).toMatchObject({
      taskId: h.selector.taskId,
      schedules: [{ jobId: h.selector.jobId, status: "interrupted" }],
      currentEvidence: "not_revalidated",
      nextCursor: null,
    });
    expect(
      (
        await client.callTool({
          name: "kontext_list_registered_schedules",
          arguments: { taskId: h.selector.taskId, hostToken: "e".repeat(64) },
        })
      ).isError,
    ).toBe(true);
    const inspected = await client.callTool({
      name: "kontext_inspect_registered_schedule",
      arguments: { ...h.selector, hostToken },
    });
    expect(inspected.isError).not.toBe(true);
    const integration = await client.callTool({
      name: "kontext_inspect_registered_integration",
      arguments: { ...h.selector, hostToken },
    });
    expect(integration.isError).not.toBe(true);
    expect(integration.structuredContent).toMatchObject({
      integration: null,
      canRequestIntegration: false,
      currentEvidence: "not_revalidated",
    });
    expect(
      (
        await client.callTool({
          name: "kontext_integrate_registered_schedule",
          arguments: {
            ...h.selector,
            hostToken,
            expectedJobIdentityDigest: (await h.operations.inspect(h.selector)).jobIdentityDigest,
            expectedIntegrationDigest: null,
            allowSubscriptionExecution: true,
          },
        })
      ).isError,
    ).toBe(true);
    const identity = (await h.operations.inspect(h.selector)).jobIdentityDigest;
    expect(inspected.structuredContent).toMatchObject({
      jobIdentityDigest: identity,
      job: { status: "interrupted" },
    });
    const request = { ...h.selector, expectedJobIdentityDigest: identity, hostToken };
    expect(
      (
        await client.callTool({
          name: "kontext_cancel_registered_schedule",
          arguments: { ...request, hostToken: "e".repeat(64) },
        })
      ).isError,
    ).toBe(true);
    const cancelled = await client.callTool({
      name: "kontext_cancel_registered_schedule",
      arguments: request,
    });
    expect(cancelled.isError).not.toBe(true);
    expect(cancelled.structuredContent).toMatchObject({
      job: { status: "interrupted", cancellationRequestedAt: expect.any(String) },
    });
    const resumed = await client.callTool({
      name: "kontext_resume_registered_schedule",
      arguments: { ...request, allowSubscriptionExecution: true },
    });
    expect(resumed.isError).not.toBe(true);
    expect(resumed.structuredContent).toMatchObject({
      job: { status: "interrupted", resumeCount: 0 },
    });
    expect(JSON.stringify([inspected, cancelled, resumed])).not.toMatch(/PRIVATE_/);
  } finally {
    await client.close();
  }
});
