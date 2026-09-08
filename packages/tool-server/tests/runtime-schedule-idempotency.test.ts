import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FileRuntimeScheduleJobStore, RuntimeScheduleJobManager } from "../src/index.js";

const directories: string[] = [];
const request = {
  requestId: "b8a2a3f8-7fb9-44da-a868-1d4019164729",
  taskId: "task:deduplicate",
  repositoryPath: "/repository",
  work: [
    {
      workItemId: "logic:one",
      prompt: "Implement approved logic",
      eligibleProviders: ["codex" as const],
    },
  ],
};

afterEach(async () => {
  await Promise.all(directories.splice(0).map((directory) => rm(directory, { recursive: true })));
});

async function fixture() {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-enqueue-identity-"));
  directories.push(directory);
  const manager = new RuntimeScheduleJobManager(new FileRuntimeScheduleJobStore(directory));
  const execute = vi.fn(async () => ({ capabilities: [], results: [] }));
  const prepare = vi.fn(async () => ({
    codeRevision: "revision:one",
    contextDigest: "digest:one",
    execute,
  }));
  return { directory, manager, execute, prepare };
}

async function settled(manager: RuntimeScheduleJobManager, jobId: string) {
  for (let attempt = 0; attempt < 200; attempt++) {
    const job = await manager.get(jobId);
    if (job.status === "completed") return job;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("Fixture schedule did not settle");
}

describe("Runtime Schedule enqueue identity", () => {
  it("recovers a lost enqueue response after restart without preparing or executing again", async () => {
    const h = await fixture();
    const accepted = await h.manager.enqueue(request, h.prepare);
    await settled(h.manager, accepted.jobId);
    const restarted = new RuntimeScheduleJobManager(new FileRuntimeScheduleJobStore(h.directory));
    const unavailablePreparation = vi.fn(async () => {
      throw new Error("Current context is no longer available");
    });

    const recovered = await restarted.enqueue(request, unavailablePreparation);

    expect(recovered).toMatchObject({
      jobId: accepted.jobId,
      requestId: request.requestId,
      status: "completed",
      codeRevision: "revision:one",
      contextDigest: "digest:one",
    });
    expect(recovered).not.toHaveProperty("request");
    expect(unavailablePreparation).not.toHaveBeenCalled();
    expect(h.execute).toHaveBeenCalledTimes(1);
  });

  it("atomically accepts only one execution across independent stores", async () => {
    const h = await fixture();
    const managers = Array.from(
      { length: 8 },
      () => new RuntimeScheduleJobManager(new FileRuntimeScheduleJobStore(h.directory)),
    );
    const jobs = await Promise.all(managers.map((manager) => manager.enqueue(request, h.prepare)));
    const ids = new Set(jobs.map((job) => job.jobId));
    expect(ids.size).toBe(1);
    const first = jobs[0];
    if (!first) throw new Error("No schedule returned");
    await settled(h.manager, first.jobId);
    expect(h.execute).toHaveBeenCalledTimes(1);
  });

  it.each([
    { taskId: "task:other" },
    { repositoryPath: "/other" },
    { maxConcurrency: 4 },
    { work: request.work.map((work) => ({ ...work, prompt: "Different instructions" })) },
  ])("refuses a reused key with different input: %j", async (change) => {
    const h = await fixture();
    const accepted = await h.manager.enqueue(request, h.prepare);
    await settled(h.manager, accepted.jobId);
    h.prepare.mockClear();
    await expect(h.manager.enqueue({ ...request, ...change }, h.prepare)).rejects.toThrow(
      "different request",
    );
    expect(h.prepare).not.toHaveBeenCalled();
    expect(h.execute).toHaveBeenCalledTimes(1);
  });

  it("compares canonical payloads rather than object key order", async () => {
    const h = await fixture();
    const first = await h.manager.enqueue(request, h.prepare);
    await settled(h.manager, first.jobId);
    const reordered = {
      work: request.work,
      repositoryPath: request.repositoryPath,
      taskId: request.taskId,
      requestId: request.requestId,
    };
    expect((await h.manager.enqueue(reordered, h.prepare)).jobId).toBe(first.jobId);
    expect(h.prepare).toHaveBeenCalledTimes(1);
  });

  it("allows a deliberate new request and preserves legacy unkeyed enqueue behavior", async () => {
    const h = await fixture();
    const { requestId: _requestId, ...legacy } = request;
    const jobs = [];
    for (const input of [request, { ...request, requestId: randomUUID() }, legacy, legacy]) {
      const job = await h.manager.enqueue(input, h.prepare);
      jobs.push(await settled(h.manager, job.jobId));
    }
    expect(new Set(jobs.map((job) => job.jobId)).size).toBe(4);
    expect(h.execute).toHaveBeenCalledTimes(4);
  });

  it("rejects malformed request identity before preparation", async () => {
    const h = await fixture();
    await expect(
      h.manager.enqueue({ ...request, requestId: "../invalid" }, h.prepare),
    ).rejects.toThrow();
    expect(h.prepare).not.toHaveBeenCalled();
  });

  it("does not resume an orphan when retrying an enqueue whose persistence acknowledgement was lost", async () => {
    const h = await fixture();
    const store = new FileRuntimeScheduleJobStore(h.directory);
    const create = store.create.bind(store);
    let jobId = "";
    vi.spyOn(store, "create").mockImplementation(async (job) => {
      await create(job);
      jobId = job.jobId;
      throw new Error("Sidecar stopped after persistence");
    });
    const interruptedOwner = new RuntimeScheduleJobManager(store);
    await expect(interruptedOwner.enqueue(request, h.prepare)).rejects.toThrow("after persistence");
    const restarted = new RuntimeScheduleJobManager(
      new FileRuntimeScheduleJobStore(h.directory),
      undefined,
      undefined,
      process.pid,
      () => false,
    );
    expect((await restarted.get(jobId)).status).toBe("interrupted");
    h.prepare.mockClear();

    expect(await restarted.enqueue(request, h.prepare)).toMatchObject({
      jobId,
      status: "interrupted",
      resumeCount: 0,
    });
    expect(h.prepare).not.toHaveBeenCalled();
    expect(h.execute).not.toHaveBeenCalled();
  });
});
