import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { WorkItemScheduleResult } from "@kontext-brain/orchestrator";
import { afterEach, describe, expect, it } from "vitest";
import {
  FileRuntimeScheduleJobStore,
  RuntimeScheduleJobManager,
  type ScheduleLogicRequest,
} from "../src/index.js";

const temporaryDirectories: string[] = [];
const request: ScheduleLogicRequest = {
  taskId: "task:async",
  repositoryPath: "/repository",
  work: [
    {
      workItemId: "work-item:handler",
      prompt: "Implement the private behavior.",
      eligibleProviders: ["codex"],
    },
  ],
};

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("RuntimeScheduleJobManager", () => {
  it("returns immediately, keeps prompts private, and persists the terminal result", async () => {
    const directory = await temporaryDirectory();
    const store = new FileRuntimeScheduleJobStore(directory);
    let complete: ((result: WorkItemScheduleResult) => void) | undefined;
    const execution = new Promise<WorkItemScheduleResult>((resolve) => {
      complete = resolve;
    });
    const manager = new RuntimeScheduleJobManager(
      store,
      () => new Date("2026-08-31T01:02:03.000Z"),
      () => "runtime-schedule:test",
    );

    const queued = await manager.enqueue(
      request,
      "commit:async",
      "sha256:context",
      () => execution,
    );

    expect(queued).toMatchObject({
      jobId: "runtime-schedule:test",
      taskId: "task:async",
      status: "queued",
    });
    expect(queued).not.toHaveProperty("request");
    await expect(waitForStatus(manager, "runtime-schedule:test", "running")).resolves.toMatchObject(
      { status: "running" },
    );

    complete?.({ capabilities: [], results: [] });
    const completed = await waitForStatus(manager, "runtime-schedule:test", "completed");
    expect(completed).toMatchObject({
      status: "completed",
      result: { capabilities: [], results: [] },
    });
    expect(completed).not.toHaveProperty("request");
    expect((await stat(scheduleFile(directory, "runtime-schedule:test"))).mode & 0o777).toBe(0o600);
  });

  it("marks a persisted orphan as interrupted after sidecar restart", async () => {
    const directory = await temporaryDirectory();
    const store = new FileRuntimeScheduleJobStore(directory);
    await store.create({
      jobId: "runtime-schedule:orphan",
      taskId: request.taskId,
      request,
      codeRevision: "commit:orphan",
      contextDigest: "sha256:orphan",
      status: "queued",
      requestedAt: "2026-08-31T01:00:00.000Z",
      ownerInstanceId: "runtime-scheduler:stopped",
      ownerProcessId: 4242,
    });
    await store.update("runtime-schedule:orphan", "queued", (job) => ({
      ...job,
      status: "running",
      startedAt: "2026-08-31T01:00:01.000Z",
      ownerInstanceId: "runtime-scheduler:stopped",
      ownerProcessId: 4242,
    }));
    const concurrentReader = new RuntimeScheduleJobManager(
      store,
      () => new Date("2026-08-31T01:04:00.000Z"),
      undefined,
      5151,
      (processId) => processId === 4242,
    );
    await expect(concurrentReader.get("runtime-schedule:orphan")).resolves.toMatchObject({
      status: "running",
    });
    const restarted = new RuntimeScheduleJobManager(
      store,
      () => new Date("2026-08-31T01:05:00.000Z"),
      undefined,
      5252,
      () => false,
    );

    await expect(restarted.get("runtime-schedule:orphan")).resolves.toMatchObject({
      status: "interrupted",
      finishedAt: "2026-08-31T01:05:00.000Z",
      diagnostic: expect.stringContaining("sidecar process stopped"),
    });
  });

  it("persists execution failure as a terminal diagnostic", async () => {
    const directory = await temporaryDirectory();
    const manager = new RuntimeScheduleJobManager(
      new FileRuntimeScheduleJobStore(directory),
      () => new Date("2026-08-31T02:00:00.000Z"),
      () => "runtime-schedule:failed",
    );
    await manager.enqueue(request, "commit:failed", "sha256:failed", () =>
      Promise.reject(new Error("runtime unavailable")),
    );

    await expect(
      waitForStatus(manager, "runtime-schedule:failed", "failed"),
    ).resolves.toMatchObject({
      status: "failed",
      diagnostic: "runtime unavailable",
    });
  });

  it("does not report cancelled until the active worker acknowledges abort", async () => {
    const directory = await temporaryDirectory();
    const manager = new RuntimeScheduleJobManager(
      new FileRuntimeScheduleJobStore(directory),
      () => new Date("2026-08-31T02:30:00.000Z"),
      () => "runtime-schedule:cancelled",
    );
    let acknowledgeAbort = (): void => undefined;
    const workerStopped = new Promise<WorkItemScheduleResult>((resolve) => {
      acknowledgeAbort = () => resolve({ capabilities: [], results: [] });
    });
    let receivedSignal: AbortSignal | undefined;
    await manager.enqueue(request, "commit:cancelled", "sha256:cancelled", (signal) => {
      receivedSignal = signal;
      return workerStopped;
    });
    await waitForStatus(manager, "runtime-schedule:cancelled", "running");

    await expect(manager.cancel("runtime-schedule:cancelled")).resolves.toMatchObject({
      status: "cancelling",
      cancellationRequestedAt: "2026-08-31T02:30:00.000Z",
    });
    expect(receivedSignal?.aborted).toBe(true);
    await expect(manager.get("runtime-schedule:cancelled")).resolves.toMatchObject({
      status: "cancelling",
    });

    acknowledgeAbort();
    await expect(
      waitForStatus(manager, "runtime-schedule:cancelled", "cancelled"),
    ).resolves.toMatchObject({
      status: "cancelled",
      diagnostic: expect.stringContaining("active workers stopped"),
    });
  });

  it("serializes completion and cancellation across independent store instances", async () => {
    const directory = await temporaryDirectory();
    const completionStore = new FileRuntimeScheduleJobStore(directory);
    const cancellationStore = new FileRuntimeScheduleJobStore(directory);
    await completionStore.create({
      jobId: "runtime-schedule:race",
      taskId: request.taskId,
      request,
      codeRevision: "commit:race",
      contextDigest: "sha256:race",
      status: "queued",
      requestedAt: "2026-08-31T03:00:00.000Z",
      ownerInstanceId: "runtime-scheduler:race",
      ownerProcessId: process.pid,
    });
    await completionStore.update("runtime-schedule:race", "queued", (job) => ({
      ...job,
      status: "running",
      startedAt: "2026-08-31T03:00:01.000Z",
    }));

    const outcomes = await Promise.allSettled([
      completionStore.update("runtime-schedule:race", "running", (job) => ({
        ...job,
        status: "completed",
        finishedAt: "2026-08-31T03:00:02.000Z",
        result: { capabilities: [], results: [] },
      })),
      cancellationStore.update("runtime-schedule:race", "running", (job) => ({
        ...job,
        status: "cancelling",
        cancellationRequestedAt: "2026-08-31T03:00:02.000Z",
      })),
    ]);

    expect(outcomes.filter((outcome) => outcome.status === "fulfilled")).toHaveLength(1);
    expect(outcomes.filter((outcome) => outcome.status === "rejected")).toHaveLength(1);
    await expect(completionStore.get("runtime-schedule:race")).resolves.toEqual(
      expect.objectContaining({ status: expect.stringMatching(/^(completed|cancelling)$/) }),
    );
  });

  it("delivers a cancellation request from another manager to the owning worker", async () => {
    const directory = await temporaryDirectory();
    const owner = new RuntimeScheduleJobManager(
      new FileRuntimeScheduleJobStore(directory),
      () => new Date("2026-08-31T03:30:00.000Z"),
      () => "runtime-schedule:remote-cancel",
    );
    await owner.enqueue(
      request,
      "commit:remote",
      "sha256:remote",
      (signal) =>
        new Promise((resolve) => {
          const stop = () => resolve({ capabilities: [], results: [] });
          signal.addEventListener("abort", stop, { once: true });
          if (signal.aborted) stop();
        }),
    );
    await waitForStatus(owner, "runtime-schedule:remote-cancel", "running");
    const remote = new RuntimeScheduleJobManager(
      new FileRuntimeScheduleJobStore(directory),
      () => new Date("2026-08-31T03:30:01.000Z"),
      undefined,
      process.pid,
      () => true,
    );

    await expect(remote.cancel("runtime-schedule:remote-cancel")).resolves.toMatchObject({
      status: "cancelling",
    });
    await expect(
      waitForStatus(owner, "runtime-schedule:remote-cancel", "cancelled"),
    ).resolves.toMatchObject({ status: "cancelled" });
  });

  it("rejects a schedule file whose private payload was modified", async () => {
    const directory = await temporaryDirectory();
    const store = new FileRuntimeScheduleJobStore(directory);
    await store.create({
      jobId: "runtime-schedule:tampered",
      taskId: request.taskId,
      request,
      codeRevision: "commit:original",
      contextDigest: "sha256:original",
      status: "queued",
      requestedAt: "2026-08-31T01:00:00.000Z",
      ownerInstanceId: "runtime-scheduler:tampered",
      ownerProcessId: 6262,
    });
    const filePath = scheduleFile(directory, "runtime-schedule:tampered");
    const envelope = JSON.parse(await readFile(filePath, "utf8"));
    envelope.payload.codeRevision = "commit:tampered";
    await writeFile(filePath, `${JSON.stringify(envelope)}\n`);

    await expect(store.get("runtime-schedule:tampered")).rejects.toThrow("payload digest mismatch");
  });
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-runtime-schedule-"));
  temporaryDirectories.push(directory);
  return directory;
}

function scheduleFile(directory: string, jobId: string): string {
  return path.join(
    directory,
    "runtime-schedules",
    `${createHash("sha256").update(jobId).digest("hex")}.json`,
  );
}

async function waitForStatus(
  manager: RuntimeScheduleJobManager,
  jobId: string,
  status: string,
): Promise<Awaited<ReturnType<RuntimeScheduleJobManager["get"]>>> {
  for (let attempt = 0; attempt < 100; attempt++) {
    const job = await manager.get(jobId);
    if (job.status === status) return job;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error(`Runtime schedule job ${jobId} did not reach ${status}`);
}
