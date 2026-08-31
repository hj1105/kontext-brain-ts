import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { FileRuntimeLeaseStore } from "@kontext-brain/local";
import type { AgentRuntimePort } from "@kontext-brain/orchestrator";
import { createRuntimeCapabilitySnapshot } from "@kontext-brain/orchestrator";
import { afterEach, describe, expect, it } from "vitest";
import {
  FileRuntimeScheduleJobStore,
  type KontextRuntimeOperations,
  KontextRuntimeToolRouter,
  LocalKontextRuntimeOperations,
  type ScheduleLogicRequest,
  applyRiskProviderPolicy,
} from "../src/index.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("KontextRuntimeToolRouter", () => {
  it("validates bounded provider selection, concurrency, retries, and token budgets", async () => {
    const operations = new RecordingRuntimeOperations();
    const router = new KontextRuntimeToolRouter(operations);
    await router.inspectRuntimes({});
    await router.getSchedule({ jobId: "runtime-schedule:one" });
    await router.cancelSchedule({ jobId: "runtime-schedule:one" });
    await router.integrateSchedule({
      jobId: "runtime-schedule:one",
      observedAt: "2026-08-31T03:00:00.000Z",
      nextAttemptAt: "2026-08-31T03:05:00.000Z",
    });
    await router.scheduleLogic({
      taskId: "task:runtime",
      repositoryPath: "/repository",
      work: [
        {
          workItemId: "work-item:handler",
          prompt: "Implement the handler.",
          eligibleProviders: ["codex", "claude"],
          pinnedProvider: "codex",
          totalTokenBudget: 20_000,
          optionalEvidenceTokenBudget: 2_000,
          receiptTtlSeconds: 900,
        },
      ],
      maxConcurrency: 4,
      maxRetries: 2,
    });

    expect(operations.inspections).toBe(1);
    expect(operations.requestedJobId).toBe("runtime-schedule:one");
    expect(operations.cancelledJobId).toBe("runtime-schedule:one");
    expect(operations.integratedJobId).toBe("runtime-schedule:one");
    expect(operations.scheduled).toEqual(
      expect.objectContaining({
        taskId: "task:runtime",
        maxConcurrency: 4,
        maxRetries: 2,
      }),
    );
    await expect(
      router.scheduleLogic({
        taskId: "task:runtime",
        repositoryPath: "/repository",
        work: [
          {
            workItemId: "work-item:handler",
            prompt: "Implement.",
            eligibleProviders: ["codex"],
          },
        ],
        maxConcurrency: 5,
      }),
    ).rejects.toThrow();
    await expect(router.getSchedule({ jobId: "", extra: true })).rejects.toThrow();
    await expect(router.cancelSchedule({ jobId: "", extra: true })).rejects.toThrow();
  });
});

describe("risk-based provider policy", () => {
  const work = [
    { eligibleProviders: ["codex", "claude"] as const },
    { eligibleProviders: ["codex", "claude"] as const },
  ];

  it("keeps one implementation provider available for a different medium-risk reviewer", () => {
    expect(applyRiskProviderPolicy(work, "medium")).toEqual([
      { eligibleProviders: ["codex"], pinnedProvider: "codex" },
      { eligibleProviders: ["codex"], pinnedProvider: "codex" },
    ]);
  });

  it("uses Claude implementation for Codex-led high-risk planning and rejects a Codex pin", () => {
    expect(applyRiskProviderPolicy(work, "high")).toEqual([
      { eligibleProviders: ["claude"], pinnedProvider: "claude" },
      { eligibleProviders: ["claude"], pinnedProvider: "claude" },
    ]);
    expect(() =>
      applyRiskProviderPolicy(
        [{ eligibleProviders: ["codex", "claude"] as const, pinnedProvider: "codex" as const }],
        "high",
      ),
    ).toThrow("provider policy");
  });
});

describe("automatic durable schedule resume", () => {
  it("revalidates current context and completes from persisted Work Item progress", async () => {
    const directory = await temporaryDirectory();
    const store = new FileRuntimeScheduleJobStore(directory);
    const scheduleRequest = resumeRequest();
    await orphanSchedule(store, scheduleRequest, true);
    let runtimeStarts = 0;
    const operations = localOperations(directory, "freshness:current", () => {
      runtimeStarts += 1;
    });

    await expect(
      operations.getSchedule({ jobId: "runtime-schedule:auto-resume" }),
    ).resolves.toEqual(expect.objectContaining({ status: "queued", resumeCount: 1 }));
    await expect(waitForCompleted(operations)).resolves.toEqual(
      expect.objectContaining({
        status: "completed",
        resumeCount: 1,
        result: expect.objectContaining({
          results: [expect.objectContaining({ workItemId: "work-item:handler" })],
        }),
      }),
    );
    expect(runtimeStarts).toBe(0);
  });

  it("keeps an orphan interrupted when its frozen context is no longer current", async () => {
    const directory = await temporaryDirectory();
    const store = new FileRuntimeScheduleJobStore(directory);
    await orphanSchedule(store, resumeRequest(), false);
    let runtimeStarts = 0;
    const operations = localOperations(directory, "freshness:changed", () => {
      runtimeStarts += 1;
    });

    await expect(
      operations.getSchedule({ jobId: "runtime-schedule:auto-resume" }),
    ).resolves.toEqual(
      expect.objectContaining({
        status: "interrupted",
        resumeBlocked: true,
        resumeDiagnostic: expect.stringContaining("original current revision and context digest"),
      }),
    );
    expect(runtimeStarts).toBe(0);
  });

  it("waits for an orphaned write lease to expire before starting unfinished work", async () => {
    const directory = await temporaryDirectory();
    const store = new FileRuntimeScheduleJobStore(directory);
    await orphanSchedule(store, resumeRequest(), false);
    await new FileRuntimeLeaseStore(directory).acquire({
      leaseId: "runtime-lease:orphan",
      taskId: "task:auto-resume",
      workItemId: "work-item:handler",
      provider: "codex",
      workspacePath: "/worktrees/handler",
      symbolIds: ["symbol:handler"],
      paths: ["src/handler.ts"],
      acquiredAt: "2026-08-31T01:00:00.000Z",
      expiresAt: "2026-08-31T01:15:00.000Z",
    });
    let runtimeStarts = 0;
    const operations = localOperations(directory, "freshness:current", () => {
      runtimeStarts += 1;
    });

    await expect(
      operations.getSchedule({ jobId: "runtime-schedule:auto-resume" }),
    ).resolves.toEqual(
      expect.objectContaining({
        status: "interrupted",
        resumeBlocked: true,
        resumeDiagnostic: expect.stringContaining("waiting for write lease runtime-lease:orphan"),
      }),
    );
    expect(runtimeStarts).toBe(0);
  });

  it("keeps unfinished recovery interrupted while its subscription runtime is unavailable", async () => {
    const directory = await temporaryDirectory();
    const store = new FileRuntimeScheduleJobStore(directory);
    await orphanSchedule(store, resumeRequest(), false);
    let runtimeStarts = 0;
    const operations = localOperations(
      directory,
      "freshness:current",
      () => {
        runtimeStarts += 1;
      },
      false,
    );

    await expect(
      operations.getSchedule({ jobId: "runtime-schedule:auto-resume" }),
    ).resolves.toEqual(
      expect.objectContaining({
        status: "interrupted",
        resumeBlocked: true,
        resumeDiagnostic: expect.stringContaining(
          "No authenticated subscription runtime is currently available",
        ),
      }),
    );
    expect(runtimeStarts).toBe(0);
  });
});

class RecordingRuntimeOperations implements KontextRuntimeOperations {
  inspections = 0;
  scheduled?: ScheduleLogicRequest;
  requestedJobId?: string;
  cancelledJobId?: string;
  integratedJobId?: string;

  async inspectRuntimes(): Promise<unknown> {
    this.inspections += 1;
    return { eligibleProviders: ["codex"] };
  }

  async scheduleLogic(request: ScheduleLogicRequest): Promise<unknown> {
    this.scheduled = request;
    return { jobId: "runtime-schedule:one", status: "queued" };
  }

  async getSchedule(request: { readonly jobId: string }): Promise<unknown> {
    this.requestedJobId = request.jobId;
    return { jobId: request.jobId, status: "running" };
  }

  async cancelSchedule(request: { readonly jobId: string }): Promise<unknown> {
    this.cancelledJobId = request.jobId;
    return { jobId: request.jobId, status: "cancelling" };
  }

  async integrateSchedule(request: { readonly jobId: string }): Promise<unknown> {
    this.integratedJobId = request.jobId;
    return { jobId: request.jobId, resultRevision: "workspace:integrated" };
  }
}

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-runtime-resume-"));
  temporaryDirectories.push(directory);
  return directory;
}

function resumeRequest(): ScheduleLogicRequest {
  return {
    taskId: "task:auto-resume",
    repositoryPath: "/repository",
    work: [
      {
        workItemId: "work-item:handler",
        prompt: "Finish the handler from durable workspace state.",
        eligibleProviders: ["codex"],
      },
    ],
  };
}

async function orphanSchedule(
  store: FileRuntimeScheduleJobStore,
  scheduleRequest: ScheduleLogicRequest,
  withProgress: boolean,
): Promise<void> {
  await store.create({
    jobId: "runtime-schedule:auto-resume",
    taskId: scheduleRequest.taskId,
    request: scheduleRequest,
    codeRevision: "commit:auto-resume",
    contextDigest: "sha256:auto-resume",
    status: "queued",
    requestedAt: "2026-08-31T01:00:00.000Z",
    ownerInstanceId: "runtime-scheduler:stopped",
    ownerProcessId: 999_999,
  });
  await store.update("runtime-schedule:auto-resume", "queued", (job) => ({
    ...job,
    status: "running",
    startedAt: "2026-08-31T01:00:01.000Z",
  }));
  if (withProgress) {
    await store.recordProgress("runtime-schedule:auto-resume", {
      capabilities: [],
      results: [
        {
          workItemId: "work-item:handler",
          status: "completed",
          provider: "codex",
          attempts: 1,
          checkpoints: [],
          diagnostics: [],
        },
      ],
    });
  }
}

function localOperations(
  directory: string,
  sourceFreshnessDigest: string,
  onRuntimeStart: () => void,
  runtimeAvailable = true,
): LocalKontextRuntimeOperations {
  const runtime: AgentRuntimePort = {
    provider: "codex",
    inspectCapabilities: async () =>
      createRuntimeCapabilitySnapshot({
        provider: "codex",
        cliPath: "codex",
        cliVersion: "test",
        installed: runtimeAvailable,
        authenticated: runtimeAvailable,
        billingPath: "subscription",
        supports: {
          structuredOutput: true,
          sessionResume: true,
          mcp: true,
          hooks: true,
          workspaceSandbox: true,
        },
        inspectedAt: "2026-08-31T01:05:00.000Z",
      }),
    start: async () => {
      onRuntimeStart();
      throw new Error("A completed checkpoint must not restart its runtime");
    },
    resume: async () => {
      throw new Error("Automatic recovery must start a fresh provider session");
    },
    terminate: async () => undefined,
  };
  return new LocalKontextRuntimeOperations(
    {
      getCurrent: async () => ({
        codeRevision: "commit:auto-resume",
        sourceFreshnessDigest,
        effectiveScopes: [],
        normativeRecords: [],
        normativeRevisionCatalog: [],
        conflicts: [],
        evidence: [],
        logicPlans: [
          {
            workItemId: "work-item:handler",
            plannedSymbolIds: ["symbol:handler"],
            allowedPaths: ["src/handler.ts"],
          },
        ],
      }),
    },
    {
      get: async () => ({
        contract: { taskId: "task:auto-resume", risk: "low" },
        snapshot: {
          contextDigest: "sha256:auto-resume",
          sourceFreshnessDigest: "freshness:current",
          effectiveScopes: [],
          normativeRevisions: [],
          requiredEvidenceIds: [],
        },
        additionalRequiredEvidenceIds: [],
      }),
    } as never,
    {} as never,
    {} as never,
    directory,
    [runtime],
    {} as never,
    {} as never,
    {} as never,
    {} as never,
    {} as never,
    () => new Date("2026-08-31T01:05:00.000Z"),
  );
}

async function waitForCompleted(operations: LocalKontextRuntimeOperations): Promise<unknown> {
  for (let attempt = 0; attempt < 100; attempt++) {
    const result = (await operations.getSchedule({
      jobId: "runtime-schedule:auto-resume",
    })) as { readonly status?: string };
    if (result.status === "completed") return result;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("Automatically resumed schedule did not complete");
}
