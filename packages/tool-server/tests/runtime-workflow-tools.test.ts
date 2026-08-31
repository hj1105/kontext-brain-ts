import { describe, expect, it } from "vitest";
import {
  type KontextRuntimeOperations,
  KontextRuntimeToolRouter,
  type ScheduleLogicRequest,
  applyRiskProviderPolicy,
} from "../src/index.js";

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
