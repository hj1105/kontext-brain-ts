import type { LogicWorkItem } from "@kontext-brain/spec";
import { describe, expect, it } from "vitest";
import {
  type AgentRuntimePort,
  InMemoryRuntimeLeaseStore,
  type RuntimeProvider,
  RuntimeScheduleCancelledError,
  type RuntimeSession,
  type RuntimeWorkInput,
  type RuntimeWorktreePort,
  WorkItemScheduler,
  createRuntimeCapabilitySnapshot,
} from "../src/index.js";

describe("WorkItemScheduler", () => {
  it("runs independent work concurrently while serializing shared paths", async () => {
    let active = 0;
    let maximumActive = 0;
    const starts: string[] = [];
    const runtime = fakeRuntime("codex", async (input) => {
      active += 1;
      maximumActive = Math.max(maximumActive, active);
      starts.push(input.workItem.workItemId);
      await new Promise((resolve) => setTimeout(resolve, 20));
      active -= 1;
      return session("codex", input.workItem.workItemId, "completed");
    });
    const scheduler = new WorkItemScheduler(
      [runtime],
      fakeWorktrees(),
      new InMemoryRuntimeLeaseStore(),
    );

    const result = await scheduler.run({
      taskId: "task:schedule",
      maxConcurrency: 4,
      work: [
        scheduled(workItem("work:a", ["src/shared.ts"])),
        scheduled(workItem("work:b", ["src/independent.ts"])),
        scheduled(workItem("work:c", ["src/shared.ts"])),
      ],
    });

    expect(maximumActive).toBe(2);
    expect(starts.indexOf("work:c")).toBeGreaterThan(starts.indexOf("work:a"));
    expect(result.results.every((item) => item.status === "completed")).toBe(true);
  });

  it("retries a failed checkpoint on another subscription runtime without cross-provider resume", async () => {
    let codexStarts = 0;
    let claudeStarts = 0;
    let resumes = 0;
    const codex = fakeRuntime(
      "codex",
      async (input) => {
        codexStarts += 1;
        return session("codex", input.workItem.workItemId, "failed");
      },
      () => {
        resumes += 1;
      },
    );
    const claude = fakeRuntime(
      "claude",
      async (input) => {
        claudeStarts += 1;
        expect(input.checkpoint?.provider).toBe("codex");
        return session("claude", input.workItem.workItemId, "completed");
      },
      () => {
        resumes += 1;
      },
    );
    const scheduler = new WorkItemScheduler(
      [codex, claude],
      fakeWorktrees(),
      new InMemoryRuntimeLeaseStore(),
    );

    const result = await scheduler.run({
      taskId: "task:transfer",
      maxRetries: 2,
      work: [
        {
          ...scheduled(workItem("work:transfer", ["src/transfer.ts"])),
          eligibleProviders: ["codex", "claude"],
        },
      ],
    });

    expect(codexStarts).toBe(1);
    expect(claudeStarts).toBe(1);
    expect(resumes).toBe(0);
    expect(result.results[0]).toEqual(
      expect.objectContaining({
        status: "completed",
        provider: "claude",
        attempts: 2,
        checkpoints: [
          expect.objectContaining({ provider: "codex" }),
          expect.objectContaining({ provider: "claude" }),
        ],
      }),
    );
  });

  it("aborts an active worker and releases its write lease before cancellation completes", async () => {
    const controller = new AbortController();
    let reportStarted = (): void => undefined;
    const started = new Promise<void>((resolve) => {
      reportStarted = resolve;
    });
    const leases = new InMemoryRuntimeLeaseStore();
    const runtime = fakeRuntime("codex", async (input) => {
      reportStarted();
      return new Promise((resolve) => {
        const cancel = () => resolve(session("codex", input.workItem.workItemId, "terminated"));
        input.signal?.addEventListener("abort", cancel, { once: true });
        if (input.signal?.aborted) cancel();
      });
    });
    const scheduler = new WorkItemScheduler([runtime], fakeWorktrees(), leases);
    const scheduledRun = scheduler.run({
      taskId: "task:cancel",
      signal: controller.signal,
      work: [scheduled(workItem("work:cancel", ["src/cancel.ts"]))],
    });

    await started;
    controller.abort();

    await expect(scheduledRun).rejects.toBeInstanceOf(RuntimeScheduleCancelledError);
    await expect(leases.listActive("2026-08-29T00:02:00.000Z")).resolves.toEqual([]);
  });

  it("persists incremental results and resumes only unfinished work from a validated checkpoint", async () => {
    const starts: string[] = [];
    const progress: string[][] = [];
    const runtime = fakeRuntime("codex", async (input) => {
      starts.push(input.workItem.workItemId);
      return session("codex", input.workItem.workItemId, "completed");
    });
    const scheduler = new WorkItemScheduler(
      [runtime],
      fakeWorktrees(),
      new InMemoryRuntimeLeaseStore(),
    );
    const first = scheduled(workItem("work:finished", ["src/finished.ts"]));
    const second = scheduled(workItem("work:remaining", ["src/remaining.ts"]));

    const result = await scheduler.run({
      taskId: "task:schedule",
      work: [first, second],
      initialResults: [
        {
          workItemId: "work:finished",
          status: "completed",
          provider: "codex",
          worktree: {
            worktreeId: "worktree:finished",
            workspacePath: "/worktrees/work:finished",
            branchName: "kontext/work:finished",
            baseRevision: first.codeRevision,
          },
          session: session("codex", "work:finished", "completed"),
          attempts: 1,
          checkpoints: [],
          diagnostics: [],
        },
      ],
      onProgress: (current) => {
        progress.push(current.results.map((item) => item.workItemId));
      },
    });

    expect(starts).toEqual(["work:remaining"]);
    expect(progress).toEqual([["work:finished", "work:remaining"]]);
    expect(result.results).toHaveLength(2);
  });

  it("rejects a durable checkpoint bound to a different revision", async () => {
    const scheduler = new WorkItemScheduler(
      [
        fakeRuntime("codex", async (input) =>
          session("codex", input.workItem.workItemId, "completed"),
        ),
      ],
      fakeWorktrees(),
      new InMemoryRuntimeLeaseStore(),
    );

    await expect(
      scheduler.run({
        taskId: "task:schedule",
        work: [scheduled(workItem("work:stale", ["src/stale.ts"]))],
        initialResults: [
          {
            workItemId: "work:stale",
            status: "completed",
            attempts: 1,
            checkpoints: [
              {
                checkpointId: "runtime-checkpoint:stale",
                taskId: "task:schedule",
                workItemId: "work:stale",
                provider: "codex",
                workspacePath: "/worktrees/work:stale",
                codeRevision: "commit:stale",
                contextDigest: "context:current",
                createdAt: "2026-08-29T00:01:00.000Z",
              },
            ],
            diagnostics: [],
          },
        ],
      }),
    ).rejects.toThrow("does not match current work");
  });
});

function workItem(workItemId: string, allowedPaths: readonly string[]): LogicWorkItem {
  return {
    workItemId,
    taskId: "task:schedule",
    plannedSymbolIds: [`symbol:${workItemId}`],
    dependsOn: [],
    allowedPaths,
    requiredVerifiers: [],
    capabilityId: `capability:${workItemId}`,
  };
}

function scheduled(item: LogicWorkItem) {
  return {
    workItem: item,
    prompt: `Implement ${item.workItemId}`,
    codeRevision: "commit:base",
    contextDigest: "context:current",
    eligibleProviders: ["codex"] as const,
  };
}

function fakeRuntime(
  provider: RuntimeProvider,
  start: (input: RuntimeWorkInput) => Promise<RuntimeSession>,
  onResume: () => void = () => undefined,
): AgentRuntimePort {
  return {
    provider,
    inspectCapabilities: async () =>
      createRuntimeCapabilitySnapshot({
        provider,
        cliPath: provider,
        cliVersion: "test",
        installed: true,
        authenticated: true,
        billingPath: "subscription",
        supports: {
          structuredOutput: true,
          sessionResume: true,
          mcp: true,
          hooks: true,
          workspaceSandbox: true,
        },
        inspectedAt: "2026-08-29T00:00:00.000Z",
      }),
    start,
    resume: async (_sessionId, input) => {
      onResume();
      return start(input);
    },
    terminate: async () => undefined,
  };
}

function fakeWorktrees(): RuntimeWorktreePort {
  return {
    prepare: async ({ taskId, workItem, baseRevision }) => ({
      worktreeId: `worktree:${workItem.workItemId}`,
      workspacePath: `/worktrees/${workItem.workItemId}`,
      branchName: `kontext/${workItem.workItemId}`,
      baseRevision,
    }),
  };
}

function session(
  provider: RuntimeProvider,
  workItemId: string,
  status: RuntimeSession["status"],
): RuntimeSession {
  return {
    sessionId: `session:${provider}:${workItemId}`,
    provider,
    providerSessionId: `${provider}-${workItemId}`,
    status,
    output: status,
    events: [],
    startedAt: "2026-08-29T00:00:00.000Z",
    completedAt: "2026-08-29T00:01:00.000Z",
  };
}
