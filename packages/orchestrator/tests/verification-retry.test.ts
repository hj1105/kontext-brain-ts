import { describe, expect, it } from "vitest";
import {
  DurableVerificationCoordinator,
  InMemoryVerificationRetryQueue,
  VerificationCoordinator,
  VerifierInfrastructureError,
  VerifierRegistry,
  createFastVerificationPlan,
} from "../src/index.js";

const firstBinding = {
  workspacePath: "/workspace",
  codeRevision: "commit:first",
  contextDigest: "context:first",
  observedAt: "2026-08-28T04:00:00.000Z",
};
const taskId = "task:verification-retry";

describe("durable verification retry orchestration", () => {
  it("retries an inconclusive verifier against the same revision and context after recovery", async () => {
    let available = false;
    const registry = new VerifierRegistry();
    registry.register(
      { kind: "query", ref: "kontext:semantic-sync" },
      {
        execute: async () => {
          if (!available) throw new VerifierInfrastructureError("index unavailable");
          return { result: "passed", output: "synchronized" };
        },
      },
    );
    const queue = new InMemoryVerificationRetryQueue();
    const durable = new DurableVerificationCoordinator(
      new VerificationCoordinator(registry),
      queue,
    );
    const plan = {
      ...createFastVerificationPlan({ affectedSymbolIds: ["symbol:a"] }),
      requirements: createFastVerificationPlan({
        affectedSymbolIds: ["symbol:a"],
      }).requirements.filter((requirement) => requirement.verifier.ref === "kontext:semantic-sync"),
    };

    const initial = await durable.executePlan({
      taskId,
      plan,
      binding: firstBinding,
      nextAttemptAt: "2026-08-28T04:01:00.000Z",
    });
    expect(initial[0]?.run.result).toBe("inconclusive");
    expect(await queue.list("queued")).toHaveLength(1);

    available = true;
    const retried = await durable.retryAvailable({
      taskId,
      currentCodeRevision: firstBinding.codeRevision,
      currentContextDigest: firstBinding.contextDigest,
      observedAt: "2026-08-28T04:01:00.000Z",
      nextAttemptAt: "2026-08-28T04:02:00.000Z",
      leaseExpiresAt: "2026-08-28T04:02:00.000Z",
    });

    expect(retried[0]?.run).toEqual(
      expect.objectContaining({
        result: "passed",
        codeRevision: firstBinding.codeRevision,
        contextDigest: firstBinding.contextDigest,
      }),
    );
    expect(await queue.list("completed")).toHaveLength(1);
  });

  it("supersedes queued retries after a newer edit changes the revision", async () => {
    const registry = new VerifierRegistry();
    const queue = new InMemoryVerificationRetryQueue();
    const durable = new DurableVerificationCoordinator(
      new VerificationCoordinator(registry),
      queue,
    );
    const plan = {
      tier: "fast" as const,
      requirements: [
        {
          tier: "fast" as const,
          verifier: { kind: "query" as const, ref: "missing:verifier" },
          subjectIds: ["symbol:a"],
        },
      ],
    };
    await durable.executePlan({
      taskId,
      plan,
      binding: firstBinding,
      nextAttemptAt: "2026-08-28T04:01:00.000Z",
    });

    const retried = await durable.retryAvailable({
      taskId,
      currentCodeRevision: "commit:newer",
      currentContextDigest: firstBinding.contextDigest,
      observedAt: "2026-08-28T04:01:00.000Z",
      nextAttemptAt: "2026-08-28T04:02:00.000Z",
      leaseExpiresAt: "2026-08-28T04:02:00.000Z",
    });

    expect(retried).toEqual([]);
    expect(await queue.list("superseded")).toHaveLength(1);
  });

  it("exhausts the same failure after at most two retries", async () => {
    const registry = new VerifierRegistry();
    const queue = new InMemoryVerificationRetryQueue();
    const durable = new DurableVerificationCoordinator(
      new VerificationCoordinator(registry),
      queue,
    );
    const plan = {
      tier: "targeted" as const,
      requirements: [
        {
          tier: "targeted" as const,
          verifier: { kind: "test" as const, ref: "missing:test" },
          subjectIds: ["work-item:a"],
        },
      ],
    };
    await durable.executePlan({
      taskId,
      plan,
      binding: firstBinding,
      maxRetries: 2,
      nextAttemptAt: "2026-08-28T04:01:00.000Z",
    });
    await durable.retryAvailable({
      taskId,
      currentCodeRevision: firstBinding.codeRevision,
      currentContextDigest: firstBinding.contextDigest,
      observedAt: "2026-08-28T04:01:00.000Z",
      nextAttemptAt: "2026-08-28T04:02:00.000Z",
      leaseExpiresAt: "2026-08-28T04:01:30.000Z",
    });
    await durable.retryAvailable({
      taskId,
      currentCodeRevision: firstBinding.codeRevision,
      currentContextDigest: firstBinding.contextDigest,
      observedAt: "2026-08-28T04:02:00.000Z",
      nextAttemptAt: "2026-08-28T04:03:00.000Z",
      leaseExpiresAt: "2026-08-28T04:02:30.000Z",
    });

    expect(await queue.list("exhausted")).toEqual([
      expect.objectContaining({ retryCount: 2, maxRetries: 2 }),
    ]);
  });
});
