import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { EnqueueVerificationRetryInput } from "@kontext-brain/orchestrator";
import { afterEach, describe, expect, it } from "vitest";
import { FileVerificationRetryQueue } from "../src/index.js";

const temporaryDirectories: string[] = [];
const retryInput: EnqueueVerificationRetryInput = {
  taskId: "task:durable-retry",
  workItemId: "work-item:durable",
  requirement: {
    tier: "targeted",
    verifier: { kind: "test", ref: "package:test" },
    subjectIds: ["work-item:durable"],
  },
  binding: {
    workspacePath: "/workspace",
    codeRevision: "commit:current",
    contextDigest: "context:current",
    observedAt: "2026-08-28T05:00:00.000Z",
  },
  verificationRunId: "verification:initial",
  maxRetries: 2,
  nextAttemptAt: "2026-08-28T05:01:00.000Z",
};

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("FileVerificationRetryQueue", () => {
  it("persists, claims, reschedules, and completes a retry across sidecar restarts", async () => {
    const directory = await temporaryDirectory();
    const first = new FileVerificationRetryQueue(directory);
    const enqueued = await first.enqueue(retryInput);
    const reopened = new FileVerificationRetryQueue(directory);

    expect(await reopened.list("queued")).toEqual([enqueued]);
    const [claimed] = await reopened.claimReady({
      taskId: retryInput.taskId,
      now: retryInput.nextAttemptAt,
      leaseExpiresAt: "2026-08-28T05:02:00.000Z",
      limit: 1,
    });
    if (!claimed) throw new Error("expected retry claim");
    expect(claimed).toEqual(expect.objectContaining({ status: "claimed", retryCount: 1 }));

    await reopened.reschedule(
      claimed,
      "verification:retry-1",
      "2026-08-28T05:03:00.000Z",
      "2026-08-28T05:01:30.000Z",
    );
    const restarted = new FileVerificationRetryQueue(directory);
    const [secondClaim] = await restarted.claimReady({
      taskId: retryInput.taskId,
      now: "2026-08-28T05:03:00.000Z",
      leaseExpiresAt: "2026-08-28T05:04:00.000Z",
      limit: 1,
    });
    if (!secondClaim) throw new Error("expected second retry claim");
    const completed = await restarted.complete(
      secondClaim,
      "verification:passed",
      "2026-08-28T05:03:30.000Z",
    );

    expect(completed).toEqual(
      expect.objectContaining({
        status: "completed",
        retryCount: 2,
        lastVerificationRunId: "verification:passed",
      }),
    );
    expect((await stat(restarted.jobFilePath(enqueued.jobId, "completed"))).mode & 0o777).toBe(
      0o600,
    );
  });

  it("supersedes a queued retry when the code revision changes", async () => {
    const queue = new FileVerificationRetryQueue(await temporaryDirectory());
    await queue.enqueue(retryInput);

    const superseded = await queue.supersedeObsolete(
      retryInput.taskId,
      { codeRevision: "commit:new", contextDigest: retryInput.binding.contextDigest },
      "2026-08-28T05:01:00.000Z",
    );

    expect(superseded).toEqual([expect.objectContaining({ status: "superseded" })]);
    expect(await queue.list("queued")).toEqual([]);
  });

  it("recovers an expired claim and rejects an obsolete claim token", async () => {
    const queue = new FileVerificationRetryQueue(await temporaryDirectory());
    await queue.enqueue(retryInput);
    const [expiredClaim] = await queue.claimReady({
      taskId: retryInput.taskId,
      now: retryInput.nextAttemptAt,
      leaseExpiresAt: "2026-08-28T05:01:30.000Z",
      limit: 1,
    });
    if (!expiredClaim) throw new Error("expected retry claim");

    const [recovered] = await queue.claimReady({
      taskId: retryInput.taskId,
      now: "2026-08-28T05:02:00.000Z",
      leaseExpiresAt: "2026-08-28T05:03:00.000Z",
      limit: 1,
    });
    if (!recovered) throw new Error("expected recovered retry claim");
    expect(recovered.retryCount).toBe(2);
    await expect(
      queue.complete(expiredClaim, "verification:stale", "2026-08-28T05:02:00.000Z"),
    ).rejects.toThrow("not held by this claim");
  });

  it("detects persisted retry payload tampering", async () => {
    const queue = new FileVerificationRetryQueue(await temporaryDirectory());
    const job = await queue.enqueue(retryInput);
    const filePath = queue.jobFilePath(job.jobId, "queued");
    const envelope = JSON.parse(await readFile(filePath, "utf8"));
    envelope.payload.codeRevision = "commit:tampered";
    await writeFile(filePath, JSON.stringify(envelope), "utf8");

    await expect(queue.list("queued")).rejects.toThrow("digest mismatch");
  });
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-verification-retry-"));
  temporaryDirectories.push(directory);
  return directory;
}
