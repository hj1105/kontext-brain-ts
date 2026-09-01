import { createHash, randomUUID } from "node:crypto";
import type {
  ClaimVerificationRetriesInput,
  EnqueueVerificationRetryInput,
  VerificationBinding,
  VerificationExecution,
  VerificationPlan,
  VerificationRetryJob,
  VerificationRetryQueue,
  VerificationRetryStatus,
} from "./domain.js";
import { VerificationCoordinator } from "./verifier-registry.js";

export interface ExecutePlanWithRetryInput {
  readonly taskId: string;
  readonly workItemId?: string;
  readonly plan: VerificationPlan;
  readonly binding: VerificationBinding;
  readonly maxRetries?: number;
  readonly nextAttemptAt: string;
}

export interface RetryAvailableVerificationInput {
  readonly taskId: string;
  readonly currentCodeRevision: string;
  readonly currentContextDigest: string;
  readonly observedAt: string;
  readonly nextAttemptAt: string;
  readonly leaseExpiresAt: string;
  readonly limit?: number;
}

export class DurableVerificationCoordinator {
  constructor(
    private readonly coordinator: VerificationCoordinator,
    private readonly retryQueue: VerificationRetryQueue,
  ) {}

  async executePlan(input: ExecutePlanWithRetryInput): Promise<readonly VerificationExecution[]> {
    const executions = await this.coordinator.executePlan(input.plan, input.binding);
    await Promise.all(
      executions.map(async (execution, index) => {
        const requirement = input.plan.requirements[index];
        if (!requirement || execution.disposition !== "retryable") return;
        await this.retryQueue.enqueue({
          taskId: input.taskId,
          workItemId: input.workItemId,
          requirement,
          binding: input.binding,
          verificationRunId: execution.run.verificationRunId,
          maxRetries: input.maxRetries ?? 2,
          nextAttemptAt: input.nextAttemptAt,
        });
      }),
    );
    return executions;
  }

  async retryAvailable(
    input: RetryAvailableVerificationInput,
  ): Promise<readonly VerificationExecution[]> {
    await this.retryQueue.supersedeObsolete(
      input.taskId,
      {
        codeRevision: input.currentCodeRevision,
        contextDigest: input.currentContextDigest,
      },
      input.observedAt,
    );
    const jobs = await this.retryQueue.claimReady({
      taskId: input.taskId,
      now: input.observedAt,
      leaseExpiresAt: input.leaseExpiresAt,
      limit: input.limit ?? 32,
    });
    const executions: VerificationExecution[] = [];
    for (const job of jobs) {
      const execution = await this.coordinator.execute(job.requirement, {
        workspacePath: job.workspacePath,
        codeRevision: job.codeRevision,
        contextDigest: job.contextDigest,
        observedAt: input.observedAt,
      });
      executions.push(execution);
      if (execution.disposition === "settled") {
        await this.retryQueue.complete(job, execution.run.verificationRunId, input.observedAt);
      } else {
        await this.retryQueue.reschedule(
          job,
          execution.run.verificationRunId,
          input.nextAttemptAt,
          input.observedAt,
        );
      }
    }
    return executions;
  }
}

export class InMemoryVerificationRetryQueue implements VerificationRetryQueue {
  private readonly jobs = new Map<string, VerificationRetryJob>();

  async enqueue(input: EnqueueVerificationRetryInput): Promise<VerificationRetryJob> {
    const jobId = verificationRetryJobId(input);
    const existing = this.jobs.get(jobId);
    if (existing) return existing;
    const job: VerificationRetryJob = {
      jobId,
      taskId: input.taskId,
      workItemId: input.workItemId,
      requirement: input.requirement,
      workspacePath: input.binding.workspacePath,
      codeRevision: input.binding.codeRevision,
      contextDigest: input.binding.contextDigest,
      status: "queued",
      retryCount: 0,
      maxRetries: input.maxRetries,
      nextAttemptAt: input.nextAttemptAt,
      initialVerificationRunId: input.verificationRunId,
      lastVerificationRunId: input.verificationRunId,
      createdAt: input.binding.observedAt,
      updatedAt: input.binding.observedAt,
    };
    this.jobs.set(jobId, job);
    return job;
  }

  async claimReady(input: ClaimVerificationRetriesInput): Promise<readonly VerificationRetryJob[]> {
    const ready = Array.from(this.jobs.values())
      .filter(
        (job) =>
          (job.status === "queued" ||
            (job.status === "claimed" &&
              job.leaseExpiresAt !== undefined &&
              job.leaseExpiresAt <= input.now)) &&
          job.taskId === input.taskId &&
          job.nextAttemptAt <= input.now,
      )
      .sort(compareJobs)
      .slice(0, Math.max(0, input.limit));
    return ready.map((job) => {
      const claimed: VerificationRetryJob = {
        ...job,
        status: "claimed",
        retryCount: job.retryCount + 1,
        claimId: randomUUID(),
        leaseExpiresAt: input.leaseExpiresAt,
        updatedAt: input.now,
      };
      this.jobs.set(job.jobId, claimed);
      return claimed;
    });
  }

  async complete(
    job: VerificationRetryJob,
    verificationRunId: string,
    updatedAt: string,
  ): Promise<VerificationRetryJob> {
    return this.finishClaim(job, "completed", verificationRunId, updatedAt);
  }

  async reschedule(
    job: VerificationRetryJob,
    verificationRunId: string,
    nextAttemptAt: string,
    updatedAt: string,
  ): Promise<VerificationRetryJob> {
    const status = job.retryCount >= job.maxRetries ? "exhausted" : "queued";
    const updated = this.requireClaim(job);
    const next: VerificationRetryJob = {
      ...updated,
      status,
      nextAttemptAt,
      lastVerificationRunId: verificationRunId,
      updatedAt,
      claimId: undefined,
      leaseExpiresAt: undefined,
    };
    this.jobs.set(job.jobId, next);
    return next;
  }

  async supersedeObsolete(
    taskId: string,
    binding: Pick<VerificationBinding, "codeRevision" | "contextDigest">,
    updatedAt: string,
  ): Promise<readonly VerificationRetryJob[]> {
    const superseded: VerificationRetryJob[] = [];
    for (const job of this.jobs.values()) {
      if (
        job.status === "queued" &&
        job.taskId === taskId &&
        (job.codeRevision !== binding.codeRevision || job.contextDigest !== binding.contextDigest)
      ) {
        const next: VerificationRetryJob = { ...job, status: "superseded", updatedAt };
        this.jobs.set(job.jobId, next);
        superseded.push(next);
      }
    }
    return superseded.sort(compareJobs);
  }

  async list(status?: VerificationRetryStatus): Promise<readonly VerificationRetryJob[]> {
    return Array.from(this.jobs.values())
      .filter((job) => status === undefined || job.status === status)
      .sort(compareJobs);
  }

  private finishClaim(
    job: VerificationRetryJob,
    status: "completed",
    verificationRunId: string,
    updatedAt: string,
  ): VerificationRetryJob {
    const current = this.requireClaim(job);
    const next: VerificationRetryJob = {
      ...current,
      status,
      lastVerificationRunId: verificationRunId,
      updatedAt,
      claimId: undefined,
      leaseExpiresAt: undefined,
    };
    this.jobs.set(job.jobId, next);
    return next;
  }

  private requireClaim(job: VerificationRetryJob): VerificationRetryJob {
    const current = this.jobs.get(job.jobId);
    if (
      !current ||
      current.status !== "claimed" ||
      !job.claimId ||
      current.claimId !== job.claimId
    ) {
      throw new Error(`Verification retry ${job.jobId} is not held by this claim`);
    }
    return current;
  }
}

export function verificationRetryJobId(input: EnqueueVerificationRetryInput): string {
  return `verification-retry:${createHash("sha256")
    .update(
      stableJson({
        taskId: input.taskId,
        workItemId: input.workItemId,
        requirement: input.requirement,
        workspacePath: input.binding.workspacePath,
        codeRevision: input.binding.codeRevision,
        contextDigest: input.binding.contextDigest,
      }),
    )
    .digest("hex")}`;
}

function compareJobs(left: VerificationRetryJob, right: VerificationRetryJob): number {
  return (
    left.nextAttemptAt.localeCompare(right.nextAttemptAt) || left.jobId.localeCompare(right.jobId)
  );
}

function stableJson(value: unknown): string {
  return JSON.stringify(stableValue(value));
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, nested]) => [key, stableValue(nested)]),
    );
  }
  return value;
}
