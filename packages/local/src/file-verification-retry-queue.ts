import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  type ClaimVerificationRetriesInput,
  type EnqueueVerificationRetryInput,
  type VerificationBinding,
  type VerificationRetryJob,
  type VerificationRetryQueue,
  type VerificationRetryStatus,
  verificationRetryJobId,
} from "@kontext-brain/orchestrator";
import { z } from "zod";

const statuses = ["queued", "claimed", "completed", "superseded", "exhausted"] as const;
const nonEmptyString = z.string().min(1);
const jobSchema = z
  .object({
    jobId: nonEmptyString,
    taskId: nonEmptyString,
    workItemId: nonEmptyString.optional(),
    requirement: z
      .object({
        tier: z.enum(["fast", "targeted", "full"]),
        verifier: z
          .object({
            kind: z.enum(["test", "typecheck", "build", "lint", "query", "manual_review"]),
            ref: nonEmptyString,
          })
          .strict(),
        subjectIds: z.array(nonEmptyString),
      })
      .strict(),
    workspacePath: nonEmptyString,
    codeRevision: nonEmptyString,
    contextDigest: nonEmptyString,
    status: z.enum(statuses),
    retryCount: z.number().int().nonnegative(),
    maxRetries: z.number().int().nonnegative(),
    nextAttemptAt: nonEmptyString,
    initialVerificationRunId: nonEmptyString,
    lastVerificationRunId: nonEmptyString,
    createdAt: nonEmptyString,
    updatedAt: nonEmptyString,
    claimId: nonEmptyString.optional(),
    leaseExpiresAt: nonEmptyString.optional(),
  })
  .strict();

interface VerificationRetryEnvelope {
  readonly schemaVersion: 1;
  readonly kind: "verification_retry";
  readonly payloadDigest: string;
  readonly payload: VerificationRetryJob;
}

export class FileVerificationRetryQueue implements VerificationRetryQueue {
  constructor(private readonly pluginDataDirectory: string) {}

  async enqueue(input: EnqueueVerificationRetryInput): Promise<VerificationRetryJob> {
    if (input.maxRetries < 0 || !Number.isInteger(input.maxRetries)) {
      throw new Error("Verification retry maxRetries must be a non-negative integer");
    }
    const jobId = verificationRetryJobId(input);
    const existing = await this.find(jobId);
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
    await atomicPrivateWrite(this.jobFilePath(jobId, "queued"), encode(job));
    return job;
  }

  async claimReady(input: ClaimVerificationRetriesInput): Promise<readonly VerificationRetryJob[]> {
    await this.recoverExpiredClaims(input.now);
    const queued = (await this.list("queued"))
      .filter((job) => job.taskId === input.taskId && job.nextAttemptAt <= input.now)
      .slice(0, Math.max(0, input.limit));
    const claimed: VerificationRetryJob[] = [];
    for (const job of queued) {
      const source = this.jobFilePath(job.jobId, "queued");
      const target = this.jobFilePath(job.jobId, "claimed");
      const next: VerificationRetryJob = {
        ...job,
        status: "claimed",
        retryCount: job.retryCount + 1,
        claimId: randomUUID(),
        leaseExpiresAt: input.leaseExpiresAt,
        updatedAt: input.now,
      };
      try {
        await atomicPrivateWrite(source, encode(next));
        await mkdir(path.dirname(target), { recursive: true, mode: 0o700 });
        await rename(source, target);
        claimed.push(next);
      } catch (error) {
        if (isNodeError(error) && error.code === "ENOENT") continue;
        throw error;
      }
    }
    return claimed;
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
    return this.finishClaim(job, status, verificationRunId, updatedAt, nextAttemptAt);
  }

  async supersedeObsolete(
    taskId: string,
    binding: Pick<VerificationBinding, "codeRevision" | "contextDigest">,
    updatedAt: string,
  ): Promise<readonly VerificationRetryJob[]> {
    const obsolete = (await this.list("queued")).filter(
      (job) =>
        job.taskId === taskId &&
        (job.codeRevision !== binding.codeRevision || job.contextDigest !== binding.contextDigest),
    );
    const superseded: VerificationRetryJob[] = [];
    for (const job of obsolete) {
      const next: VerificationRetryJob = { ...job, status: "superseded", updatedAt };
      try {
        await this.move(job, next);
        superseded.push(next);
      } catch (error) {
        if (isNodeError(error) && error.code === "ENOENT") continue;
        throw error;
      }
    }
    return superseded.sort(compareJobs);
  }

  async list(status?: VerificationRetryStatus): Promise<readonly VerificationRetryJob[]> {
    const selected = status ? [status] : statuses;
    const jobs: VerificationRetryJob[] = [];
    for (const candidateStatus of selected) {
      const directory = this.statusDirectory(candidateStatus);
      let entries: string[];
      try {
        entries = await readdir(directory);
      } catch (error) {
        if (isNodeError(error) && error.code === "ENOENT") continue;
        throw error;
      }
      for (const entry of entries.filter((value) => value.endsWith(".json")).sort()) {
        const job = await readJob(path.join(directory, entry));
        if (job.status !== candidateStatus) {
          throw new Error(`Verification retry ${job.jobId} status does not match its location`);
        }
        jobs.push(job);
      }
    }
    return jobs.sort(compareJobs);
  }

  jobFilePath(jobId: string, status: VerificationRetryStatus): string {
    return path.join(
      this.statusDirectory(status),
      `${createHash("sha256").update(jobId).digest("hex")}.json`,
    );
  }

  private async recoverExpiredClaims(now: string): Promise<void> {
    const expired = (await this.list("claimed")).filter(
      (job) => job.leaseExpiresAt !== undefined && job.leaseExpiresAt <= now,
    );
    for (const job of expired) {
      const queued: VerificationRetryJob = {
        ...job,
        status: "queued",
        claimId: undefined,
        leaseExpiresAt: undefined,
        updatedAt: now,
      };
      await this.move(job, queued);
    }
  }

  private async finishClaim(
    job: VerificationRetryJob,
    status: "queued" | "completed" | "exhausted",
    verificationRunId: string,
    updatedAt: string,
    nextAttemptAt = job.nextAttemptAt,
  ): Promise<VerificationRetryJob> {
    const current = await readJob(this.jobFilePath(job.jobId, "claimed"));
    if (!job.claimId || current.claimId !== job.claimId) {
      throw new Error(`Verification retry ${job.jobId} is not held by this claim`);
    }
    const next: VerificationRetryJob = {
      ...current,
      status,
      nextAttemptAt,
      lastVerificationRunId: verificationRunId,
      updatedAt,
      claimId: undefined,
      leaseExpiresAt: undefined,
    };
    await this.move(current, next);
    return next;
  }

  private async move(current: VerificationRetryJob, next: VerificationRetryJob): Promise<void> {
    const source = this.jobFilePath(current.jobId, current.status);
    const target = this.jobFilePath(next.jobId, next.status);
    await atomicPrivateWrite(source, encode(next));
    await mkdir(path.dirname(target), { recursive: true, mode: 0o700 });
    await rename(source, target);
  }

  private async find(jobId: string): Promise<VerificationRetryJob | undefined> {
    for (const status of statuses) {
      try {
        return await readJob(this.jobFilePath(jobId, status));
      } catch (error) {
        if (isNodeError(error) && error.code === "ENOENT") continue;
        throw error;
      }
    }
    return undefined;
  }

  private statusDirectory(status: VerificationRetryStatus): string {
    return path.join(this.pluginDataDirectory, "verification-retries", status);
  }
}

function encode(job: VerificationRetryJob): string {
  const payload = jobSchema.parse(job) as VerificationRetryJob;
  const envelope: VerificationRetryEnvelope = {
    schemaVersion: 1,
    kind: "verification_retry",
    payloadDigest: digest(payload),
    payload,
  };
  return `${JSON.stringify(envelope, null, 2)}\n`;
}

async function readJob(filePath: string): Promise<VerificationRetryJob> {
  const parsed: unknown = JSON.parse(await readFile(filePath, "utf8"));
  const envelope = z
    .object({
      schemaVersion: z.literal(1),
      kind: z.literal("verification_retry"),
      payloadDigest: nonEmptyString,
      payload: jobSchema,
    })
    .strict()
    .parse(parsed) as VerificationRetryEnvelope;
  if (digest(envelope.payload) !== envelope.payloadDigest) {
    throw new Error("Verification retry payload digest mismatch");
  }
  return envelope.payload;
}

async function atomicPrivateWrite(filePath: string, serialized: string): Promise<void> {
  const directory = path.dirname(filePath);
  const temporaryPath = path.join(directory, `.${randomUUID()}.tmp`);
  await mkdir(directory, { recursive: true, mode: 0o700 });
  try {
    await writeFile(temporaryPath, serialized, { encoding: "utf8", mode: 0o600 });
    await rename(temporaryPath, filePath);
  } catch (error) {
    await rm(temporaryPath, { force: true }).catch(() => undefined);
    throw error;
  }
}

function digest(value: unknown): string {
  return `sha256:${createHash("sha256")
    .update(JSON.stringify(stableValue(value)))
    .digest("hex")}`;
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

function compareJobs(left: VerificationRetryJob, right: VerificationRetryJob): number {
  return (
    left.nextAttemptAt.localeCompare(right.nextAttemptAt) || left.jobId.localeCompare(right.jobId)
  );
}

function isNodeError(value: unknown): value is NodeJS.ErrnoException {
  return value instanceof Error && "code" in value;
}
