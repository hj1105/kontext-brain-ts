import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import type { WorkItemScheduleResult } from "@kontext-brain/orchestrator";
import { z } from "zod";
import {
  type ScheduleLogicRequest,
  scheduleLogicRequestSchema,
} from "./runtime-schedule-contract.js";

const nonEmptyString = z.string().min(1);
const mutationLockOwnerSchema = z
  .object({
    lockId: nonEmptyString,
    processId: z.number().int().positive(),
  })
  .strict();
const scheduleStatusSchema = z.enum([
  "queued",
  "running",
  "cancelling",
  "completed",
  "failed",
  "interrupted",
  "cancelled",
]);
const scheduleJobSchema = z
  .object({
    jobId: nonEmptyString,
    taskId: nonEmptyString,
    request: scheduleLogicRequestSchema,
    codeRevision: nonEmptyString,
    contextDigest: nonEmptyString,
    status: scheduleStatusSchema,
    requestedAt: z.string().datetime(),
    cancellationRequestedAt: z.string().datetime().optional(),
    startedAt: z.string().datetime().optional(),
    finishedAt: z.string().datetime().optional(),
    ownerInstanceId: nonEmptyString.optional(),
    ownerProcessId: z.number().int().positive().optional(),
    result: z.unknown().optional(),
    diagnostic: nonEmptyString.optional(),
  })
  .strict();

export type RuntimeScheduleStatus = z.infer<typeof scheduleStatusSchema>;

export interface RuntimeScheduleJob {
  readonly jobId: string;
  readonly taskId: string;
  readonly request: ScheduleLogicRequest;
  readonly codeRevision: string;
  readonly contextDigest: string;
  readonly status: RuntimeScheduleStatus;
  readonly requestedAt: string;
  readonly cancellationRequestedAt?: string;
  readonly startedAt?: string;
  readonly finishedAt?: string;
  readonly ownerInstanceId?: string;
  readonly ownerProcessId?: number;
  readonly result?: WorkItemScheduleResult;
  readonly diagnostic?: string;
}

export interface RuntimeScheduleJobView {
  readonly jobId: string;
  readonly taskId: string;
  readonly codeRevision: string;
  readonly contextDigest: string;
  readonly status: RuntimeScheduleStatus;
  readonly requestedAt: string;
  readonly cancellationRequestedAt?: string;
  readonly startedAt?: string;
  readonly finishedAt?: string;
  readonly result?: WorkItemScheduleResult;
  readonly diagnostic?: string;
}

interface RuntimeScheduleJobEnvelope {
  readonly schemaVersion: 1;
  readonly kind: "runtime_schedule_job";
  readonly payloadDigest: string;
  readonly payload: RuntimeScheduleJob;
}

export class FileRuntimeScheduleJobStore {
  private operation: Promise<void> = Promise.resolve();

  constructor(
    private readonly pluginDataDirectory: string,
    private readonly processId = process.pid,
    private readonly ownerIsAlive: (processId: number) => boolean = isProcessAlive,
  ) {}

  async create(job: RuntimeScheduleJob): Promise<void> {
    await this.mutate(job.jobId, async () => {
      assertValidJob(job);
      if (job.status !== "queued") {
        throw new Error("A new runtime schedule job must be queued");
      }
      if (await this.getUnsafe(job.jobId)) {
        throw new Error(`Runtime schedule job ${job.jobId} already exists`);
      }
      await atomicPrivateWrite(this.filePath(job.jobId), encode(job));
    });
  }

  async get(jobId: string): Promise<RuntimeScheduleJob | undefined> {
    return this.exclusive(() => this.getUnsafe(jobId));
  }

  async update(
    jobId: string,
    expectedStatus: RuntimeScheduleStatus | readonly RuntimeScheduleStatus[],
    update: (current: RuntimeScheduleJob) => RuntimeScheduleJob,
  ): Promise<RuntimeScheduleJob> {
    return this.mutate(jobId, async () => {
      const current = await this.getUnsafe(jobId);
      if (!current) throw new Error(`Runtime schedule job ${jobId} does not exist`);
      const expected = Array.isArray(expectedStatus) ? expectedStatus : [expectedStatus];
      if (!expected.includes(current.status)) {
        throw new Error(
          `Runtime schedule job ${jobId} is ${current.status}; expected ${expected.join(" or ")}`,
        );
      }
      const next = update(current);
      assertImmutableFields(current, next);
      assertValidTransition(current.status, next.status);
      assertValidJob(next);
      await atomicPrivateWrite(this.filePath(jobId), encode(next));
      return next;
    });
  }

  private async getUnsafe(jobId: string): Promise<RuntimeScheduleJob | undefined> {
    try {
      const job = await readJob(this.filePath(jobId));
      if (job.jobId !== jobId) {
        throw new Error("Runtime schedule job does not match its storage location");
      }
      return job;
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") return undefined;
      throw error;
    }
  }

  private filePath(jobId: string): string {
    return path.join(
      this.pluginDataDirectory,
      "runtime-schedules",
      `${createHash("sha256").update(jobId).digest("hex")}.json`,
    );
  }

  private lockPath(jobId: string): string {
    return path.join(
      this.pluginDataDirectory,
      "runtime-schedule-locks",
      `${createHash("sha256").update(jobId).digest("hex")}.lock`,
    );
  }

  private async mutate<T>(jobId: string, operation: () => Promise<T>): Promise<T> {
    return this.exclusive(async () => {
      const release = await this.acquireMutationLock(jobId);
      try {
        return await operation();
      } finally {
        await release();
      }
    });
  }

  private async acquireMutationLock(jobId: string): Promise<() => Promise<void>> {
    const lockPath = this.lockPath(jobId);
    const owner = { lockId: `runtime-schedule-lock:${randomUUID()}`, processId: this.processId };
    await mkdir(path.dirname(lockPath), { recursive: true, mode: 0o700 });
    for (let attempt = 0; attempt < 200; attempt++) {
      try {
        await writeFile(lockPath, `${JSON.stringify(owner)}\n`, {
          encoding: "utf8",
          mode: 0o600,
          flag: "wx",
        });
        return async () => {
          const current = await this.readMutationLockOwner(lockPath);
          if (current?.lockId === owner.lockId) await moveAsideFile(lockPath);
        };
      } catch (error) {
        if (!isNodeError(error) || error.code !== "EEXIST") throw error;
      }
      const current = await this.readMutationLockOwner(lockPath);
      if (current && !this.ownerIsAlive(current.processId)) {
        await moveAsideFile(lockPath);
        continue;
      }
      if (!current && (await fileOlderThan(lockPath, 1_000))) {
        await moveAsideFile(lockPath);
        continue;
      }
      await delay(10);
    }
    throw new Error(`Runtime schedule job ${jobId} mutation lock is busy`);
  }

  private async readMutationLockOwner(
    lockPath: string,
  ): Promise<z.infer<typeof mutationLockOwnerSchema> | undefined> {
    try {
      return mutationLockOwnerSchema.parse(JSON.parse(await readFile(lockPath, "utf8")));
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") return undefined;
      if (error instanceof SyntaxError || error instanceof z.ZodError) return undefined;
      throw error;
    }
  }

  private async exclusive<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.operation;
    let release = (): void => undefined;
    this.operation = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    try {
      return await operation();
    } finally {
      release();
    }
  }
}

export class RuntimeScheduleJobManager {
  private readonly active = new Map<
    string,
    { readonly execution: Promise<void>; readonly controller: AbortController }
  >();
  private readonly instanceId: string;

  constructor(
    private readonly store: FileRuntimeScheduleJobStore,
    private readonly now: () => Date = () => new Date(),
    private readonly newJobId: () => string = () => `runtime-schedule:${randomUUID()}`,
    private readonly processId = process.pid,
    private readonly ownerIsAlive: (processId: number) => boolean = isProcessAlive,
  ) {
    this.instanceId = `runtime-scheduler:${randomUUID()}`;
  }

  async enqueue(
    request: ScheduleLogicRequest,
    codeRevision: string,
    contextDigest: string,
    execute: (signal: AbortSignal) => Promise<WorkItemScheduleResult>,
  ): Promise<RuntimeScheduleJobView> {
    const job: RuntimeScheduleJob = {
      jobId: this.newJobId(),
      taskId: request.taskId,
      request,
      codeRevision,
      contextDigest,
      status: "queued",
      requestedAt: this.now().toISOString(),
      ownerInstanceId: this.instanceId,
      ownerProcessId: this.processId,
    };
    await this.store.create(job);
    const controller = new AbortController();
    const execution = this.execute(job.jobId, controller, execute);
    this.active.set(job.jobId, { execution, controller });
    void execution.then(
      () => this.active.delete(job.jobId),
      () => this.active.delete(job.jobId),
    );
    return publicView(job);
  }

  async get(jobId: string): Promise<RuntimeScheduleJobView> {
    const job = await this.requiredJob(jobId);
    return publicView(await this.reconcileOrphan(job));
  }

  async cancel(jobId: string): Promise<RuntimeScheduleJobView> {
    let job = await this.reconcileOrphan(await this.requiredJob(jobId));
    if (isTerminal(job.status)) return publicView(job);
    if (job.status !== "cancelling") {
      try {
        job = await this.store.update(jobId, ["queued", "running"], (current) => ({
          ...current,
          status: "cancelling",
          cancellationRequestedAt: this.now().toISOString(),
        }));
      } catch {
        job = (await this.store.get(jobId)) ?? job;
      }
    }
    if (job.status === "cancelling") this.active.get(jobId)?.controller.abort();
    return publicView(job);
  }

  private async requiredJob(jobId: string): Promise<RuntimeScheduleJob> {
    const job = await this.store.get(jobId);
    if (!job) throw new Error(`Runtime schedule job ${jobId} does not exist`);
    return job;
  }

  private async reconcileOrphan(job: RuntimeScheduleJob): Promise<RuntimeScheduleJob> {
    const locallyActive = this.active.has(job.jobId);
    const remotelyActive = job.ownerProcessId ? this.ownerIsAlive(job.ownerProcessId) : false;
    if (
      (job.status === "queued" || job.status === "running" || job.status === "cancelling") &&
      !locallyActive &&
      !remotelyActive
    ) {
      try {
        return await this.store.update(
          job.jobId,
          ["queued", "running", "cancelling"],
          (current) => ({
            ...current,
            status: "interrupted",
            finishedAt: this.now().toISOString(),
            diagnostic:
              "The owning sidecar process stopped before the schedule reached a terminal state",
          }),
        );
      } catch {
        return (await this.store.get(job.jobId)) ?? job;
      }
    }
    return job;
  }

  private async execute(
    jobId: string,
    controller: AbortController,
    execute: (signal: AbortSignal) => Promise<WorkItemScheduleResult>,
  ): Promise<void> {
    const cancellationPoll = setInterval(() => {
      void this.observeCancellation(jobId, controller).catch(() => undefined);
    }, 250);
    cancellationPoll.unref();
    try {
      await this.store.update(jobId, "queued", (current) => ({
        ...current,
        status: "running",
        startedAt: this.now().toISOString(),
      }));
      const result = await execute(controller.signal);
      const current = await this.store.get(jobId);
      if (controller.signal.aborted || current?.status === "cancelling") {
        await this.markCancelled(jobId);
        return;
      }
      await this.store.update(jobId, "running", (job) => ({
        ...job,
        status: "completed",
        finishedAt: this.now().toISOString(),
        result,
      }));
    } catch (error) {
      const current = await this.store.get(jobId);
      if (!current || isTerminal(current.status)) return;
      if (controller.signal.aborted || current.status === "cancelling") {
        await this.markCancelled(jobId);
        return;
      }
      await this.store.update(jobId, ["queued", "running"], (job) => ({
        ...job,
        status: "failed",
        finishedAt: this.now().toISOString(),
        diagnostic: error instanceof Error ? error.message : String(error),
      }));
    } finally {
      clearInterval(cancellationPoll);
    }
  }

  private async observeCancellation(jobId: string, controller: AbortController): Promise<void> {
    if (controller.signal.aborted) return;
    const job = await this.store.get(jobId);
    if (job?.status === "cancelling") controller.abort();
  }

  private async markCancelled(jobId: string): Promise<void> {
    const current = await this.store.get(jobId);
    if (!current || isTerminal(current.status)) return;
    await this.store.update(jobId, ["queued", "running", "cancelling"], (job) => ({
      ...job,
      status: "cancelled",
      cancellationRequestedAt: job.cancellationRequestedAt ?? this.now().toISOString(),
      finishedAt: this.now().toISOString(),
      diagnostic: "Runtime schedule cancellation completed after active workers stopped",
    }));
  }
}

function publicView(job: RuntimeScheduleJob): RuntimeScheduleJobView {
  return {
    jobId: job.jobId,
    taskId: job.taskId,
    codeRevision: job.codeRevision,
    contextDigest: job.contextDigest,
    status: job.status,
    requestedAt: job.requestedAt,
    cancellationRequestedAt: job.cancellationRequestedAt,
    startedAt: job.startedAt,
    finishedAt: job.finishedAt,
    result: job.result,
    diagnostic: job.diagnostic,
  };
}

function isTerminal(status: RuntimeScheduleStatus): boolean {
  return (
    status === "completed" ||
    status === "failed" ||
    status === "interrupted" ||
    status === "cancelled"
  );
}

function assertValidTransition(current: RuntimeScheduleStatus, next: RuntimeScheduleStatus): void {
  const allowed: Readonly<Record<RuntimeScheduleStatus, readonly RuntimeScheduleStatus[]>> = {
    queued: ["running", "cancelling", "failed", "interrupted"],
    running: ["cancelling", "completed", "failed", "interrupted"],
    cancelling: ["cancelled", "interrupted"],
    completed: [],
    failed: [],
    interrupted: [],
    cancelled: [],
  };
  if (!allowed[current].includes(next)) {
    throw new Error(`Invalid runtime schedule transition ${current} -> ${next}`);
  }
}

function isProcessAlive(processId: number): boolean {
  try {
    process.kill(processId, 0);
    return true;
  } catch (error) {
    return isNodeError(error) && error.code === "EPERM";
  }
}

function assertImmutableFields(current: RuntimeScheduleJob, next: RuntimeScheduleJob): void {
  const immutable = (job: RuntimeScheduleJob) => ({
    jobId: job.jobId,
    taskId: job.taskId,
    request: job.request,
    codeRevision: job.codeRevision,
    contextDigest: job.contextDigest,
    requestedAt: job.requestedAt,
  });
  if (JSON.stringify(immutable(current)) !== JSON.stringify(immutable(next))) {
    throw new Error(`Runtime schedule job ${current.jobId} immutable fields changed`);
  }
}

function assertValidJob(job: RuntimeScheduleJob): void {
  scheduleJobSchema.parse(job);
  if (job.taskId !== job.request.taskId) {
    throw new Error("Runtime schedule job task does not match its request");
  }
  if (job.startedAt && job.startedAt < job.requestedAt) {
    throw new Error("Runtime schedule job cannot start before it was requested");
  }
  if (job.finishedAt && job.finishedAt < job.requestedAt) {
    throw new Error("Runtime schedule job cannot finish before it was requested");
  }
  if (job.startedAt && job.finishedAt && job.finishedAt < job.startedAt) {
    throw new Error("Runtime schedule job cannot finish before it started");
  }
  if (
    job.status === "queued" &&
    (!job.ownerInstanceId ||
      !job.ownerProcessId ||
      job.startedAt ||
      job.finishedAt ||
      job.cancellationRequestedAt ||
      job.result ||
      job.diagnostic)
  ) {
    throw new Error("A queued runtime schedule job must identify its owner before execution");
  }
  if (
    job.status === "running" &&
    (!job.startedAt ||
      !job.ownerInstanceId ||
      !job.ownerProcessId ||
      job.finishedAt ||
      job.cancellationRequestedAt ||
      job.result ||
      job.diagnostic)
  ) {
    throw new Error("A running runtime schedule job must identify its owner and start time");
  }
  if (
    job.status === "cancelling" &&
    (!job.ownerInstanceId ||
      !job.ownerProcessId ||
      !job.cancellationRequestedAt ||
      job.finishedAt ||
      job.result ||
      job.diagnostic)
  ) {
    throw new Error("A cancelling runtime schedule job must retain its owner and request time");
  }
  if (isTerminal(job.status) && !job.finishedAt) {
    throw new Error("A terminal runtime schedule job must have a finish time");
  }
  if (job.status === "completed" && !job.result) {
    throw new Error("A completed runtime schedule job must contain its result");
  }
  if (job.status === "completed" && (!job.startedAt || !job.ownerProcessId)) {
    throw new Error("A completed runtime schedule job must retain its execution owner");
  }
  if (
    (job.status === "failed" || job.status === "interrupted" || job.status === "cancelled") &&
    !job.diagnostic
  ) {
    throw new Error(`${job.status} runtime schedule job must contain a diagnostic`);
  }
  if (job.status === "cancelled" && !job.cancellationRequestedAt) {
    throw new Error("A cancelled runtime schedule job must retain its cancellation request time");
  }
}

function encode(job: RuntimeScheduleJob): string {
  assertValidJob(job);
  const envelope: RuntimeScheduleJobEnvelope = {
    schemaVersion: 1,
    kind: "runtime_schedule_job",
    payloadDigest: digest(job),
    payload: job,
  };
  return `${JSON.stringify(envelope, null, 2)}\n`;
}

async function readJob(filePath: string): Promise<RuntimeScheduleJob> {
  const parsed: unknown = JSON.parse(await readFile(filePath, "utf8"));
  const envelope = z
    .object({
      schemaVersion: z.literal(1),
      kind: z.literal("runtime_schedule_job"),
      payloadDigest: nonEmptyString,
      payload: scheduleJobSchema,
    })
    .strict()
    .parse(parsed);
  if (digest(envelope.payload) !== envelope.payloadDigest) {
    throw new Error("Runtime schedule job payload digest mismatch");
  }
  const job = envelope.payload as RuntimeScheduleJob;
  assertValidJob(job);
  return job;
}

function digest(value: unknown): string {
  return `sha256:${createHash("sha256").update(canonicalJson(value)).digest("hex")}`;
}

function canonicalJson(value: unknown): string {
  if (value === undefined) return "null";
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, item]) => item !== undefined)
      .sort(([left], [right]) => left.localeCompare(right));
    return `{${entries
      .map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

async function atomicPrivateWrite(filePath: string, contents: string): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true, mode: 0o700 });
  const temporaryPath = `${filePath}.${randomUUID()}.tmp`;
  await writeFile(temporaryPath, contents, { encoding: "utf8", mode: 0o600, flag: "wx" });
  await rename(temporaryPath, filePath);
}

async function moveAsideFile(filePath: string): Promise<boolean> {
  const target = `${filePath}.${randomUUID()}.released`;
  try {
    await rename(filePath, target);
    await rm(target, { force: true });
    return true;
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return false;
    throw error;
  }
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function fileOlderThan(filePath: string, milliseconds: number): Promise<boolean> {
  try {
    return Date.now() - (await stat(filePath)).mtimeMs >= milliseconds;
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return false;
    throw error;
  }
}

function isNodeError(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && "code" in error;
}
