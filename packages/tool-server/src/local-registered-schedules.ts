import { z } from "zod";
import {
  FileRuntimeScheduleJobStore,
  type RuntimeScheduleJob,
} from "./file-runtime-schedule-job-store.js";
import { taskInventoryRequestSchema } from "./local-task-inventory.js";
import { requireLocalTaskOwner } from "./local-task-owner.js";
import type { KontextRuntimeOperations } from "./runtime-workflow-tools.js";
import { taskPlanningDigest } from "./task-planning-contract.js";

export const registeredScheduleSelectorSchema = z
  .object({ taskId: z.string().min(1), jobId: z.string().min(1) })
  .strict();
export const registeredScheduleControlSchema = registeredScheduleSelectorSchema.extend({
  expectedJobIdentityDigest: z.string().regex(/^sha256:[a-f0-9]{64}$/),
});
export const registeredScheduleResumeSchema = registeredScheduleControlSchema.extend({
  allowSubscriptionExecution: z.literal(true),
});
export const registeredScheduleListSchema = taskInventoryRequestSchema
  .pick({ limit: true, cursor: true })
  .extend({ taskId: z.string().min(1) });

export class LocalRegisteredScheduleOperations {
  constructor(
    private readonly directory: string,
    private readonly runtime: Pick<KontextRuntimeOperations, "getSchedule" | "cancelSchedule">,
  ) {}

  async list(input: unknown) {
    const request = registeredScheduleListSchema.parse(input);
    const owner = await requireLocalTaskOwner(this.directory, request.taskId);
    const ownership = (value: typeof owner) => [value.principal, value.registration.owner];
    const store = new FileRuntimeScheduleJobStore(this.directory);
    const capture = async () =>
      (await store.listTaskSummaries(new Set([request.taskId]))).sort(
        (left, right) =>
          right.requestedAt.localeCompare(left.requestedAt) ||
          left.jobId.localeCompare(right.jobId),
      );
    const schedules = await capture();
    const digest = (rows: typeof schedules) =>
      taskPlanningDigest([ownership(owner), request.taskId, rows]);
    const inventoryDigest = digest(schedules);
    if (
      digest(await capture()) !== inventoryDigest ||
      taskPlanningDigest(ownership(await requireLocalTaskOwner(this.directory, request.taskId))) !==
        taskPlanningDigest(ownership(owner)) ||
      (request.cursor &&
        (request.cursor.digest !== inventoryDigest || request.cursor.offset >= schedules.length))
    )
      throw new Error("Schedule history changed; reload the first page");
    const offset = request.cursor?.offset ?? 0;
    const next = offset + request.limit;
    return {
      version: 1 as const,
      taskId: request.taskId,
      organizationId: owner.principal.organizationId,
      observation: "saved_metadata_only" as const,
      currentEvidence: "not_revalidated" as const,
      inventoryDigest,
      schedules: schedules.slice(offset, next),
      nextCursor: next < schedules.length ? { digest: inventoryDigest, offset: next } : null,
    };
  }

  async inspect(input: unknown) {
    const request = registeredScheduleSelectorSchema.parse(input);
    const owner = await requireLocalTaskOwner(this.directory, request.taskId);
    const store = new FileRuntimeScheduleJobStore(this.directory);
    const job = await store.get(request.jobId);
    if (!job || job.taskId !== request.taskId)
      throw new Error("Registered schedule identity mismatch");
    const ownership = (value: typeof owner) => [value.principal, value.registration.owner];
    const identity = taskPlanningDigest([
      ownership(owner),
      {
        jobId: job.jobId,
        taskId: job.taskId,
        request: job.request,
        codeRevision: job.codeRevision,
        contextDigest: job.contextDigest,
        requestedAt: job.requestedAt,
      },
    ]);
    const result = projectSavedSchedule(job);
    if (
      taskPlanningDigest(await store.get(request.jobId)) !== taskPlanningDigest(job) ||
      taskPlanningDigest(ownership(await requireLocalTaskOwner(this.directory, request.taskId))) !==
        taskPlanningDigest(ownership(owner))
    ) {
      throw new Error("Registered schedule changed during observation; inspect again");
    }
    return {
      version: 1 as const,
      observation: "saved_metadata_only" as const,
      currentEvidence: "not_revalidated" as const,
      jobIdentityDigest: identity,
      ...result,
    };
  }

  async resume(input: unknown) {
    const request = registeredScheduleResumeSchema.parse(input);
    return this.control(request, "resume");
  }
  async cancel(input: unknown) {
    return this.control(registeredScheduleControlSchema.parse(input), "cancel");
  }
  private async control(
    request: z.infer<typeof registeredScheduleControlSchema>,
    action: "resume" | "cancel",
  ) {
    const before = await this.inspect({ taskId: request.taskId, jobId: request.jobId });
    if (before.jobIdentityDigest !== request.expectedJobIdentityDigest)
      throw new Error("Reviewed schedule identity changed; inspect again");
    const response = z
      .object({ taskId: z.string(), jobId: z.string(), resumeBlocked: z.boolean().optional() })
      .parse(
        await (action === "resume"
          ? this.runtime.getSchedule({ jobId: request.jobId })
          : this.runtime.cancelSchedule({ jobId: request.jobId })),
      );
    if (response.taskId !== request.taskId || response.jobId !== request.jobId)
      throw new Error("Runtime returned a different schedule; outcome unknown");
    const after = await this.inspect({ taskId: request.taskId, jobId: request.jobId });
    if (after.jobIdentityDigest !== before.jobIdentityDigest)
      throw new Error("Schedule identity changed after request; outcome unknown");
    return {
      ...after,
      command: {
        action,
        ...(action === "resume" ? { resumeBlocked: response.resumeBlocked === true } : {}),
      },
    };
  }
}

function projectSavedSchedule(job: RuntimeScheduleJob) {
  const results = z
    .object({
      results: z.array(
        z.object({
          workItemId: z.string().min(1),
          status: z.enum(["completed", "failed"]),
          attempts: z.number().int().nonnegative(),
          provider: z.enum(["codex", "claude"]).optional(),
          changeBundleId: z.string().min(1).optional(),
          settlementProofId: z.string().min(1).optional(),
        }),
      ),
    })
    .parse(job.result ?? job.progress ?? { results: [] }).results;
  const workIds = new Set(job.request.work.map((work) => work.workItemId));
  if (
    new Set(results.map((result) => result.workItemId)).size !== results.length ||
    results.some((result) => !workIds.has(result.workItemId))
  )
    throw new Error("Saved schedule progress identity mismatch");
  return {
    job: {
      jobId: job.jobId,
      taskId: job.taskId,
      codeRevision: job.codeRevision,
      contextDigest: job.contextDigest,
      repositoryPath: job.request.repositoryPath,
      status: job.status,
      requestedAt: job.requestedAt,
      startedAt: job.startedAt,
      finishedAt: job.finishedAt,
      cancellationRequestedAt: job.cancellationRequestedAt,
      resumeCount: job.resumeCount ?? 0,
      lastResumedAt: job.lastResumedAt,
    },
    workItems: job.request.work.map((work) => {
      const result = results.find((result) => result.workItemId === work.workItemId);
      return {
        workItemId: work.workItemId,
        eligibleProviders: work.eligibleProviders,
        pinnedProvider: work.pinnedProvider,
        result: result ?? null,
      };
    }),
    diagnosticPresent: Boolean(job.diagnostic),
  };
}
