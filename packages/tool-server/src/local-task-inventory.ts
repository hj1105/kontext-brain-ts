import { FileTaskContextRepository } from "@kontext-brain/local";
import { z } from "zod";
import { FileIntegratedTaskStateStore } from "./file-integrated-task-state-store.js";
import { FileRuntimeScheduleJobStore } from "./file-runtime-schedule-job-store.js";
import { FileTaskFinalizationStore } from "./file-task-finalization-store.js";
import { loadLocalKnowledgePrincipal } from "./local-knowledge-principal.js";
import { taskPlanningDigest } from "./task-planning-contract.js";

export const taskInventoryRequestSchema = z
  .object({
    limit: z.number().int().min(1).max(100).default(50),
    workspaceId: z.string().min(1).optional(),
    cursor: z
      .object({
        digest: z.string().regex(/^sha256:[a-f0-9]{64}$/),
        offset: z.number().int().positive(),
      })
      .strict()
      .optional(),
  })
  .strict();

/** Saved host records only: listing never refreshes a schedule, sources or completion evidence. */
export class LocalTaskInventoryOperations {
  constructor(private readonly directory: string) {}

  async list(input: unknown) {
    const request = taskInventoryRequestSchema.parse(input);
    const principal = await loadLocalKnowledgePrincipal(this.directory);
    const capture = async () => {
      const repository = new FileTaskContextRepository(this.directory);
      const tasks = (await repository.listInitialTaskMetadata(principal))
        .filter(
          (task) =>
            !request.workspaceId ||
            task.owner.contextSelection?.workspaceId === request.workspaceId,
        )
        .sort(
          (left, right) =>
            right.createdAt.localeCompare(left.createdAt) ||
            left.taskId.localeCompare(right.taskId),
        );
      const schedules = await new FileRuntimeScheduleJobStore(this.directory).listTaskSummaries(
        new Set(tasks.map((task) => task.taskId)),
      );
      const schedulesByTask = new Map<string, typeof schedules>();
      for (const schedule of schedules) {
        const jobs = schedulesByTask.get(schedule.taskId) ?? [];
        jobs.push(schedule);
        schedulesByTask.set(schedule.taskId, jobs);
      }
      const integrations = new FileIntegratedTaskStateStore(this.directory);
      const finalizations = new FileTaskFinalizationStore(this.directory);
      const rows = [];
      for (const task of tasks) {
        const selection = task.owner.contextSelection;
        if (!selection) throw new Error("Task inventory lacks registered context selection");
        const prepared = await repository.get(task.taskId);
        if (
          !prepared ||
          prepared.contract.taskId !== task.taskId ||
          prepared.snapshot.taskId !== task.taskId
        )
          throw new Error("Task inventory prepared identity changed");
        const jobs = (schedulesByTask.get(task.taskId) ?? []).sort(
          (left, right) =>
            right.requestedAt.localeCompare(left.requestedAt) ||
            left.jobId.localeCompare(right.jobId),
        );
        const integration = await integrations.get(task.taskId);
        const record = (await finalizations.list(principal, task.taskId)).at(-1);
        rows.push({
          taskId: task.taskId,
          intent: prepared.contract.intent,
          risk: prepared.contract.risk,
          workspaceId: selection.workspaceId,
          workspacePath: task.owner.workspacePath,
          createdAt: task.createdAt,
          contextDigest: prepared.snapshot.contextDigest,
          scheduleCount: jobs.length,
          unsettledScheduleCount: jobs.filter(
            (job) => !["completed", "failed", "cancelled"].includes(job.status),
          ).length,
          latestSchedule: jobs[0] ?? null,
          integration: integration
            ? {
                jobId: integration.scheduleJobId,
                gitCommit: integration.gitCommit,
                createdAt: integration.createdAt,
              }
            : null,
          finalization: record
            ? {
                recordId: record.recordId,
                jobId: record.request.jobId,
                gitCommit: record.gitCommit,
                completedAt: record.completedAt,
              }
            : null,
        });
      }
      return rows;
    };
    const rows = await capture();
    const inventoryDigest = taskPlanningDigest([principal, request.workspaceId ?? null, rows]);
    if (
      taskPlanningDigest([principal, request.workspaceId ?? null, await capture()]) !==
        inventoryDigest ||
      taskPlanningDigest(await loadLocalKnowledgePrincipal(this.directory)) !==
        taskPlanningDigest(principal)
    )
      throw new Error("Task inventory changed during observation; reload the first page");
    if (
      request.cursor &&
      (request.cursor.digest !== inventoryDigest || request.cursor.offset >= rows.length)
    )
      throw new Error("Task inventory changed; reload the first page");
    const offset = request.cursor?.offset ?? 0;
    const next = offset + request.limit;
    return {
      version: 1 as const,
      organizationId: principal.organizationId,
      observation: "saved_metadata_only" as const,
      currentEvidence: "not_revalidated" as const,
      tasks: rows.slice(offset, next),
      inventoryDigest,
      nextCursor: next < rows.length ? { digest: inventoryDigest, offset: next } : null,
    };
  }
}
