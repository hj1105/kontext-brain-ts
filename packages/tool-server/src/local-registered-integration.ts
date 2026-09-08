import { realpath } from "node:fs/promises";
import { z } from "zod";
import {
  FileIntegratedTaskStateStore,
  integratedTaskStateSchema,
} from "./file-integrated-task-state-store.js";
import {
  type LocalRegisteredScheduleOperations,
  registeredScheduleControlSchema,
  registeredScheduleSelectorSchema,
} from "./local-registered-schedules.js";
import { requireLocalTaskOwner } from "./local-task-owner.js";
import type { KontextRuntimeOperations } from "./runtime-workflow-tools.js";
import { taskPlanningDigest } from "./task-planning-contract.js";

export const registeredIntegrationRequestSchema = registeredScheduleControlSchema.extend({
  expectedIntegrationDigest: z
    .string()
    .regex(/^sha256:[a-f0-9]{64}$/)
    .nullable(),
  allowSubscriptionExecution: z.literal(true),
});

export class LocalRegisteredIntegrationOperations {
  constructor(
    private readonly directory: string,
    private readonly schedules: Pick<LocalRegisteredScheduleOperations, "inspect">,
    private readonly runtime: Pick<KontextRuntimeOperations, "integrateSchedule">,
    private readonly now: () => Date = () => new Date(),
  ) {}

  async inspect(input: unknown) {
    const request = registeredScheduleSelectorSchema.parse(input);
    const schedule = await this.schedules.inspect(request);
    const store = new FileIntegratedTaskStateStore(this.directory);
    const integration = (await store.get(request.taskId)) ?? null;
    if (
      integration?.scheduleJobId === request.jobId &&
      (integration.baseRevision !== schedule.job.codeRevision ||
        integration.contextDigest !== schedule.job.contextDigest ||
        taskPlanningDigest([...integration.workItemIds].sort()) !==
          taskPlanningDigest(schedule.workItems.map((work) => work.workItemId).sort()))
    ) {
      throw new Error("Saved integration does not match the selected execution");
    }
    const after = await this.schedules.inspect(request);
    if (
      taskPlanningDigest(schedule) !== taskPlanningDigest(after) ||
      taskPlanningDigest((await store.get(request.taskId)) ?? null) !==
        taskPlanningDigest(integration)
    ) {
      throw new Error("Saved integration changed during observation; inspect again");
    }
    const canIntegrate =
      schedule.job.status === "completed" &&
      !schedule.job.cancellationRequestedAt &&
      schedule.workItems.length > 0 &&
      schedule.workItems.every((work) => work.result?.status === "completed");
    return {
      version: 1 as const,
      taskId: request.taskId,
      jobId: request.jobId,
      jobIdentityDigest: schedule.jobIdentityDigest,
      observation: "saved_metadata_only" as const,
      currentEvidence: "not_revalidated" as const,
      scheduleStatus: schedule.job.status,
      canRequestIntegration: canIntegrate,
      integrationDigest: integration ? taskPlanningDigest(integration) : null,
      integration,
    };
  }

  async integrate(input: unknown) {
    const request = registeredIntegrationRequestSchema.parse(input);
    const selector = { taskId: request.taskId, jobId: request.jobId };
    const before = await this.inspect(selector);
    if (
      before.jobIdentityDigest !== request.expectedJobIdentityDigest ||
      before.integrationDigest !== request.expectedIntegrationDigest ||
      !before.canRequestIntegration
    ) {
      throw new Error("Reviewed execution or integration changed; inspect before integrating");
    }
    const schedule = await this.schedules.inspect(selector);
    const owner = await requireLocalTaskOwner(this.directory, request.taskId);
    if (
      schedule.jobIdentityDigest !== before.jobIdentityDigest ||
      (await realpath(schedule.job.repositoryPath)) !==
        (await realpath(owner.registration.owner.workspacePath))
    ) {
      throw new Error("Registered integration workspace identity mismatch");
    }
    const observed = this.now();
    const result = z.object({ state: integratedTaskStateSchema }).parse(
      await this.runtime.integrateSchedule({
        jobId: request.jobId,
        observedAt: observed.toISOString(),
        nextAttemptAt: new Date(observed.getTime() + 60_000).toISOString(),
        expectedIntegrationDigest: request.expectedIntegrationDigest,
      }),
    );
    const after = await this.inspect(selector);
    if (
      after.jobIdentityDigest !== before.jobIdentityDigest ||
      after.integration?.scheduleJobId !== request.jobId ||
      taskPlanningDigest(result.state) !== after.integrationDigest
    ) {
      throw new Error("Integration outcome changed or is unknown; inspect before retrying");
    }
    return { ...after, command: "integrate" as const };
  }
}
