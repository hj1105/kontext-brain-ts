import { z } from "zod";
import {
  FileTaskFinalizationStore,
  taskFinalizationRecordSchema,
  taskFinalizationRequestSchema,
} from "./file-task-finalization-store.js";
import type { LocalTaskCompletionAssessment } from "./local-task-completion-assessment.js";
import { requireLocalTaskOwner } from "./local-task-owner.js";
import { taskPlanningDigest } from "./task-planning-contract.js";

export const inspectFinalizationRequestSchema = z
  .object({
    taskId: z.string().min(1),
    requestId: z.string().uuid().optional(),
  })
  .strict();

export const revalidateFinalizationRequestSchema = z
  .object({
    taskId: z.string().min(1),
    expectedRecordId: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  })
  .strict();

export class LocalTaskFinalizationOperations {
  private readonly store: FileTaskFinalizationStore;
  constructor(
    private readonly directory: string,
    private readonly assessment: Pick<LocalTaskCompletionAssessment, "assess">,
  ) {
    this.store = new FileTaskFinalizationStore(directory);
  }

  async finalize(input: unknown) {
    const request = taskFinalizationRequestSchema.parse(input);
    const { principal } = await requireLocalTaskOwner(this.directory, request.taskId);
    return this.store.mutate(principal, request.taskId, async () => {
      const previous = (await this.store.list(principal, request.taskId)).find(
        (record) => record.request.requestId === request.requestId,
      );
      if (previous) {
        if (taskPlanningDigest(previous.request) !== taskPlanningDigest(request))
          throw new Error("Finalization request identity conflicts with its original input");
        return { created: false, record: previous, currentEvidence: "not_revalidated" as const };
      }
      const current = await this.assessment.assess({
        taskId: request.taskId,
        jobId: request.jobId,
      });
      if (current.completionBasisDigest !== request.expectedCompletionBasisDigest)
        throw new Error("Completion evidence changed after user review");
      if (
        current.state !== "done" ||
        current.issues.length ||
        !current.accuracyManifest ||
        current.context.status !== "current"
      )
        throw new Error("Task completion requirements are not satisfied");
      const payload = {
        schemaVersion: 1 as const,
        organizationId: principal.organizationId,
        subjectId: principal.subjectId,
        request,
        codeRevision: current.codeRevision,
        contextDigest: current.context.contextDigest,
        gitCommit: current.gitCommit,
        accuracyManifestId: current.accuracyManifest.manifestId,
        accuracyManifest: current.accuracyManifest,
        verificationRunIds: [...current.accuracyManifest.verificationRunIds],
        completedAt: current.observedAt,
      };
      const record = taskFinalizationRecordSchema.parse({
        ...payload,
        recordId: taskPlanningDigest(payload),
      });
      await this.store.append(principal, request.taskId, record);
      return { created: true, record, currentEvidence: "validated_at_recording" as const };
    });
  }

  async inspect(input: unknown) {
    const request = inspectFinalizationRequestSchema.parse(input);
    const { principal } = await requireLocalTaskOwner(this.directory, request.taskId);
    const records = await this.store.list(principal, request.taskId);
    const record = request.requestId
      ? (records.find((item) => item.request.requestId === request.requestId) ?? null)
      : (records.at(-1) ?? null);
    return { taskId: request.taskId, record, currentEvidence: "not_revalidated" as const };
  }

  async revalidate(input: unknown) {
    const request = revalidateFinalizationRequestSchema.parse(input);
    const { record } = await this.inspect({ taskId: request.taskId });
    if (!record || record.recordId !== request.expectedRecordId)
      throw new Error("Finalization history changed; read the latest record before revalidation");
    const current = await this.assessment.assess({
      taskId: request.taskId,
      jobId: record.request.jobId,
    });
    if (current.taskId !== request.taskId || current.jobId !== record.request.jobId)
      throw new Error("Completion assessment belongs to a different Task or schedule");
    const latest = await this.inspect({ taskId: request.taskId });
    if (latest.record?.recordId !== record.recordId)
      throw new Error("Finalization history changed during revalidation");
    const valid =
      current.completionBasisDigest === record.request.expectedCompletionBasisDigest &&
      current.state === "done" &&
      current.issues.length === 0 &&
      current.accuracyManifest?.taskId === request.taskId &&
      current.context.status === "current" &&
      current.accuracyManifest.resultCodeRevision === current.codeRevision &&
      current.accuracyManifest.contextDigest === current.context.contextDigest;
    return {
      taskId: request.taskId,
      recordId: record.recordId,
      currentEvidence: valid ? ("revalidated_current" as const) : ("changed" as const),
      recordedCompletionBasisDigest: record.request.expectedCompletionBasisDigest,
      observedCompletionBasisDigest: current.completionBasisDigest,
      observedAt: current.observedAt,
      state: current.state,
      issueCount: current.issues.length,
      codeRevision: current.codeRevision,
      contextDigest: current.context.contextDigest,
      contextStatus: current.context.status,
    };
  }
}
