import { randomUUID } from "node:crypto";
import { mkdir, open, readFile, rename, rm } from "node:fs/promises";
import path from "node:path";
import { withLocalFileMutationLock } from "@kontext-brain/local";
import { z } from "zod";
import { taskPlanningDigest } from "./task-planning-contract.js";

const id = z.string().min(1);
const digest = z.string().regex(/^sha256:[a-f0-9]{64}$/);
const manifestSchema = z
  .object({
    manifestId: id,
    taskId: id,
    taskContractDigest: id,
    contextDigest: id,
    baseCodeRevision: id,
    resultCodeRevision: id,
    normativeRevisions: z.array(
      z
        .object({
          kind: z.enum(["decision", "domain_term", "invariant"]),
          recordId: id,
          revisionId: id,
        })
        .strict(),
    ),
    evidenceIds: z.array(id),
    workItemIds: z.array(id),
    changeBundleIds: z.array(id),
    changedSymbolIds: z.array(id),
    verificationRunIds: z.array(id),
    reviewFindingIds: z.array(id),
    emergencyBypassIds: z.array(id),
    createdAt: z.string().datetime(),
  })
  .strict();
export const taskFinalizationRequestSchema = z
  .object({
    taskId: id,
    jobId: id,
    requestId: z.string().uuid(),
    expectedCompletionBasisDigest: digest,
  })
  .strict();
export type TaskFinalizationRequest = z.infer<typeof taskFinalizationRequestSchema>;
export const taskFinalizationRecordSchema = z
  .object({
    schemaVersion: z.literal(1),
    recordId: digest,
    organizationId: id,
    subjectId: id,
    request: taskFinalizationRequestSchema,
    codeRevision: id,
    contextDigest: id,
    gitCommit: id,
    accuracyManifestId: id,
    accuracyManifest: manifestSchema,
    verificationRunIds: z.array(id),
    completedAt: z.string().datetime(),
  })
  .strict();
export type TaskFinalizationRecord = z.infer<typeof taskFinalizationRecordSchema>;
type Owner = { organizationId: string; subjectId: string };
const envelopeSchema = z
  .object({
    kind: z.literal("task_finalizations"),
    taskId: id,
    organizationId: id,
    subjectId: id,
    digest,
    records: z.array(taskFinalizationRecordSchema).max(1000),
  })
  .strict();

/** Immutable completion history; a stored record alone is not current code evidence. */
export class FileTaskFinalizationStore {
  constructor(private readonly directory: string) {}

  mutate<T>(owner: Owner, taskId: string, action: () => Promise<T>): Promise<T> {
    return withLocalFileMutationLock(`${this.filePath(owner, taskId)}.lock`, action);
  }

  async list(owner: Owner, taskId: string): Promise<TaskFinalizationRecord[]> {
    let serialized: string;
    try {
      serialized = await readFile(this.filePath(owner, taskId), "utf8");
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT") return [];
      throw error;
    }
    const envelope = envelopeSchema.parse(JSON.parse(serialized));
    if (
      envelope.taskId !== taskId ||
      envelope.organizationId !== owner.organizationId ||
      envelope.subjectId !== owner.subjectId ||
      taskPlanningDigest(envelope.records) !== envelope.digest
    )
      throw new Error("Task finalization history integrity check failed");
    const requests = new Set<string>();
    for (const record of envelope.records) {
      this.validateRecord(owner, taskId, record);
      if (requests.has(record.request.requestId))
        throw new Error("Duplicate finalization request in history");
      requests.add(record.request.requestId);
    }
    return envelope.records;
  }

  async append(owner: Owner, taskId: string, input: TaskFinalizationRecord): Promise<void> {
    const record = taskFinalizationRecordSchema.parse(input);
    this.validateRecord(owner, taskId, record);
    const records = await this.list(owner, taskId);
    const existing = records.find((entry) => entry.request.requestId === record.request.requestId);
    if (existing) {
      if (taskPlanningDigest(existing) !== taskPlanningDigest(record))
        throw new Error("Finalization request is immutable");
      return;
    }
    if (records.length >= 1000) throw new Error("Task finalization history is full");
    records.push(record);
    const filename = this.filePath(owner, taskId);
    const temporary = `${filename}.${randomUUID()}.tmp`;
    await mkdir(path.dirname(filename), { recursive: true, mode: 0o700 });
    try {
      const file = await open(temporary, "wx", 0o600);
      try {
        await file.writeFile(
          JSON.stringify({
            kind: "task_finalizations",
            taskId,
            organizationId: owner.organizationId,
            subjectId: owner.subjectId,
            digest: taskPlanningDigest(records),
            records,
          }),
        );
        await file.sync();
      } finally {
        await file.close();
      }
      await rename(temporary, filename);
    } finally {
      await rm(temporary, { force: true });
    }
  }

  private validateRecord(owner: Owner, taskId: string, record: TaskFinalizationRecord): void {
    const { recordId, ...payload } = record;
    if (
      record.request.taskId !== taskId ||
      record.organizationId !== owner.organizationId ||
      record.subjectId !== owner.subjectId ||
      recordId !== taskPlanningDigest(payload) ||
      record.accuracyManifestId !== record.accuracyManifest.manifestId ||
      record.accuracyManifest.taskId !== taskId ||
      record.accuracyManifest.contextDigest !== record.contextDigest ||
      record.accuracyManifest.resultCodeRevision !== record.codeRevision ||
      JSON.stringify(record.verificationRunIds) !==
        JSON.stringify(record.accuracyManifest.verificationRunIds)
    )
      throw new Error("Task finalization record owner or digest mismatch");
  }

  private filePath(owner: Owner, taskId: string): string {
    return path.join(
      this.directory,
      "task-finalizations",
      `${taskPlanningDigest([owner.organizationId, owner.subjectId, taskId]).slice(7)}.json`,
    );
  }
}
