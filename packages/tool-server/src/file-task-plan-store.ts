import { randomUUID } from "node:crypto";
import { mkdir, open, readFile, rename, rm } from "node:fs/promises";
import path from "node:path";
import { withLocalFileMutationLock } from "@kontext-brain/local";
import {
  type TaskPlanRecord,
  taskPlanRecordSchema,
  taskPlanningDigest,
} from "./task-planning-contract.js";

export class FileTaskPlanStore {
  constructor(private readonly directory: string) {}

  mutate<T>(key: string, operation: () => Promise<T>): Promise<T> {
    return withLocalFileMutationLock(`${this.filePath(key)}.lock`, operation);
  }

  async get(key: string): Promise<TaskPlanRecord | undefined> {
    let encoded: string;
    try {
      encoded = await readFile(this.filePath(key), "utf8");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
      throw error;
    }
    const envelope = JSON.parse(encoded);
    if (
      envelope.kind !== "task_plan" ||
      envelope.key !== key ||
      envelope.digest !== taskPlanningDigest(envelope.payload)
    )
      throw new Error("Task plan integrity check failed");
    const record = taskPlanRecordSchema.parse(envelope.payload);
    if (key !== this.key(record, record.request.requestId))
      throw new Error("Task plan owner mismatch");
    return record;
  }

  async put(key: string, input: TaskPlanRecord): Promise<void> {
    const payload = taskPlanRecordSchema.parse(input);
    if (key !== this.key(payload, payload.request.requestId))
      throw new Error("Task plan owner mismatch");
    const filePath = this.filePath(key);
    await mkdir(path.dirname(filePath), { recursive: true, mode: 0o700 });
    const temporary = `${filePath}.${randomUUID()}.tmp`;
    try {
      const handle = await open(temporary, "wx", 0o600);
      try {
        await handle.writeFile(
          JSON.stringify({ kind: "task_plan", key, digest: taskPlanningDigest(payload), payload }),
        );
        await handle.sync();
      } finally {
        await handle.close();
      }
      await rename(temporary, filePath);
    } finally {
      await rm(temporary, { force: true });
    }
  }

  key(owner: { organizationId: string; subjectId: string }, requestId: string): string {
    return taskPlanningDigest([owner.organizationId, owner.subjectId, requestId]);
  }

  private filePath(key: string): string {
    return path.join(this.directory, "task-plans", `${taskPlanningDigest(key).slice(7)}.json`);
  }
}
