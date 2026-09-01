import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import type { RuntimeProvider } from "@kontext-brain/orchestrator";
import { z } from "zod";

const nonEmptyString = z.string().min(1);
const integratedTaskStateSchema = z
  .object({
    taskId: nonEmptyString,
    scheduleJobId: nonEmptyString,
    repositoryPath: nonEmptyString,
    workspacePath: nonEmptyString,
    baseRevision: nonEmptyString,
    gitCommit: nonEmptyString,
    resultRevision: nonEmptyString,
    contextDigest: nonEmptyString,
    changeBundleIds: z.array(nonEmptyString).min(1),
    workItemIds: z.array(nonEmptyString).min(1),
    changedPaths: z.array(nonEmptyString).min(1),
    changedSymbolIds: z.array(nonEmptyString).min(1),
    authorProviders: z.array(z.enum(["codex", "claude"])).min(1),
    createdAt: z.string().datetime(),
  })
  .strict();

export interface IntegratedTaskState {
  readonly taskId: string;
  readonly scheduleJobId: string;
  readonly repositoryPath: string;
  readonly workspacePath: string;
  readonly baseRevision: string;
  readonly gitCommit: string;
  readonly resultRevision: string;
  readonly contextDigest: string;
  readonly changeBundleIds: readonly string[];
  readonly workItemIds: readonly string[];
  readonly changedPaths: readonly string[];
  readonly changedSymbolIds: readonly string[];
  readonly authorProviders: readonly RuntimeProvider[];
  readonly createdAt: string;
}

export interface IntegratedTaskStateStore {
  get(taskId: string): Promise<IntegratedTaskState | undefined>;
  put(state: IntegratedTaskState): Promise<IntegratedTaskState>;
}

interface IntegratedTaskStateEnvelope {
  readonly schemaVersion: 1;
  readonly kind: "integrated_task_state";
  readonly taskId: string;
  readonly payloadDigest: string;
  readonly payload: IntegratedTaskState;
}

export class FileIntegratedTaskStateStore implements IntegratedTaskStateStore {
  constructor(private readonly pluginDataDirectory: string) {}

  async get(taskId: string): Promise<IntegratedTaskState | undefined> {
    try {
      const envelope = z
        .object({
          schemaVersion: z.literal(1),
          kind: z.literal("integrated_task_state"),
          taskId: nonEmptyString,
          payloadDigest: nonEmptyString,
          payload: integratedTaskStateSchema,
        })
        .strict()
        .parse(
          JSON.parse(await readFile(this.filePath(taskId), "utf8")),
        ) as IntegratedTaskStateEnvelope;
      if (envelope.taskId !== taskId || envelope.payload.taskId !== taskId) {
        throw new Error("Integrated Task state does not match its storage location");
      }
      if (digest(envelope.payload) !== envelope.payloadDigest) {
        throw new Error("Integrated Task state payload digest mismatch");
      }
      return envelope.payload;
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") return undefined;
      throw error;
    }
  }

  async put(state: IntegratedTaskState): Promise<IntegratedTaskState> {
    const parsed = integratedTaskStateSchema.parse(state) as IntegratedTaskState;
    const current = await this.get(parsed.taskId);
    if (current && current.createdAt > parsed.createdAt) {
      throw new Error("Integrated Task state cannot move backwards in time");
    }
    if (current && current.scheduleJobId === parsed.scheduleJobId) {
      if (JSON.stringify(current) !== JSON.stringify(parsed)) {
        throw new Error(`Integration for schedule ${parsed.scheduleJobId} is immutable`);
      }
      return current;
    }
    const envelope: IntegratedTaskStateEnvelope = {
      schemaVersion: 1,
      kind: "integrated_task_state",
      taskId: parsed.taskId,
      payloadDigest: digest(parsed),
      payload: parsed,
    };
    await atomicPrivateWrite(
      this.filePath(parsed.taskId),
      `${JSON.stringify(envelope, null, 2)}\n`,
    );
    return parsed;
  }

  filePath(taskId: string): string {
    return path.join(
      this.pluginDataDirectory,
      "integrated-tasks",
      `${createHash("sha256").update(taskId).digest("hex")}.json`,
    );
  }
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

function isNodeError(value: unknown): value is NodeJS.ErrnoException {
  return value instanceof Error && "code" in value;
}
