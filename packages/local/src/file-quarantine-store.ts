import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { type QuarantineStore, isQuarantineRecordValid } from "@kontext-brain/orchestrator";
import type { QuarantineRecord } from "@kontext-brain/spec";
import { z } from "zod";

const nonEmptyString = z.string().min(1);
const recordSchema = z
  .object({
    quarantineId: nonEmptyString,
    taskId: nonEmptyString.optional(),
    workItemId: nonEmptyString.optional(),
    codeRevision: nonEmptyString,
    contextDigest: nonEmptyString.optional(),
    paths: z.array(z.string()),
    symbolIds: z.array(nonEmptyString),
    reasons: z.array(
      z.enum([
        "missing_capability",
        "expired_capability",
        "context_mismatch",
        "path_out_of_scope",
        "symbol_out_of_scope",
        "unobserved_write",
      ]),
    ),
    status: z.enum(["active", "released"]),
    observedAt: nonEmptyString,
    releasedAt: nonEmptyString.optional(),
    releasedBy: nonEmptyString.optional(),
  })
  .strict();

interface QuarantineEnvelope {
  readonly schemaVersion: 1;
  readonly kind: "quarantine";
  readonly payloadDigest: string;
  readonly payload: QuarantineRecord;
}

export class FileQuarantineStore implements QuarantineStore {
  constructor(private readonly pluginDataDirectory: string) {}

  async put(record: QuarantineRecord): Promise<QuarantineRecord> {
    assertValidRecord(record);
    const existing = await this.get(record.quarantineId);
    if (existing) {
      if (JSON.stringify(existing) !== JSON.stringify(record)) {
        throw new Error(`Quarantine Record ${record.quarantineId} is immutable`);
      }
      return existing;
    }
    await atomicPrivateWrite(this.filePath(record.quarantineId), encode(record));
    return record;
  }

  async get(quarantineId: string): Promise<QuarantineRecord | undefined> {
    let record: QuarantineRecord;
    try {
      record = await readRecord(this.filePath(quarantineId));
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") return undefined;
      throw error;
    }
    if (record.quarantineId !== quarantineId) {
      throw new Error("Quarantine Record does not match its storage location");
    }
    return record;
  }

  async list(status?: QuarantineRecord["status"]): Promise<readonly QuarantineRecord[]> {
    let entries: string[];
    try {
      entries = await readdir(this.directory());
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") return [];
      throw error;
    }
    const records = await Promise.all(
      entries
        .filter((entry) => entry.endsWith(".json"))
        .sort()
        .map((entry) => readRecord(path.join(this.directory(), entry))),
    );
    return records
      .filter((record) => status === undefined || record.status === status)
      .sort(
        (left, right) =>
          left.observedAt.localeCompare(right.observedAt) ||
          left.quarantineId.localeCompare(right.quarantineId),
      );
  }

  async release(
    quarantineId: string,
    releasedBy: string,
    releasedAt: string,
  ): Promise<QuarantineRecord> {
    if (!releasedBy.trim() || !releasedAt.trim()) {
      throw new Error("Releasing quarantine requires an actor and timestamp");
    }
    const current = await this.get(quarantineId);
    if (!current) throw new Error(`Quarantine Record ${quarantineId} does not exist`);
    if (current.status === "released") return current;
    const released: QuarantineRecord = {
      ...current,
      status: "released",
      releasedBy,
      releasedAt,
    };
    assertValidRecord(released);
    await atomicPrivateWrite(this.filePath(quarantineId), encode(released));
    return released;
  }

  filePath(quarantineId: string): string {
    return path.join(
      this.directory(),
      `${createHash("sha256").update(quarantineId).digest("hex")}.json`,
    );
  }

  private directory(): string {
    return path.join(this.pluginDataDirectory, "quarantine");
  }
}

function encode(record: QuarantineRecord): string {
  const payload = recordSchema.parse(record) as QuarantineRecord;
  const envelope: QuarantineEnvelope = {
    schemaVersion: 1,
    kind: "quarantine",
    payloadDigest: digest(payload),
    payload,
  };
  return `${JSON.stringify(envelope, null, 2)}\n`;
}

async function readRecord(filePath: string): Promise<QuarantineRecord> {
  const parsed: unknown = JSON.parse(await readFile(filePath, "utf8"));
  const envelope = z
    .object({
      schemaVersion: z.literal(1),
      kind: z.literal("quarantine"),
      payloadDigest: nonEmptyString,
      payload: recordSchema,
    })
    .strict()
    .parse(parsed) as QuarantineEnvelope;
  if (digest(envelope.payload) !== envelope.payloadDigest) {
    throw new Error("Quarantine Record payload digest mismatch");
  }
  assertValidRecord(envelope.payload);
  return envelope.payload;
}

function assertValidRecord(record: QuarantineRecord): void {
  recordSchema.parse(record);
  if (!isQuarantineRecordValid(record)) {
    throw new Error("Quarantine Record immutable ID mismatch");
  }
  if (record.status === "released" && (!record.releasedAt || !record.releasedBy)) {
    throw new Error("Released Quarantine Record requires release audit fields");
  }
  if (record.status === "active" && (record.releasedAt || record.releasedBy)) {
    throw new Error("Active Quarantine Record cannot have release audit fields");
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
