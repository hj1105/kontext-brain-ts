import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { z } from "zod";

const nonEmptyString = z.string().min(1);
const eventSchema = z
  .object({
    toolUseId: nonEmptyString,
    workspacePath: nonEmptyString,
    taskId: nonEmptyString,
    workItemId: nonEmptyString,
    receiptId: nonEmptyString,
    contextDigest: nonEmptyString,
    baselineRevision: nonEmptyString,
    authorizedPaths: z.array(nonEmptyString).min(1),
    authorizedAt: z.string().datetime(),
    consumedAt: z.string().datetime().optional(),
  })
  .strict();

export interface WriteAuthorizationEvent {
  readonly toolUseId: string;
  readonly workspacePath: string;
  readonly taskId: string;
  readonly workItemId: string;
  readonly receiptId: string;
  readonly contextDigest: string;
  readonly baselineRevision: string;
  readonly authorizedPaths: readonly string[];
  readonly authorizedAt: string;
  readonly consumedAt?: string;
}

interface EventEnvelope {
  readonly schemaVersion: 1;
  readonly kind: "write_authorization_event";
  readonly payloadDigest: string;
  readonly event: WriteAuthorizationEvent;
}

export class FileWriteAuthorizationEventStore {
  constructor(private readonly pluginDataDirectory: string) {}

  async put(event: WriteAuthorizationEvent): Promise<void> {
    const payload = eventSchema.parse(event) as WriteAuthorizationEvent;
    const existing = await this.read(this.filePath(event.toolUseId, "pending")).catch((error) => {
      if (isNodeError(error) && error.code === "ENOENT") return undefined;
      throw error;
    });
    if (existing) {
      if (JSON.stringify(existing) !== JSON.stringify(payload)) {
        throw new Error(`Write authorization ${event.toolUseId} is immutable`);
      }
      return;
    }
    await atomicPrivateWrite(this.filePath(event.toolUseId, "pending"), encode(payload));
  }

  async consume(
    toolUseId: string,
    consumedAt: string,
  ): Promise<WriteAuthorizationEvent | undefined> {
    const source = this.filePath(toolUseId, "pending");
    let event: WriteAuthorizationEvent;
    try {
      event = await this.read(source);
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") return undefined;
      throw error;
    }
    const consumed = eventSchema.parse({ ...event, consumedAt }) as WriteAuthorizationEvent;
    const target = this.filePath(toolUseId, "consumed");
    await atomicPrivateWrite(source, encode(consumed));
    await mkdir(path.dirname(target), { recursive: true, mode: 0o700 });
    await rename(source, target);
    return consumed;
  }

  async consumeForWorkspace(
    workspacePath: string,
    baselineRevision: string,
    consumedAt: string,
  ): Promise<WriteAuthorizationEvent | undefined> {
    const directory = path.join(this.pluginDataDirectory, "write-authorizations", "pending");
    let entries: string[];
    try {
      entries = await readdir(directory);
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") return undefined;
      throw error;
    }
    const candidates: WriteAuthorizationEvent[] = [];
    for (const entry of entries.filter((value) => value.endsWith(".json")).sort()) {
      const event = await this.read(path.join(directory, entry));
      if (
        event.workspacePath === workspacePath &&
        event.baselineRevision === baselineRevision &&
        event.consumedAt === undefined
      ) {
        candidates.push(event);
      }
    }
    candidates.sort(
      (left, right) =>
        left.authorizedAt.localeCompare(right.authorizedAt) ||
        left.toolUseId.localeCompare(right.toolUseId),
    );
    const selected = candidates[0];
    return selected ? this.consume(selected.toolUseId, consumedAt) : undefined;
  }

  filePath(toolUseId: string, status: "pending" | "consumed"): string {
    return path.join(
      this.pluginDataDirectory,
      "write-authorizations",
      status,
      `${createHash("sha256").update(toolUseId).digest("hex")}.json`,
    );
  }

  private async read(filePath: string): Promise<WriteAuthorizationEvent> {
    const parsed: unknown = JSON.parse(await readFile(filePath, "utf8"));
    const envelope = z
      .object({
        schemaVersion: z.literal(1),
        kind: z.literal("write_authorization_event"),
        payloadDigest: nonEmptyString,
        event: eventSchema,
      })
      .strict()
      .parse(parsed) as EventEnvelope;
    if (digest(envelope.event) !== envelope.payloadDigest) {
      throw new Error("Write authorization event digest mismatch");
    }
    return envelope.event;
  }
}

function encode(event: WriteAuthorizationEvent): string {
  const envelope: EventEnvelope = {
    schemaVersion: 1,
    kind: "write_authorization_event",
    payloadDigest: digest(event),
    event,
  };
  return `${JSON.stringify(envelope, null, 2)}\n`;
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
