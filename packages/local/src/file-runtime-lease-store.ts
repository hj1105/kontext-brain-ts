import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import type { RuntimeLease, RuntimeLeaseStore } from "@kontext-brain/orchestrator";
import { z } from "zod";

const nonEmptyString = z.string().min(1);
const leaseSchema = z
  .object({
    leaseId: nonEmptyString,
    taskId: nonEmptyString,
    workItemId: nonEmptyString,
    provider: z.enum(["codex", "claude"]),
    workspacePath: nonEmptyString,
    symbolIds: z.array(nonEmptyString),
    paths: z.array(nonEmptyString),
    acquiredAt: z.string().datetime(),
    expiresAt: z.string().datetime(),
    releasedAt: z.string().datetime().optional(),
  })
  .strict();
const lockOwnerSchema = z
  .object({
    leaseId: nonEmptyString,
    expiresAt: z.string().datetime(),
  })
  .strict();

interface RuntimeLeaseEnvelope {
  readonly schemaVersion: 1;
  readonly kind: "runtime_lease";
  readonly payloadDigest: string;
  readonly payload: RuntimeLease;
}

export class FileRuntimeLeaseStore implements RuntimeLeaseStore {
  private operation: Promise<void> = Promise.resolve();

  constructor(private readonly pluginDataDirectory: string) {}

  async acquire(lease: RuntimeLease): Promise<boolean> {
    return this.exclusive(async () => {
      assertValidLease(lease);
      const existing = await this.get(lease.leaseId);
      if (existing) {
        if (JSON.stringify(existing) !== JSON.stringify(lease)) {
          throw new Error(`Runtime lease ${lease.leaseId} is immutable`);
        }
        return existing.releasedAt === undefined && existing.expiresAt > lease.acquiredAt;
      }
      const active = await this.listActiveUnsafe(lease.acquiredAt);
      if (active.some((candidate) => scopesConflict(candidate, lease))) return false;
      const claimedScopes: string[] = [];
      try {
        for (const scope of leaseScopes(lease)) {
          if (!(await this.claimScope(scope, lease))) return false;
          claimedScopes.push(scope);
        }
        await atomicPrivateWrite(this.filePath(lease.leaseId), encode(lease));
        return true;
      } finally {
        if (claimedScopes.length !== leaseScopes(lease).length) {
          await this.releaseScopeClaims(claimedScopes, lease.leaseId);
        }
      }
    });
  }

  async release(leaseId: string, releasedAt: string): Promise<void> {
    await this.exclusive(async () => {
      const lease = await this.get(leaseId);
      if (!lease || lease.releasedAt) return;
      const released = { ...lease, releasedAt };
      assertValidLease(released);
      await atomicPrivateWrite(this.filePath(leaseId), encode(released));
      await this.releaseScopeClaims(leaseScopes(lease), leaseId);
    });
  }

  async listActive(now: string): Promise<readonly RuntimeLease[]> {
    return this.exclusive(() => this.listActiveUnsafe(now));
  }

  private async listActiveUnsafe(now: string): Promise<readonly RuntimeLease[]> {
    return (await this.listAll())
      .filter((lease) => !lease.releasedAt && lease.expiresAt > now)
      .sort(compareLeases);
  }

  private async listAll(): Promise<readonly RuntimeLease[]> {
    let entries: string[];
    try {
      entries = await readdir(this.directory());
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") return [];
      throw error;
    }
    return Promise.all(
      entries
        .filter((entry) => entry.endsWith(".json"))
        .sort()
        .map((entry) => readLease(path.join(this.directory(), entry))),
    );
  }

  private async get(leaseId: string): Promise<RuntimeLease | undefined> {
    try {
      const lease = await readLease(this.filePath(leaseId));
      if (lease.leaseId !== leaseId) {
        throw new Error("Runtime lease does not match its storage location");
      }
      return lease;
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") return undefined;
      throw error;
    }
  }

  private filePath(leaseId: string): string {
    return path.join(
      this.directory(),
      `${createHash("sha256").update(leaseId).digest("hex")}.json`,
    );
  }

  private directory(): string {
    return path.join(this.pluginDataDirectory, "runtime-leases");
  }

  private lockDirectory(scope: string): string {
    return path.join(
      this.pluginDataDirectory,
      "runtime-lease-locks",
      `${createHash("sha256").update(scope).digest("hex")}.lock`,
    );
  }

  private async claimScope(scope: string, lease: RuntimeLease): Promise<boolean> {
    const lockDirectory = this.lockDirectory(scope);
    await mkdir(path.dirname(lockDirectory), { recursive: true, mode: 0o700 });
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        await mkdir(lockDirectory, { mode: 0o700 });
        try {
          await atomicPrivateWrite(
            path.join(lockDirectory, "owner.json"),
            `${JSON.stringify({ leaseId: lease.leaseId, expiresAt: lease.expiresAt })}\n`,
          );
          return true;
        } catch (error) {
          await moveAside(lockDirectory);
          throw error;
        }
      } catch (error) {
        if (!isNodeError(error) || error.code !== "EEXIST") throw error;
      }
      const owner = await this.readLockOwner(lockDirectory);
      if (!owner) {
        await delay(20);
        continue;
      }
      const owningLease = await this.get(owner.leaseId);
      const reclaimable =
        owner.expiresAt <= lease.acquiredAt || owningLease?.releasedAt !== undefined;
      if (!reclaimable) return false;
      if (await moveAside(lockDirectory)) continue;
    }
    return false;
  }

  private async readLockOwner(
    lockDirectory: string,
  ): Promise<z.infer<typeof lockOwnerSchema> | undefined> {
    try {
      return lockOwnerSchema.parse(
        JSON.parse(await readFile(path.join(lockDirectory, "owner.json"), "utf8")),
      );
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") return undefined;
      throw error;
    }
  }

  private async releaseScopeClaims(scopes: readonly string[], leaseId: string): Promise<void> {
    for (const scope of scopes) {
      const lockDirectory = this.lockDirectory(scope);
      const owner = await this.readLockOwner(lockDirectory);
      if (owner?.leaseId !== leaseId) continue;
      await moveAside(lockDirectory);
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

function encode(lease: RuntimeLease): string {
  assertValidLease(lease);
  const envelope: RuntimeLeaseEnvelope = {
    schemaVersion: 1,
    kind: "runtime_lease",
    payloadDigest: digest(lease),
    payload: lease,
  };
  return `${JSON.stringify(envelope, null, 2)}\n`;
}

async function readLease(filePath: string): Promise<RuntimeLease> {
  const parsed: unknown = JSON.parse(await readFile(filePath, "utf8"));
  const envelope = z
    .object({
      schemaVersion: z.literal(1),
      kind: z.literal("runtime_lease"),
      payloadDigest: nonEmptyString,
      payload: leaseSchema,
    })
    .strict()
    .parse(parsed) as RuntimeLeaseEnvelope;
  if (digest(envelope.payload) !== envelope.payloadDigest) {
    throw new Error("Runtime lease payload digest mismatch");
  }
  assertValidLease(envelope.payload);
  return envelope.payload;
}

function assertValidLease(lease: RuntimeLease): void {
  leaseSchema.parse(lease);
  if (lease.expiresAt <= lease.acquiredAt) {
    throw new Error("Runtime lease expiry must be after acquisition");
  }
  if (lease.releasedAt && lease.releasedAt < lease.acquiredAt) {
    throw new Error("Runtime lease cannot be released before acquisition");
  }
}

function scopesConflict(left: RuntimeLease, right: RuntimeLease): boolean {
  return (
    left.workspacePath === right.workspacePath ||
    left.symbolIds.some((symbolId) => right.symbolIds.includes(symbolId)) ||
    left.paths.some((allowedPath) => right.paths.includes(allowedPath))
  );
}

function leaseScopes(lease: RuntimeLease): readonly string[] {
  return Array.from(
    new Set([
      `workspace:${path.resolve(lease.workspacePath)}`,
      ...lease.symbolIds.map((symbolId) => `symbol:${symbolId}`),
      ...lease.paths.map((allowedPath) => `path:${allowedPath}`),
    ]),
  ).sort();
}

function compareLeases(left: RuntimeLease, right: RuntimeLease): number {
  return (
    left.acquiredAt.localeCompare(right.acquiredAt) || left.leaseId.localeCompare(right.leaseId)
  );
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

async function moveAside(directory: string): Promise<boolean> {
  const target = `${directory}.${randomUUID()}.released`;
  try {
    await rename(directory, target);
    await rm(target, { recursive: true, force: true });
    return true;
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return false;
    throw error;
  }
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
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
