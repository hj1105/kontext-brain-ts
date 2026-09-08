import { createHash, randomUUID } from "node:crypto";
import { mkdir, open, opendir, readFile, rename, unlink } from "node:fs/promises";
import path from "node:path";
import type { Principal } from "@kontext-brain/core";
import { z } from "zod";
import {
  type NativeSessionOrigin,
  nativeSessionOriginSchema,
} from "./native-session-source-reader.js";

const hash = z.string().regex(/^sha256:[a-f0-9]{64}$/);
export const sourceInventoryRequestSchema = z
  .object({
    limit: z.number().int().min(1).max(100).default(50),
    includeNativeSessions: z.boolean().default(false),
    cursor: z
      .object({ digest: hash, offset: z.number().int().min(1).max(10_000) })
      .strict()
      .optional(),
  })
  .strict();
export const sourceResourceIdSchema = z.string().min(1).max(65_536);
export const sourceSharingRequestSchema = z
  .object({
    resourceId: sourceResourceIdSchema,
    expectedRevision: z.number().int().positive().max(Number.MAX_SAFE_INTEGER),
    expectedContentHash: hash,
    dataClassification: z.enum(["public", "internal", "confidential", "restricted"]),
    allowedRuntimeProviders: z.array(z.enum(["codex", "claude"])).max(2),
  })
  .strict();
export type SourceSharingRequest = z.infer<typeof sourceSharingRequestSchema>;
const sharingSchema = sourceSharingRequestSchema.pick({
  dataClassification: true,
  allowedRuntimeProviders: true,
});
const auditSchema = z
  .object({
    revision: z.number().int().positive(),
    contentHash: hash,
    actor: z.string().min(1),
    at: z.string().datetime(),
    reason: z.enum(["captured", "source_changed", "sharing_changed"]),
    sharing: sharingSchema.nullable(),
  })
  .strict();
const markdownRecordSchema = z
  .object({
    schemaVersion: z.literal(1),
    organizationId: z.string().min(1),
    subjectId: z.string().min(1),
    resourceId: z.string().min(1),
    workspacePath: z.string().min(1),
    relativePath: z.string().min(1),
    sourceExternalId: z.string().min(1),
    contentHash: hash,
    revision: z.number().int().positive().max(Number.MAX_SAFE_INTEGER),
    sharing: sharingSchema.nullable(),
    audit: z.array(auditSchema).min(1),
  })
  .strict();
const recordSchema = z.union([
  markdownRecordSchema,
  markdownRecordSchema
    .omit({ schemaVersion: true, workspacePath: true, relativePath: true })
    .extend({
      schemaVersion: z.literal(2),
      nativeSession: nativeSessionOriginSchema,
    }),
]);
export type LocalSourceRecord = z.infer<typeof recordSchema>;
export type LocalSourceLocator =
  | { schemaVersion: 1; workspacePath: string; relativePath: string }
  | { schemaVersion: 2; nativeSession: NativeSessionOrigin };
export function sourceLocator(record: LocalSourceRecord): LocalSourceLocator {
  return record.schemaVersion === 1
    ? { schemaVersion: 1, workspacePath: record.workspacePath, relativePath: record.relativePath }
    : { schemaVersion: 2, nativeSession: record.nativeSession };
}
export function sourceConnector(record: LocalSourceRecord): string {
  return record.schemaVersion === 1 ? "local-markdown" : "kondex-session";
}

/** Host metadata, not ontology Facts. Writers hold LocalKnowledgeOperations' management lock. */
export class LocalSourceRegistry {
  constructor(private readonly dataDirectory: string) {}

  async list(principal: Principal): Promise<string[]> {
    const directory = path.join(this.dataDirectory, "knowledge", "source-registry");
    let entries: Awaited<ReturnType<typeof opendir>>;
    try {
      entries = await opendir(directory);
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT") return [];
      throw error;
    }
    const resourceIds: string[] = [];
    let scanned = 0;
    for await (const entry of entries) {
      if (++scanned > 10_000) throw new Error("Source inventory exceeds its scan limit");
      if (!/^[a-f0-9]{64}\.json$/.test(entry.name)) continue;
      if (!entry.isFile()) throw new Error("Source registration must be a regular file");
      const filename = path.join(directory, entry.name);
      const candidate = recordSchema.parse(JSON.parse(await readFile(filename, "utf8")));
      if (
        candidate.organizationId !== principal.organizationId ||
        candidate.subjectId !== principal.subjectId
      )
        continue;
      if (filename !== this.filename(principal, candidate.resourceId))
        throw new Error("Source registration location mismatch");
      const record = await this.get(principal, candidate.resourceId);
      if (!record) throw new Error("Source registration disappeared during inventory");
      resourceIds.push(record.resourceId);
    }
    return resourceIds.sort();
  }

  async get(principal: Principal, resourceId: string): Promise<LocalSourceRecord | null> {
    let bytes: string;
    try {
      bytes = await readFile(this.filename(principal, resourceId), "utf8");
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT") return null;
      throw error;
    }
    const record = recordSchema.parse(JSON.parse(bytes));
    if (
      record.organizationId !== principal.organizationId ||
      record.subjectId !== principal.subjectId ||
      record.resourceId !== resourceId
    )
      throw new Error("Source registration identity mismatch");
    const last = record.audit.at(-1);
    if (
      !last ||
      last.revision !== record.revision ||
      last.contentHash !== record.contentHash ||
      JSON.stringify(last.sharing) !== JSON.stringify(record.sharing) ||
      record.audit.some(
        (event, index) => event.actor !== principal.subjectId || event.revision !== index + 1,
      )
    )
      throw new Error("Source registration audit mismatch");
    return record;
  }

  async save(record: LocalSourceRecord): Promise<void> {
    const validated = recordSchema.parse(record);
    const file = this.filename(validated, validated.resourceId);
    await mkdir(path.dirname(file), { recursive: true, mode: 0o700 });
    const temporary = `${file}.${randomUUID()}.tmp`;
    const handle = await open(temporary, "wx", 0o600);
    try {
      await handle.writeFile(JSON.stringify(validated));
      await handle.sync();
      await handle.close();
      await rename(temporary, file);
    } finally {
      await handle.close();
      await unlink(temporary).catch((error: unknown) => {
        if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) throw error;
      });
    }
  }

  private filename(principal: Pick<Principal, "organizationId" | "subjectId">, resourceId: string) {
    const key = createHash("sha256")
      .update(JSON.stringify([principal.organizationId, principal.subjectId, resourceId]))
      .digest("hex");
    return path.join(this.dataDirectory, "knowledge", "source-registry", `${key}.json`);
  }
}
