import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rename, rm, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { codeLanguages } from "@kontext-brain/code";
import { z } from "zod";
import type {
  WriteAuthorizationBinding,
  WriteAuthorizationBindingStore,
} from "./task-workflow-tools.js";

const workspaceSnapshotSchema = z
  .object({
    workspacePath: z.string().min(1),
    revision: z.string().min(1),
    files: z.array(
      z
        .object({
          path: z.string().min(1),
          kind: z.enum(["file", "symlink", "missing"]),
          contentDigest: z.string().min(1),
        })
        .strict(),
    ),
  })
  .strict();

const codeSymbolIdentitySchema = z
  .object({
    codebaseId: z.string().min(1),
    relativePath: z.string().min(1),
    language: z.enum(codeLanguages),
    kind: z.enum([
      "module",
      "class",
      "interface",
      "type",
      "enum",
      "function",
      "method",
      "constructor",
      "getter",
      "setter",
      "named_arrow",
      "field",
      "constant",
    ]),
    qualifiedName: z.string().min(1),
    signatureDiscriminator: z.string(),
  })
  .strict();

const codeSymbolSnapshotSchema = z
  .object({
    codebaseId: z.string().min(1),
    workspaceRevision: z.string().min(1),
    symbols: z.array(
      z
        .object({
          symbolId: z.string().min(1),
          identity: codeSymbolIdentitySchema,
          behaviorBearing: z.boolean(),
          contentHash: z.string().min(1),
        })
        .strict(),
    ),
  })
  .strict();

const bindingSchema = z
  .object({
    request: z
      .object({
        taskId: z.string().min(1),
        logic: z
          .object({
            workItemId: z.string().min(1),
            plannedSymbolIds: z.array(z.string().min(1)).min(1),
          })
          .strict(),
        runtimeProvider: z.string().min(1),
        issuedAt: z.string().datetime(),
        expiresAt: z.string().datetime(),
        totalTokenBudget: z.number().int().positive(),
        optionalEvidenceTokenBudget: z.number().int().nonnegative(),
      })
      .strict(),
    allowedPaths: z.array(z.string().min(1)).min(1),
    receipt: z
      .object({
        receiptId: z.string().min(1),
        taskId: z.string().min(1),
        workItemId: z.string().min(1),
        plannedSymbolIds: z.array(z.string().min(1)),
        allowedPaths: z.array(z.string().min(1)),
        contextDigest: z.string().min(1),
        normativeRevisions: z.array(
          z
            .object({
              kind: z.enum(["decision", "domain_term", "invariant"]),
              recordId: z.string().min(1),
              revisionId: z.string().min(1),
            })
            .strict(),
        ),
        evidenceIds: z.array(z.string().min(1)),
        issuedAt: z.string().datetime(),
        expiresAt: z.string().datetime(),
      })
      .strict(),
    initialBaseline: workspaceSnapshotSchema,
    baseline: workspaceSnapshotSchema,
    symbolBaseline: codeSymbolSnapshotSchema,
  })
  .strict();

const envelopeSchema = z
  .object({
    schemaVersion: z.literal(2),
    workspacePath: z.string().min(1),
    payloadDigest: z.string().min(1),
    binding: bindingSchema,
  })
  .strict();

/** Persists short-lived write capabilities so command hooks can fail closed. */
export class FileWriteAuthorizationBindingStore implements WriteAuthorizationBindingStore {
  constructor(private readonly pluginDataDirectory: string) {}

  async get(workspacePath: string): Promise<WriteAuthorizationBinding | undefined> {
    const normalizedWorkspace = path.resolve(workspacePath);
    let serialized: string;
    try {
      serialized = await readFile(this.filePath(normalizedWorkspace), "utf8");
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") return undefined;
      throw error;
    }
    const envelope = envelopeSchema.parse(JSON.parse(serialized));
    if (envelope.workspacePath !== normalizedWorkspace) {
      throw new Error("Write capability does not match its workspace");
    }
    if (digest(envelope.binding) !== envelope.payloadDigest) {
      throw new Error("Write capability digest mismatch");
    }
    return envelope.binding;
  }

  async list(): Promise<
    readonly {
      readonly workspacePath: string;
      readonly binding: WriteAuthorizationBinding;
    }[]
  > {
    const directory = path.join(this.pluginDataDirectory, "write-capabilities");
    let entries: string[];
    try {
      entries = await readdir(directory);
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") return [];
      throw error;
    }
    const bindings = await Promise.all(
      entries
        .filter((entry) => entry.endsWith(".json"))
        .sort()
        .map(async (entry) => {
          const envelope = envelopeSchema.parse(
            JSON.parse(await readFile(path.join(directory, entry), "utf8")),
          );
          if (digest(envelope.binding) !== envelope.payloadDigest) {
            throw new Error("Write capability digest mismatch");
          }
          return { workspacePath: envelope.workspacePath, binding: envelope.binding };
        }),
    );
    return bindings.sort((left, right) => left.workspacePath.localeCompare(right.workspacePath));
  }

  async put(workspacePath: string, binding: WriteAuthorizationBinding): Promise<void> {
    const normalizedWorkspace = path.resolve(workspacePath);
    const payload = bindingSchema.parse(binding);
    const envelope = {
      schemaVersion: 2 as const,
      workspacePath: normalizedWorkspace,
      payloadDigest: digest(payload),
      binding: payload,
    };
    const filePath = this.filePath(normalizedWorkspace);
    const directory = path.dirname(filePath);
    const temporaryPath = path.join(directory, `.${randomUUID()}.tmp`);
    await mkdir(directory, { recursive: true, mode: 0o700 });
    try {
      await writeFile(temporaryPath, `${JSON.stringify(envelope, null, 2)}\n`, {
        encoding: "utf8",
        mode: 0o600,
      });
      await rename(temporaryPath, filePath);
    } catch (error) {
      await rm(temporaryPath, { force: true }).catch(() => undefined);
      throw error;
    }
  }

  async delete(workspacePath: string): Promise<void> {
    await unlink(this.filePath(path.resolve(workspacePath))).catch((error) => {
      if (!isNodeError(error) || error.code !== "ENOENT") throw error;
    });
  }

  filePath(workspacePath: string): string {
    return path.join(
      this.pluginDataDirectory,
      "write-capabilities",
      `${createHash("sha256").update(path.resolve(workspacePath)).digest("hex")}.json`,
    );
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
