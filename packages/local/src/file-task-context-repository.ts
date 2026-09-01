import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import type {
  CurrentTaskContextState,
  PreparedTaskContext,
  PreparedTaskContextStore,
  TaskContextStateProvider,
} from "@kontext-brain/context";
import { z } from "zod";

export interface TaskContextStateWriteOptions {
  readonly expectedDigest?: string;
}

export interface TaskContextStateWriteResult {
  readonly digest: string;
  readonly created: boolean;
}

type EnvelopeKind = "current" | "prepared";

interface TaskContextEnvelope<T> {
  readonly schemaVersion: 1;
  readonly kind: EnvelopeKind;
  readonly taskId: string;
  readonly payloadDigest: string;
  readonly payload: T;
}

const nonEmptyString = z.string().min(1);
const scopeSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("personal"), subjectId: nonEmptyString }).strict(),
  z.object({ kind: z.literal("workspace"), workspaceId: nonEmptyString }).strict(),
  z.object({ kind: z.literal("codebase"), codebaseId: nonEmptyString }).strict(),
  z.object({ kind: z.literal("organization"), organizationId: nonEmptyString }).strict(),
]);
const evidenceRefSchema = z
  .object({ evidenceId: nonEmptyString, sourceSpan: z.string().optional() })
  .strict();
const egressSchema = z
  .object({
    dataClassification: z.enum(["public", "internal", "confidential", "restricted"]),
    allowedRuntimeProviders: z.array(nonEmptyString),
  })
  .strict();
const normativeBaseShape = {
  organizationId: nonEmptyString,
  recordId: nonEmptyString,
  revisionId: nonEmptyString,
  scope: scopeSchema,
  evidence: z.array(evidenceRefSchema).min(1),
  egress: egressSchema,
  authoredBy: nonEmptyString,
  authoredAt: nonEmptyString,
  supersedesRevisionId: nonEmptyString.optional(),
};
const verifierSchema = z
  .object({
    kind: z.enum(["test", "typecheck", "build", "lint", "query", "manual_review"]),
    ref: nonEmptyString,
  })
  .strict();
const normativeRevisionSchema = z.discriminatedUnion("kind", [
  z
    .object({ ...normativeBaseShape, kind: z.literal("decision"), statement: nonEmptyString })
    .strict(),
  z
    .object({
      ...normativeBaseShape,
      kind: z.literal("domain_term"),
      term: nonEmptyString,
      definition: nonEmptyString,
      avoid: z.array(z.string()).optional(),
    })
    .strict(),
  z
    .object({
      ...normativeBaseShape,
      kind: z.literal("invariant"),
      statement: nonEmptyString,
      verifiers: z.array(verifierSchema),
    })
    .strict(),
]);
const activationSchema = z
  .object({
    organizationId: nonEmptyString,
    kind: z.enum(["decision", "domain_term", "invariant"]),
    recordId: nonEmptyString,
    revisionId: nonEmptyString,
    scope: scopeSchema,
    state: z.enum(["accepted_local", "accepted", "retired"]),
    acceptedBy: nonEmptyString,
    acceptedAt: nonEmptyString,
    mergeCommit: nonEmptyString.optional(),
  })
  .strict();
const effectiveRecordSchema = z
  .object({
    origin: z.enum(["local", "managed"]),
    revision: normativeRevisionSchema,
    activation: activationSchema,
  })
  .strict();
const conflictSchema = z
  .object({
    kind: z.enum(["decision", "domain_term", "invariant"]),
    recordId: nonEmptyString,
    localRevisionId: nonEmptyString,
    managedRevisionIds: z.array(nonEmptyString),
  })
  .strict();
const contextEvidenceSchema = z
  .object({
    evidenceId: nonEmptyString,
    text: z.string(),
    sourceSpan: z.string().optional(),
    availability: z.enum(["current", "stale", "conflict", "inaccessible", "unavailable"]),
    allowedRuntimeProviders: z.array(nonEmptyString),
    relevance: z.number().optional(),
  })
  .strict();
const codeSymbolIdentitySchema = z
  .object({
    codebaseId: nonEmptyString.optional(),
    relativePath: nonEmptyString.optional(),
    language: z.enum(["typescript", "javascript"]).optional(),
    kind: z
      .enum([
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
      ])
      .optional(),
    qualifiedName: nonEmptyString.optional(),
    signatureDiscriminator: z.string().optional(),
  })
  .strict();
const plannedSymbolSchema = z
  .object({
    plannedSymbolId: nonEmptyString,
    taskId: nonEmptyString,
    intendedIdentity: codeSymbolIdentitySchema,
    responsibility: nonEmptyString,
    boundSymbolId: nonEmptyString.optional(),
  })
  .strict();
const logicPlanSchema = z
  .object({
    workItemId: nonEmptyString,
    plannedSymbolIds: z.array(nonEmptyString).min(1),
    plannedSymbols: z.array(plannedSymbolSchema).min(1).optional(),
    allowedPaths: z.array(nonEmptyString).min(1),
    dependsOn: z.array(nonEmptyString).optional(),
    requiredVerifiers: z.array(verifierSchema).optional(),
    capabilityId: nonEmptyString.optional(),
  })
  .strict();
const currentStateSchema = z
  .object({
    codeRevision: nonEmptyString,
    sourceFreshnessDigest: nonEmptyString,
    effectiveScopes: z.array(scopeSchema),
    normativeRecords: z.array(effectiveRecordSchema),
    normativeRevisionCatalog: z.array(normativeRevisionSchema),
    conflicts: z.array(conflictSchema),
    evidence: z.array(contextEvidenceSchema),
    logicPlans: z.array(logicPlanSchema),
  })
  .strict();
const taskContractSchema = z
  .object({
    taskId: nonEmptyString,
    intent: nonEmptyString,
    acceptance: z
      .array(
        z
          .object({
            criterionId: nonEmptyString,
            statement: nonEmptyString,
            verifier: verifierSchema,
          })
          .strict(),
      )
      .min(1),
    nonGoals: z.array(z.string()),
    targets: z.array(nonEmptyString).min(1),
    risk: z.enum(["low", "medium", "high"]),
  })
  .strict();
const revisionRefSchema = z
  .object({
    kind: z.enum(["decision", "domain_term", "invariant"]),
    recordId: nonEmptyString,
    revisionId: nonEmptyString,
  })
  .strict();
const snapshotSchema = z
  .object({
    taskId: nonEmptyString,
    baseCodeRevision: nonEmptyString,
    effectiveScopes: z.array(scopeSchema),
    normativeRevisions: z.array(revisionRefSchema),
    requiredEvidenceIds: z.array(nonEmptyString),
    sourceFreshnessDigest: nonEmptyString,
    contextDigest: nonEmptyString,
    createdAt: nonEmptyString,
  })
  .strict();
const preparedSchema = z
  .object({
    contract: taskContractSchema,
    snapshot: snapshotSchema,
    additionalRequiredEvidenceIds: z.array(nonEmptyString),
  })
  .strict();

/**
 * Durable, private task context state owned by the local sidecar.
 *
 * File names are hashes of Task IDs, payloads are integrity checked, and every
 * write uses a same-directory atomic rename with owner-only permissions.
 */
export class FileTaskContextRepository
  implements TaskContextStateProvider, PreparedTaskContextStore
{
  constructor(private readonly pluginDataDirectory: string) {}

  async getCurrent(taskId: string): Promise<CurrentTaskContextState> {
    return this.readRequired("current", taskId, currentStateSchema);
  }

  async publishCurrent(
    taskId: string,
    state: CurrentTaskContextState,
    options: TaskContextStateWriteOptions = {},
  ): Promise<TaskContextStateWriteResult> {
    const payload = currentStateSchema.parse(state) as CurrentTaskContextState;
    return this.write("current", taskId, payload, options);
  }

  async get(taskId: string): Promise<PreparedTaskContext | undefined> {
    return this.readOptional("prepared", taskId, preparedSchema);
  }

  async put(value: PreparedTaskContext): Promise<void> {
    const payload = preparedSchema.parse(value) as PreparedTaskContext;
    if (payload.contract.taskId !== payload.snapshot.taskId) {
      throw new Error("Prepared Task Contract and Snapshot Task IDs do not match");
    }
    await this.write("prepared", payload.contract.taskId, payload);
  }

  currentStateFilePath(taskId: string): string {
    return this.filePath("current", taskId);
  }

  preparedTaskFilePath(taskId: string): string {
    return this.filePath("prepared", taskId);
  }

  private async readRequired<T>(
    kind: EnvelopeKind,
    taskId: string,
    schema: z.ZodType<T>,
  ): Promise<T> {
    const value = await this.readOptional(kind, taskId, schema);
    if (!value) throw new Error(`Task "${taskId}" has no ${kind} context state`);
    return value;
  }

  private async readOptional<T>(
    kind: EnvelopeKind,
    taskId: string,
    schema: z.ZodType<T>,
  ): Promise<T | undefined> {
    let serialized: string;
    try {
      serialized = await readFile(this.filePath(kind, taskId), "utf8");
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") return undefined;
      throw error;
    }
    const envelope = decodeEnvelope(serialized, kind, taskId, schema);
    return envelope.payload;
  }

  private async write<T>(
    kind: EnvelopeKind,
    taskId: string,
    payload: T,
    options: TaskContextStateWriteOptions = {},
  ): Promise<TaskContextStateWriteResult> {
    assertTaskId(taskId);
    const filePath = this.filePath(kind, taskId);
    const existing = await readExistingDigest(filePath);
    if (options.expectedDigest !== undefined && options.expectedDigest !== existing) {
      throw new Error(`${kind} Task context state changed since it was read`);
    }
    const payloadDigest = digest(payload);
    const envelope: TaskContextEnvelope<T> = {
      schemaVersion: 1,
      kind,
      taskId,
      payloadDigest,
      payload,
    };
    await atomicPrivateWrite(filePath, `${JSON.stringify(envelope, null, 2)}\n`);
    return { digest: payloadDigest, created: existing === undefined };
  }

  private filePath(kind: EnvelopeKind, taskId: string): string {
    assertTaskId(taskId);
    return path.join(
      this.pluginDataDirectory,
      "task-context",
      kind,
      `${createHash("sha256").update(taskId).digest("hex")}.json`,
    );
  }
}

function decodeEnvelope<T>(
  serialized: string,
  kind: EnvelopeKind,
  taskId: string,
  schema: z.ZodType<T>,
): TaskContextEnvelope<T> {
  const parsed: unknown = JSON.parse(serialized);
  const envelopeSchema = z
    .object({
      schemaVersion: z.literal(1),
      kind: z.literal(kind),
      taskId: nonEmptyString,
      payloadDigest: nonEmptyString,
      payload: schema,
    })
    .strict();
  const envelope = envelopeSchema.parse(parsed) as TaskContextEnvelope<T>;
  if (envelope.taskId !== taskId) {
    throw new Error(`${kind} Task context state does not match its storage location`);
  }
  if (digest(envelope.payload) !== envelope.payloadDigest) {
    throw new Error(`${kind} Task context payload digest mismatch`);
  }
  return envelope;
}

async function readExistingDigest(filePath: string): Promise<string | undefined> {
  try {
    const parsed: unknown = JSON.parse(await readFile(filePath, "utf8"));
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      !("payloadDigest" in parsed) ||
      typeof parsed.payloadDigest !== "string"
    ) {
      throw new Error("Invalid Task context envelope");
    }
    return parsed.payloadDigest;
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return undefined;
    throw error;
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

function assertTaskId(taskId: string): void {
  if (!taskId.trim()) throw new Error("Task ID is required");
}

function isNodeError(value: unknown): value is NodeJS.ErrnoException {
  return value instanceof Error && "code" in value;
}
