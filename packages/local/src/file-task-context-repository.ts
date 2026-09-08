import { createHash, randomUUID } from "node:crypto";
import { mkdir, open, readFile, rename, rm } from "node:fs/promises";
import path from "node:path";
import { codeLanguages } from "@kontext-brain/code";
import type {
  CurrentTaskContextState,
  PreparedTaskContext,
  PreparedTaskContextStore,
  TaskContextStateProvider,
} from "@kontext-brain/context";
import { prepareTaskContextSnapshot } from "@kontext-brain/context";
import type { TaskContract } from "@kontext-brain/spec";
import { z } from "zod";
import { listHashedRecordFiles } from "./hashed-record-inventory.js";
import { withLocalFileMutationLock } from "./local-file-mutation-lock.js";

export interface TaskContextStateWriteOptions {
  /** null is create-only; undefined preserves trusted unconditional publishing. */
  readonly expectedDigest?: string | null;
}

export interface TaskContextStateWriteResult {
  readonly digest: string;
  readonly created: boolean;
}

export interface CurrentTaskContextVersion {
  readonly state: CurrentTaskContextState;
  readonly digest: string;
}

type EnvelopeKind = "current" | "prepared" | "initial";
export interface InitialTaskOwner {
  readonly organizationId: string;
  readonly subjectId: string;
  readonly workspacePath: string;
  readonly workspaceSeed?: { readonly repositoryPath: string; readonly codeRevision: string };
  readonly requestId: string;
  readonly requestDigest: string;
  readonly contextSelection?: {
    readonly workspaceId: string;
    readonly sourceResourceIds: readonly string[];
  };
}
export interface InitializeTaskRequest {
  readonly owner: InitialTaskOwner;
  readonly contract: TaskContract;
  readonly state: CurrentTaskContextState;
  readonly additionalRequiredEvidenceIds?: readonly string[];
  readonly createdAt: string;
}

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
    provenance: z
      .object({
        resourceId: nonEmptyString,
        chunkId: nonEmptyString,
        resourceTitle: nonEmptyString,
        source: z
          .object({
            connectorId: nonEmptyString,
            externalId: nonEmptyString,
            type: nonEmptyString,
          })
          .strict(),
        observedAt: z.string().datetime(),
        contentHash: nonEmptyString,
        ontologyNodeIds: z.array(nonEmptyString),
      })
      .strict()
      .optional(),
  })
  .strict();
const codeSymbolIdentitySchema = z
  .object({
    codebaseId: nonEmptyString.optional(),
    relativePath: nonEmptyString.optional(),
    language: z.enum(codeLanguages).optional(),
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
export const logicPlanSchema = z
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
const governanceLinkSchema = z
  .object({
    plannedSymbolId: nonEmptyString,
    recordId: nonEmptyString,
    revisionId: nonEmptyString,
    origin: z.enum(["curated", "deterministic", "proposed"]),
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
    governanceLinks: z.array(governanceLinkSchema).optional(),
    sourceEvidenceIds: z.array(nonEmptyString).optional(),
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
const initialOwnerSchema = z
  .object({
    organizationId: nonEmptyString,
    subjectId: nonEmptyString,
    workspacePath: nonEmptyString,
    workspaceSeed: z
      .object({
        repositoryPath: nonEmptyString,
        codeRevision: z.string().regex(/^(?:[a-f0-9]{40}|[a-f0-9]{64})$/),
      })
      .strict()
      .optional(),
    requestId: z.string().uuid(),
    requestDigest: z.string().regex(/^sha256:[a-f0-9]{64}$/),
    contextSelection: z
      .object({ workspaceId: nonEmptyString, sourceResourceIds: z.array(nonEmptyString) })
      .strict()
      .optional(),
  })
  .strict();
const initialTaskSchema = z
  .object({
    owner: initialOwnerSchema,
    state: currentStateSchema,
    prepared: preparedSchema,
  })
  .strict();
export type InitialTaskRegistration = z.infer<typeof initialTaskSchema>;

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

  async listInitialTaskMetadata(owner: Pick<InitialTaskOwner, "organizationId" | "subjectId">) {
    const directory = path.join(this.pluginDataDirectory, "task-context", "initial");
    const tasks: {
      taskId: string;
      owner: InitialTaskOwner;
      intent: string;
      risk: TaskContract["risk"];
      createdAt: string;
    }[] = [];
    for (const filename of await listHashedRecordFiles(directory)) {
      const serialized = await readFile(path.join(directory, filename), "utf8");
      const taskId = nonEmptyString.parse(JSON.parse(serialized).taskId);
      if (path.basename(this.filePath("initial", taskId)) !== filename)
        throw new Error("Initial Task inventory storage identity mismatch");
      const { payload } = decodeEnvelope(serialized, "initial", taskId, initialTaskSchema);
      if (
        payload.prepared.contract.taskId !== taskId ||
        payload.prepared.snapshot.taskId !== taskId
      )
        throw new Error("Initial Task inventory identity mismatch");
      if (
        payload.owner.organizationId !== owner.organizationId ||
        payload.owner.subjectId !== owner.subjectId ||
        !payload.owner.contextSelection
      )
        continue;
      tasks.push({
        taskId,
        owner: payload.owner,
        intent: payload.prepared.contract.intent,
        risk: payload.prepared.contract.risk,
        createdAt: payload.prepared.snapshot.createdAt,
      });
    }
    return tasks;
  }

  /** Publishes ownership, current state and the first frozen snapshot at one visibility point. */
  async initializeTask(
    request: InitializeTaskRequest,
  ): Promise<{ created: boolean; registration: InitialTaskRegistration }> {
    const owner = initialOwnerSchema.parse(request.owner);
    const contract = taskContractSchema.parse(request.contract);
    const state = currentStateSchema.parse(request.state);
    const createdAt = z.string().datetime().parse(request.createdAt);
    const additionalRequiredEvidenceIds = [
      ...new Set(request.additionalRequiredEvidenceIds ?? []),
    ].sort();
    const prepared = {
      contract,
      snapshot: prepareTaskContextSnapshot({
        contract,
        baseCodeRevision: state.codeRevision,
        effectiveScopes: state.effectiveScopes,
        normativeRecords: state.normativeRecords,
        additionalRequiredEvidenceIds: [
          ...new Set([...additionalRequiredEvidenceIds, ...(state.sourceEvidenceIds ?? [])]),
        ].sort(),
        sourceFreshnessDigest: state.sourceFreshnessDigest,
        createdAt,
      }),
      additionalRequiredEvidenceIds,
    };
    const registration = initialTaskSchema.parse({ owner, state, prepared });
    return this.withTaskMutation(contract.taskId, async () => {
      const existing = await this.getInitialRegistration(contract.taskId);
      if (existing) {
        if (digest(existing) !== digest(registration))
          throw new Error("Task initialization conflicts with its original registration");
        return { created: false, registration: existing };
      }
      if (
        (await this.readOptional("current", contract.taskId, currentStateSchema)) ||
        (await this.readOptional("prepared", contract.taskId, preparedSchema))
      )
        throw new Error(
          "Task already has context; refusing to replace it with an initial registration",
        );
      await atomicPrivateWrite(
        this.filePath("initial", contract.taskId),
        JSON.stringify({
          schemaVersion: 1,
          kind: "initial",
          taskId: contract.taskId,
          payloadDigest: digest(registration),
          payload: registration,
        }),
      );
      return { created: true, registration };
    });
  }

  async getInitialRegistration(taskId: string): Promise<InitialTaskRegistration | undefined> {
    const value = await this.readOptional("initial", taskId, initialTaskSchema);
    if (
      value &&
      (value.prepared.contract.taskId !== taskId || value.prepared.snapshot.taskId !== taskId)
    )
      throw new Error("Initial Task registration identity mismatch");
    return value;
  }

  async getCurrent(taskId: string): Promise<CurrentTaskContextState> {
    return (await this.getCurrentVersion(taskId)).state;
  }

  async getCurrentVersion(taskId: string): Promise<CurrentTaskContextVersion> {
    const initial = await this.getInitialRegistration(taskId);
    const envelope = await this.readOptionalEnvelope("current", taskId, currentStateSchema);
    if (!envelope) {
      if (!initial) throw new Error(`Task "${taskId}" has no current context state`);
      return { state: initial.state, digest: digest(initial.state) };
    }
    return { state: envelope.payload, digest: envelope.payloadDigest };
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
    const initial = await this.getInitialRegistration(taskId);
    return (await this.readOptional("prepared", taskId, preparedSchema)) ?? initial?.prepared;
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

  private async readOptional<T>(
    kind: EnvelopeKind,
    taskId: string,
    schema: z.ZodType<T>,
  ): Promise<T | undefined> {
    return (await this.readOptionalEnvelope(kind, taskId, schema))?.payload;
  }

  private async readOptionalEnvelope<T>(
    kind: EnvelopeKind,
    taskId: string,
    schema: z.ZodType<T>,
  ): Promise<TaskContextEnvelope<T> | undefined> {
    let serialized: string;
    try {
      serialized = await readFile(this.filePath(kind, taskId), "utf8");
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") return undefined;
      throw error;
    }
    return decodeEnvelope(serialized, kind, taskId, schema);
  }

  private async write<T>(
    kind: "current" | "prepared",
    taskId: string,
    payload: T,
    options: TaskContextStateWriteOptions = {},
  ): Promise<TaskContextStateWriteResult> {
    assertTaskId(taskId);
    const filePath = this.filePath(kind, taskId);
    return this.withTaskMutation(taskId, () =>
      withLocalFileMutationLock(`${filePath}.lock`, async () => {
        const initial = await this.getInitialRegistration(taskId);
        const existing =
          (await readExistingDigest(filePath, kind, taskId)) ??
          (initial ? digest(kind === "current" ? initial.state : initial.prepared) : undefined);
        if (options.expectedDigest !== undefined && options.expectedDigest !== (existing ?? null)) {
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
      }),
    );
  }

  private withTaskMutation<T>(taskId: string, mutation: () => Promise<T>): Promise<T> {
    return withLocalFileMutationLock(`${this.filePath("initial", taskId)}.lock`, mutation);
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

async function readExistingDigest(
  filePath: string,
  kind: EnvelopeKind,
  taskId: string,
): Promise<string | undefined> {
  try {
    const serialized = await readFile(filePath, "utf8");
    return kind === "current"
      ? decodeEnvelope(serialized, kind, taskId, currentStateSchema).payloadDigest
      : decodeEnvelope(serialized, kind, taskId, preparedSchema).payloadDigest;
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
    const handle = await open(temporaryPath, "wx", 0o600);
    try {
      await handle.writeFile(serialized, "utf8");
      await handle.sync();
    } finally {
      await handle.close();
    }
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
