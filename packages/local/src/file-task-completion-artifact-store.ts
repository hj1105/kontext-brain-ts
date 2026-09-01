import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import type { TaskCompletionArtifactStore } from "@kontext-brain/orchestrator";
import type { AccuracyManifest, ChangeBundle, VerificationRun } from "@kontext-brain/spec";
import { z } from "zod";

const nonEmptyString = z.string().min(1);
const revisionRefSchema = z
  .object({
    kind: z.enum(["decision", "domain_term", "invariant"]),
    recordId: nonEmptyString,
    revisionId: nonEmptyString,
  })
  .strict();
const verificationRunSchema = z
  .object({
    verificationRunId: nonEmptyString,
    tier: z.enum(["fast", "targeted", "full"]),
    verifierKind: z.enum(["test", "typecheck", "build", "lint", "query", "manual_review"]),
    verifierRef: nonEmptyString,
    codeRevision: nonEmptyString,
    contextDigest: nonEmptyString,
    subjectIds: z.array(nonEmptyString),
    result: z.enum(["passed", "failed", "inconclusive"]),
    outputDigest: nonEmptyString.optional(),
    observedAt: nonEmptyString,
  })
  .strict();
const changeBundleSchema = z
  .object({
    bundleId: nonEmptyString,
    taskId: nonEmptyString,
    workItemId: nonEmptyString,
    baseRevision: nonEmptyString,
    resultRevision: nonEmptyString,
    taskContextDigest: nonEmptyString,
    patchDigest: nonEmptyString,
    changedSymbolIds: z.array(nonEmptyString),
    changedPaths: z.array(nonEmptyString),
    contextReceiptIds: z.array(nonEmptyString),
    evidenceIds: z.array(nonEmptyString),
    normativeRevisions: z.array(revisionRefSchema),
    verificationRunIds: z.array(nonEmptyString),
    proposals: z.array(nonEmptyString),
    unresolved: z.array(nonEmptyString),
    submittedAt: nonEmptyString,
  })
  .strict();
const accuracyManifestSchema = z
  .object({
    manifestId: nonEmptyString,
    taskId: nonEmptyString,
    taskContractDigest: nonEmptyString,
    contextDigest: nonEmptyString,
    baseCodeRevision: nonEmptyString,
    resultCodeRevision: nonEmptyString,
    normativeRevisions: z.array(revisionRefSchema),
    evidenceIds: z.array(nonEmptyString),
    workItemIds: z.array(nonEmptyString),
    changeBundleIds: z.array(nonEmptyString),
    changedSymbolIds: z.array(nonEmptyString),
    verificationRunIds: z.array(nonEmptyString),
    reviewFindingIds: z.array(nonEmptyString),
    emergencyBypassIds: z.array(nonEmptyString),
    createdAt: nonEmptyString,
  })
  .strict();
const payloadSchema = z
  .object({
    verificationRuns: z.array(verificationRunSchema),
    changeBundles: z.array(changeBundleSchema),
    accuracyManifest: accuracyManifestSchema.optional(),
  })
  .strict();

interface TaskCompletionPayload {
  readonly verificationRuns: readonly VerificationRun[];
  readonly changeBundles: readonly ChangeBundle[];
  readonly accuracyManifest?: AccuracyManifest;
}

interface TaskCompletionEnvelope {
  readonly schemaVersion: 1;
  readonly kind: "task_completion_artifacts";
  readonly taskId: string;
  readonly payloadDigest: string;
  readonly payload: TaskCompletionPayload;
}

export class FileTaskCompletionArtifactStore implements TaskCompletionArtifactStore {
  constructor(private readonly pluginDataDirectory: string) {}

  async listVerificationRuns(taskId: string): Promise<readonly VerificationRun[]> {
    return (await this.read(taskId)).verificationRuns;
  }

  async putVerificationRuns(
    taskId: string,
    runs: readonly VerificationRun[],
  ): Promise<readonly VerificationRun[]> {
    const current = await this.read(taskId);
    const byId = new Map(current.verificationRuns.map((run) => [run.verificationRunId, run]));
    for (const run of runs) {
      const existing = byId.get(run.verificationRunId);
      if (existing && JSON.stringify(existing) !== JSON.stringify(run)) {
        throw new Error(`Verification Run ${run.verificationRunId} is immutable`);
      }
      byId.set(run.verificationRunId, verificationRunSchema.parse(run) as VerificationRun);
    }
    const verificationRuns = Array.from(byId.values()).sort((left, right) =>
      left.verificationRunId.localeCompare(right.verificationRunId),
    );
    await this.write(taskId, { ...current, verificationRuns });
    return verificationRuns;
  }

  async listChangeBundles(taskId: string): Promise<readonly ChangeBundle[]> {
    return (await this.read(taskId)).changeBundles;
  }

  async putChangeBundle(bundle: ChangeBundle): Promise<ChangeBundle> {
    const current = await this.read(bundle.taskId);
    const existing = current.changeBundles.find(
      (candidate) => candidate.bundleId === bundle.bundleId,
    );
    if (existing) {
      if (JSON.stringify(existing) !== JSON.stringify(bundle)) {
        throw new Error(`Change Bundle ${bundle.bundleId} is immutable`);
      }
      return existing;
    }
    const parsed = changeBundleSchema.parse(bundle) as ChangeBundle;
    await this.write(bundle.taskId, {
      ...current,
      changeBundles: [...current.changeBundles, parsed].sort((left, right) =>
        left.bundleId.localeCompare(right.bundleId),
      ),
    });
    return parsed;
  }

  async getAccuracyManifest(taskId: string): Promise<AccuracyManifest | undefined> {
    return (await this.read(taskId)).accuracyManifest;
  }

  async putAccuracyManifest(manifest: AccuracyManifest): Promise<AccuracyManifest> {
    const current = await this.read(manifest.taskId);
    if (
      current.accuracyManifest &&
      current.accuracyManifest.manifestId === manifest.manifestId &&
      JSON.stringify(current.accuracyManifest) !== JSON.stringify(manifest)
    ) {
      throw new Error(`Accuracy Manifest ${manifest.manifestId} is immutable`);
    }
    const parsed = accuracyManifestSchema.parse(manifest) as AccuracyManifest;
    await this.write(manifest.taskId, { ...current, accuracyManifest: parsed });
    return parsed;
  }

  filePath(taskId: string): string {
    return path.join(
      this.pluginDataDirectory,
      "task-completion",
      `${createHash("sha256").update(taskId).digest("hex")}.json`,
    );
  }

  private async read(taskId: string): Promise<TaskCompletionPayload> {
    let serialized: string;
    try {
      serialized = await readFile(this.filePath(taskId), "utf8");
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") {
        return { verificationRuns: [], changeBundles: [] };
      }
      throw error;
    }
    const parsed: unknown = JSON.parse(serialized);
    const envelope = z
      .object({
        schemaVersion: z.literal(1),
        kind: z.literal("task_completion_artifacts"),
        taskId: nonEmptyString,
        payloadDigest: nonEmptyString,
        payload: payloadSchema,
      })
      .strict()
      .parse(parsed) as TaskCompletionEnvelope;
    if (envelope.taskId !== taskId) {
      throw new Error("Task completion artifacts do not match their storage location");
    }
    if (digest(envelope.payload) !== envelope.payloadDigest) {
      throw new Error("Task completion artifact payload digest mismatch");
    }
    return envelope.payload;
  }

  private async write(taskId: string, payload: TaskCompletionPayload): Promise<void> {
    const parsed = payloadSchema.parse(payload) as TaskCompletionPayload;
    const envelope: TaskCompletionEnvelope = {
      schemaVersion: 1,
      kind: "task_completion_artifacts",
      taskId,
      payloadDigest: digest(parsed),
      payload: parsed,
    };
    await atomicPrivateWrite(this.filePath(taskId), `${JSON.stringify(envelope, null, 2)}\n`);
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
