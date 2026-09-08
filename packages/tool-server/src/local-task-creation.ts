import { createHash } from "node:crypto";
import path from "node:path";
import {
  FileTaskContextRepository,
  assembleCurrentTaskContextState,
  logicPlanSchema,
  withLocalFileMutationLock,
} from "@kontext-brain/local";
import { z } from "zod";
import { loadLocalKnowledgePrincipal } from "./local-knowledge-principal.js";
import { sourceResourceIdSchema } from "./local-source-registry.js";
import { collectPersonalTaskContext, prepareTaskWorkspace } from "./local-task-preparation.js";
import { inspectRuntimeTask } from "./runtime-task-inspection.js";
import { prepareTaskToolShape } from "./task-workflow-tools.js";

const creationPlanSchema = logicPlanSchema.extend({
  plannedSymbols: z
    .array(logicPlanSchema.shape.plannedSymbols.unwrap().element.omit({ taskId: true }))
    .min(1)
    .optional(),
});
export const localTaskCreationRequestSchema = z
  .object({
    requestId: z.string().uuid(),
    workspacePath: z.string().min(1),
    workspaceId: z.string().min(1),
    expectedCodeRevision: z.string().regex(/^[a-f0-9]{40,64}$/),
    expectedSourceFreshnessDigest: z
      .string()
      .regex(/^sha256:[a-f0-9]{64}$/)
      .optional(),
    contract: prepareTaskToolShape.contract.omit({ taskId: true }),
    sourceResourceIds: z.array(sourceResourceIdSchema).max(32),
    logicPlans: z.array(creationPlanSchema).min(1).max(128),
  })
  .strict();
export type LocalTaskCreationRequest = z.infer<typeof localTaskCreationRequestSchema>;

/** Finalizes a reviewed personal-context plan; does not ask a model to plan or start workers. */
export class LocalTaskCreationOperations {
  constructor(
    private readonly dataDirectory: string,
    private readonly environment: NodeJS.ProcessEnv = process.env,
  ) {}

  async createTask(input: LocalTaskCreationRequest) {
    const request = localTaskCreationRequestSchema.parse(input);
    const principal = await loadLocalKnowledgePrincipal(this.dataDirectory);
    const taskId = `host-task:${digest([principal.organizationId, principal.subjectId, request.requestId]).slice(7)}`;
    const requestDigest = digest(request);
    const repository = new FileTaskContextRepository(this.dataDirectory);
    return withLocalFileMutationLock(
      path.join(this.dataDirectory, "task-creation", `${digest(taskId).slice(7)}.lock`),
      async () => {
        const existing = await repository.getInitialRegistration(taskId);
        if (existing) {
          if (
            existing.owner.organizationId !== principal.organizationId ||
            existing.owner.subjectId !== principal.subjectId ||
            existing.owner.requestId !== request.requestId ||
            existing.owner.requestDigest !== requestDigest
          )
            throw new Error(
              "Task creation request conflicts with its registered owner or original request",
            );
          return {
            created: false,
            taskId,
            inspectionBasis: "stored_context" as const,
            inspection: await inspectRuntimeTask(taskId, repository, repository),
          };
        }
        const { workspacePath, workspaceSeed } = await prepareTaskWorkspace(
          this.dataDirectory,
          request,
          this.environment,
        );
        const collected = await collectPersonalTaskContext(this.dataDirectory, {
          ...request,
          taskId,
          codeRevision: request.expectedCodeRevision,
        });
        if (
          request.expectedSourceFreshnessDigest !== undefined &&
          request.expectedSourceFreshnessDigest !== collected.state.sourceFreshnessDigest
        )
          throw new Error("Reviewed planning context changed before Task creation");
        const contract = { ...request.contract, taskId };
        const assembled = assembleCurrentTaskContextState({
          taskId,
          organizationId: principal.organizationId,
          codeRevision: request.expectedCodeRevision,
          baseScopes: [
            { kind: "personal", subjectId: principal.subjectId },
            { kind: "workspace", workspaceId: request.workspaceId },
          ],
          localManifest: collected.localManifest,
          evidence: collected.state.evidence,
          logicPlans: request.logicPlans.map(({ plannedSymbols, ...plan }) => ({
            ...plan,
            ...(plannedSymbols
              ? { plannedSymbols: plannedSymbols.map((symbol) => ({ ...symbol, taskId })) }
              : {}),
          })),
        });
        if (
          (await prepareTaskWorkspace(this.dataDirectory, request, this.environment))
            .workspacePath !== workspacePath
        )
          throw new Error("Reviewed workspace identity changed during Task creation");
        const state = {
          ...assembled,
          sourceEvidenceIds: collected.sourceEvidenceIds,
        };
        await repository.initializeTask({
          owner: {
            organizationId: principal.organizationId,
            subjectId: principal.subjectId,
            workspacePath,
            ...(workspaceSeed ? { workspaceSeed } : {}),
            requestId: request.requestId,
            requestDigest,
            contextSelection: {
              workspaceId: request.workspaceId,
              sourceResourceIds: [...new Set(request.sourceResourceIds)].sort(),
            },
          },
          contract,
          state,
          additionalRequiredEvidenceIds: [],
          createdAt: new Date().toISOString(),
        });
        return {
          created: true,
          taskId,
          inspectionBasis: "stored_context" as const,
          inspection: await inspectRuntimeTask(taskId, repository, repository),
        };
      },
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
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, nested]) => [key, stableValue(nested)]),
    );
  return value;
}
