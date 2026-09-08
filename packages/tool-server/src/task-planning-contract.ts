import { createHash } from "node:crypto";
import { z } from "zod";
import { localTaskCreationRequestSchema } from "./local-task-creation.js";

export const taskPlanningRequestSchema = localTaskCreationRequestSchema
  .pick({
    requestId: true,
    workspacePath: true,
    workspaceId: true,
    sourceResourceIds: true,
  })
  .extend({
    goal: z.string().trim().min(1).max(32_768),
    provider: z.enum(["codex", "claude"]),
  })
  .strict();
export type TaskPlanningRequest = z.infer<typeof taskPlanningRequestSchema>;

export const taskPlanRefinementRequestSchema = z
  .object({
    requestId: z.string().uuid(),
    parentRequestId: z.string().uuid(),
    expectedParentDigest: z.string().regex(/^sha256:[a-f0-9]{64}$/),
    feedback: z.string().trim().min(1).max(8_192),
  })
  .strict();
export type TaskPlanRefinementRequest = z.infer<typeof taskPlanRefinementRequestSchema>;
export const taskPlanRefinementSchema = taskPlanRefinementRequestSchema.omit({ requestId: true });
export type TaskPlanRefinement = z.infer<typeof taskPlanRefinementSchema>;

export const taskPlanProposalSchema = localTaskCreationRequestSchema
  .pick({ contract: true, logicPlans: true })
  .superRefine((proposal, ctx) => {
    const issue = (message: string) => ctx.addIssue({ code: "custom", message });
    const exactPath = (value: string) =>
      value.length > 0 &&
      !/[\\:*?\[\]{}]/.test(value) &&
      !Array.from(value).some((char) => char.charCodeAt(0) < 32) &&
      value.split("/").every((part) => part && part !== "." && part !== ".." && part !== ".git");
    const work = new Map(proposal.logicPlans.map((plan) => [plan.workItemId, plan]));
    if (work.size !== proposal.logicPlans.length) issue("Duplicate Logic Work Item ID");
    if (
      new Set(proposal.contract.acceptance.map((item) => item.criterionId)).size !==
      proposal.contract.acceptance.length
    )
      issue("Duplicate acceptance criterion ID");
    const symbols = new Set<string>();
    const behaviorKinds = new Set([
      "function",
      "method",
      "constructor",
      "getter",
      "setter",
      "named_arrow",
    ]);
    for (const plan of proposal.logicPlans) {
      if (!plan.allowedPaths.every(exactPath))
        issue("Plan requires exact workspace-relative paths");
      if (
        !plan.plannedSymbols ||
        plan.plannedSymbols.length !== plan.plannedSymbolIds.length ||
        new Set(plan.plannedSymbolIds).size !== plan.plannedSymbolIds.length
      )
        issue("Describe every Planned Symbol exactly once");
      for (const symbol of plan.plannedSymbols ?? []) {
        if (
          symbols.has(symbol.plannedSymbolId) ||
          !plan.plannedSymbolIds.includes(symbol.plannedSymbolId)
        )
          issue("Planned Symbol must have one Logic Work Item owner");
        symbols.add(symbol.plannedSymbolId);
        if (
          !symbol.intendedIdentity.relativePath ||
          !plan.allowedPaths.includes(symbol.intendedIdentity.relativePath) ||
          !behaviorKinds.has(symbol.intendedIdentity.kind ?? "")
        )
          issue("Planned Symbol requires a behavior-bearing kind and an allowed path");
      }
      if (plan.capabilityId) issue("The host mints implementation capabilities after approval");
      if (plan.plannedSymbols?.some((symbol) => symbol.boundSymbolId))
        issue("Proposed symbols cannot assert a verified binding");
    }
    const visiting = new Set<string>();
    const visited = new Set<string>();
    const visit = (id: string): boolean => {
      if (visiting.has(id) || !work.has(id)) return false;
      if (visited.has(id)) return true;
      visiting.add(id);
      for (const dependency of work.get(id)?.dependsOn ?? []) if (!visit(dependency)) return false;
      visiting.delete(id);
      visited.add(id);
      return true;
    };
    if (![...work.keys()].every(visit))
      issue("Logic dependencies must be an acyclic graph of known Work Items");
  });
export type TaskPlanProposal = z.infer<typeof taskPlanProposalSchema>;

export const taskPlanRecordSchema = z
  .object({
    schemaVersion: z.literal(1),
    organizationId: z.string().uuid(),
    subjectId: z.string().min(1),
    request: taskPlanningRequestSchema,
    refinement: taskPlanRefinementSchema.optional(),
    status: z.enum(["planning", "review", "failed", "unverifiable", "approved"]),
    requestedAt: z.string().datetime(),
    codeRevision: z.string().optional(),
    contextDigest: z.string().optional(),
    evidenceIds: z.array(z.string()).optional(),
    proposal: taskPlanProposalSchema.optional(),
    planDigest: z
      .string()
      .regex(/^sha256:[a-f0-9]{64}$/)
      .optional(),
    taskId: z.string().optional(),
    diagnostic: z.string().optional(),
  })
  .strict();
export type TaskPlanRecord = z.infer<typeof taskPlanRecordSchema>;

export function taskPlanningDigest(value: unknown): string {
  const canonical = (value: unknown): unknown =>
    Array.isArray(value)
      ? value.map(canonical)
      : value && typeof value === "object"
        ? Object.fromEntries(
            Object.entries(value)
              .filter(([, item]) => item !== undefined)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([key, item]) => [key, canonical(item)]),
          )
        : value;
  return `sha256:${createHash("sha256")
    .update(JSON.stringify(canonical(value)))
    .digest("hex")}`;
}
