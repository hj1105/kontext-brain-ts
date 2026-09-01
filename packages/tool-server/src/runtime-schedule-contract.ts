import type { RuntimeProvider } from "@kontext-brain/orchestrator";
import { z } from "zod";

const nonEmptyString = z.string().min(1);
const providerSchema = z.enum(["codex", "claude"]);

export const inspectRuntimesToolShape = {};
export const scheduleLogicToolShape = {
  taskId: nonEmptyString,
  repositoryPath: nonEmptyString,
  work: z
    .array(
      z
        .object({
          workItemId: nonEmptyString,
          prompt: nonEmptyString,
          eligibleProviders: z.array(providerSchema).min(1),
          pinnedProvider: providerSchema.optional(),
          totalTokenBudget: z.number().int().positive().optional(),
          optionalEvidenceTokenBudget: z.number().int().nonnegative().optional(),
          receiptTtlSeconds: z.number().int().min(60).max(3600).optional(),
        })
        .strict(),
    )
    .min(1),
  maxConcurrency: z.number().int().min(1).max(4).optional(),
  maxRetries: z.number().int().min(0).max(2).optional(),
};
export const getScheduleToolShape = { jobId: nonEmptyString };
export const cancelScheduleToolShape = { jobId: nonEmptyString };
export const integrateScheduleToolShape = {
  jobId: nonEmptyString,
  observedAt: z.string().datetime(),
  nextAttemptAt: z.string().datetime(),
};

export const scheduleLogicRequestSchema = z.object(scheduleLogicToolShape).strict();

export interface ScheduleLogicRequest {
  readonly taskId: string;
  readonly repositoryPath: string;
  readonly work: readonly {
    readonly workItemId: string;
    readonly prompt: string;
    readonly eligibleProviders: readonly RuntimeProvider[];
    readonly pinnedProvider?: RuntimeProvider;
    readonly totalTokenBudget?: number;
    readonly optionalEvidenceTokenBudget?: number;
    readonly receiptTtlSeconds?: number;
  }[];
  readonly maxConcurrency?: number;
  readonly maxRetries?: number;
}

export interface GetScheduleRequest {
  readonly jobId: string;
}

export type CancelScheduleRequest = GetScheduleRequest;

export interface IntegrateScheduleRequest {
  readonly jobId: string;
  readonly observedAt: string;
  readonly nextAttemptAt: string;
}
