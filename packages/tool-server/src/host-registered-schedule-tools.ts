import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import {
  type LocalRegisteredScheduleOperations,
  registeredScheduleControlSchema,
  registeredScheduleListSchema,
  registeredScheduleResumeSchema,
  registeredScheduleSelectorSchema,
} from "./local-registered-schedules.js";
import { workflowToolResult } from "./task-workflow-tools.js";

export function registerHostRegisteredScheduleTools(
  server: McpServer,
  operations: LocalRegisteredScheduleOperations,
  authorize: (token: string) => void,
) {
  const hostToken = z.string().regex(/^[a-f0-9]{64}$/);
  server.tool(
    "kontext_list_registered_schedules",
    "Host-only: page saved execution history for an owned registered Task. Metadata only; never probes liveness, resumes workers or assesses completion. A changed cursor requires reloading the first page.",
    { ...registeredScheduleListSchema.shape, hostToken },
    { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await operations.list(request));
    },
  );
  server.tool(
    "kontext_inspect_registered_schedule",
    "Host-only: inspect saved progress for an owned registered Task and exact schedule. Never probes liveness, resumes workers or assesses completion; excludes prompts and diagnostics text.",
    { ...registeredScheduleSelectorSchema.shape, hostToken },
    { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await operations.inspect(request));
    },
  );
  server.tool(
    "kontext_resume_registered_schedule",
    "Host-only: explicitly revalidate and possibly resume the reviewed owned schedule using its stored subscription request. May start coding workers. Never invents an enqueue request, bypasses existing evidence/lease gates or falls back to API billing. Lost responses must be inspected, not automatically replayed.",
    { ...registeredScheduleResumeSchema.shape, hostToken },
    { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: true },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await operations.resume(request));
    },
  );
  server.tool(
    "kontext_cancel_registered_schedule",
    "Host-only: request cancellation of the reviewed owned schedule. Persists intent even for interrupted schedules; interrupted or cancelling is not proof all workers stopped. Never resumes workers. Lost responses must be inspected, not automatically replayed.",
    { ...registeredScheduleControlSchema.shape, hostToken },
    { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: true },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await operations.cancel(request));
    },
  );
}
