import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import type { LocalTaskPlanningOperations } from "./local-task-planning.js";
import {
  taskPlanRefinementRequestSchema,
  taskPlanningRequestSchema,
} from "./task-planning-contract.js";
import { workflowToolResult } from "./task-workflow-tools.js";

export function registerHostTaskPlanningTools(
  server: McpServer,
  operations: LocalTaskPlanningOperations,
  authorize: (token: string) => void,
): void {
  const hostToken = z.string().regex(/^[a-f0-9]{64}$/);
  const requestId = z.string().uuid();
  server.tool(
    "kontext_start_plan",
    "Host-only: explicitly start subscription-backed plan generation for a user goal. Returns promptly with a durable request; never approves or starts implementation. Retry the same request ID to inspect, not dispatch twice.",
    { ...taskPlanningRequestSchema.shape, hostToken },
    { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: true },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await operations.startPlan(request));
    },
  );
  server.tool(
    "kontext_refine_plan",
    "Host-only: explicitly generate a new unapproved draft from an exact unapproved parent and user feedback. Rechecks code and context before transmission; never changes the parent or an approved Task. Replay the same request ID without another model call.",
    { ...taskPlanRefinementRequestSchema.shape, hostToken },
    { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: true },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await operations.refinePlan(request));
    },
  );
  server.tool(
    "kontext_inspect_plan",
    "Host-only: inspect a durable plan request without starting a model or approving work.",
    { requestId, hostToken },
    { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await operations.inspectPlan(request));
    },
  );
  server.tool(
    "kontext_cancel_plan",
    "Host-only: request cancellation of this instance's planner; unknown ownership is unverifiable, never evidence of exit.",
    { requestId, hostToken },
    { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await operations.cancelPlan(request));
    },
  );
  server.tool(
    "kontext_approve_plan",
    "Host-only: register a Task from the exact proposal explicitly approved by the user, after rechecking the code and context. Does not start implementation.",
    { requestId, expectedPlanDigest: z.string().regex(/^sha256:[a-f0-9]{64}$/), hostToken },
    { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await operations.approvePlan(request));
    },
  );
}
