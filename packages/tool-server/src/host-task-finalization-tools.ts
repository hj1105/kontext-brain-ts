import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { taskFinalizationRequestSchema } from "./file-task-finalization-store.js";
import {
  type LocalTaskFinalizationOperations,
  inspectFinalizationRequestSchema,
  revalidateFinalizationRequestSchema,
} from "./local-task-finalization.js";
import { workflowToolResult } from "./task-workflow-tools.js";

export function registerHostTaskFinalizationTools(
  server: McpServer,
  operations: LocalTaskFinalizationOperations,
  authorize: (token: string) => void,
): void {
  const hostToken = z.string().regex(/^[a-f0-9]{64}$/);
  server.tool(
    "kontext_revalidate_finalization",
    "Host-only: explicitly reassess the latest reviewed finalization against current code and evidence. May write assessment audit artifacts, but never changes finalization history, grants approvals or starts a model. A current result is an observation, not a permanent done flag.",
    { ...revalidateFinalizationRequestSchema.shape, hostToken },
    { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await operations.revalidate(request));
    },
  );
  server.tool(
    "kontext_finalize_task",
    "Host-only: record the user's explicit finalization of an exact reviewed completion basis after revalidating every existing completion requirement. Never creates owner approval or changes project code. Exact UUID replay returns historical evidence without claiming it is still current.",
    { ...taskFinalizationRequestSchema.shape, hostToken },
    { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await operations.finalize(request));
    },
  );
  server.tool(
    "kontext_inspect_finalization",
    "Host-only: read a Task's durable finalization record without revalidating code, changing state, or starting a model. A historical record is not proof of current completion.",
    { ...inspectFinalizationRequestSchema.shape, hostToken },
    { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await operations.inspect(request));
    },
  );
}
