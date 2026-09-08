import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import {
  type LocalRegisteredIntegrationOperations,
  registeredIntegrationRequestSchema,
} from "./local-registered-integration.js";
import { registeredScheduleSelectorSchema } from "./local-registered-schedules.js";
import { workflowToolResult } from "./task-workflow-tools.js";

export function registerHostRegisteredIntegrationTools(
  server: McpServer,
  operations: LocalRegisteredIntegrationOperations,
  authorize: (token: string) => void,
) {
  const hostToken = z.string().regex(/^[a-f0-9]{64}$/);
  server.tool(
    "kontext_inspect_registered_integration",
    "Host-only: read saved integration metadata for an owned Task and selected execution. May show a different execution's latest integration. Never runs Git, verification, review or model workers; not current completion proof.",
    { ...registeredScheduleSelectorSchema.shape, hostToken },
    { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await operations.inspect(request));
    },
  );
  server.tool(
    "kontext_integrate_registered_schedule",
    "Host-only: explicitly integrate the reviewed completed execution using existing Change Bundle, context, verification and independent-review gates. May create a Git integration worktree/commit, run verification commands and subscription review agents, and replace this Task's latest integration record. Requires reviewed job/integration identities and explicit subscription consent. Lost responses must be inspected, never automatically replayed.",
    { ...registeredIntegrationRequestSchema.shape, hostToken },
    { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: true },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await operations.integrate(request));
    },
  );
}
