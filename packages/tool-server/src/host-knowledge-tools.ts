import { timingSafeEqual } from "node:crypto";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { registerHostRegisteredIntegrationTools } from "./host-registered-integration-tools.js";
import { registerHostRegisteredScheduleTools } from "./host-registered-schedule-tools.js";
import { registerHostTaskFinalizationTools } from "./host-task-finalization-tools.js";
import { registerHostTaskPlanningTools } from "./host-task-planning-tools.js";
import type { LocalKnowledgeOperations } from "./local-knowledge-operations.js";
import type { LocalRegisteredIntegrationOperations } from "./local-registered-integration.js";
import type { LocalRegisteredScheduleOperations } from "./local-registered-schedules.js";
import {
  sourceInventoryRequestSchema,
  sourceResourceIdSchema,
  sourceSharingRequestSchema,
} from "./local-source-registry.js";
import {
  type LocalTaskCompletionAssessment,
  taskCompletionAssessmentRequestSchema,
} from "./local-task-completion-assessment.js";
import {
  type LocalTaskCreationOperations,
  localTaskCreationRequestSchema,
} from "./local-task-creation.js";
import type { LocalTaskFinalizationOperations } from "./local-task-finalization.js";
import {
  type LocalTaskInventoryOperations,
  taskInventoryRequestSchema,
} from "./local-task-inventory.js";
import type { LocalTaskPlanningOperations } from "./local-task-planning.js";
import { nativeSessionRegistrationSchema } from "./native-session-source-reader.js";
import { workflowToolResult } from "./task-workflow-tools.js";

export interface HostKnowledgeOperations {
  readonly token: string;
  readonly operations: Pick<
    LocalKnowledgeOperations,
    | "registerMarkdownSource"
    | "registerSessionSource"
    | "inspectSource"
    | "refreshSource"
    | "setSourceSharing"
    | "listSources"
  >;
  readonly taskCreation?: Pick<LocalTaskCreationOperations, "createTask">;
  readonly taskPlanning?: LocalTaskPlanningOperations;
  readonly taskCompletion?: LocalTaskCompletionAssessment;
  readonly taskFinalization?: LocalTaskFinalizationOperations;
  readonly taskInventory?: LocalTaskInventoryOperations;
  readonly registeredSchedules?: LocalRegisteredScheduleOperations;
  readonly registeredIntegration?: LocalRegisteredIntegrationOperations;
}
export function takeHostKnowledgeCapability(environment: NodeJS.ProcessEnv): string | undefined {
  const token = environment.KONTEXT_HOST_MANAGEMENT_TOKEN;
  if (
    !Reflect.deleteProperty(environment, "KONTEXT_HOST_MANAGEMENT_TOKEN") ||
    environment.KONTEXT_HOST_MANAGEMENT_TOKEN !== undefined
  )
    throw new Error("Could not remove host-management capability from the environment");
  if (token !== undefined && !/^[a-f0-9]{64}$/.test(token))
    throw new Error("Invalid host-management capability");
  return token;
}
export function registerHostKnowledgeTools(server: McpServer, host: HostKnowledgeOperations): void {
  if (!/^[a-f0-9]{64}$/.test(host.token)) throw new Error("Invalid host-management capability");
  const authorize = (token: string) => {
    if (!timingSafeEqual(Buffer.from(token), Buffer.from(host.token)))
      throw new Error("Host-management capability required");
  };
  if (host.taskPlanning) registerHostTaskPlanningTools(server, host.taskPlanning, authorize);
  if (host.registeredSchedules)
    registerHostRegisteredScheduleTools(server, host.registeredSchedules, authorize);
  if (host.registeredIntegration)
    registerHostRegisteredIntegrationTools(server, host.registeredIntegration, authorize);
  if (host.taskFinalization)
    registerHostTaskFinalizationTools(server, host.taskFinalization, authorize);
  if (host.taskInventory) {
    const inventory = host.taskInventory;
    server.tool(
      "kontext_list_tasks",
      "Host-only: list the current principal's registered Task metadata, saved schedule status, integration and historical completion records. Never resumes execution, recaptures sources, assesses completion or marks Tasks done. Cursor changes require a first-page reload.",
      { ...taskInventoryRequestSchema.shape, hostToken: z.string().regex(/^[a-f0-9]{64}$/) },
      { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
      async ({ hostToken, ...request }) => {
        authorize(hostToken);
        return workflowToolResult(await inventory.list(request));
      },
    );
  }
  server.tool(
    "kontext_list_sources",
    "Host-only: list the current principal's registered source metadata and saved model permissions. No source bodies, live file recapture, model dispatch or approval. Pagination refuses changed inventories; reload from the first page.",
    { ...sourceInventoryRequestSchema.shape, hostToken: z.string().regex(/^[a-f0-9]{64}$/) },
    { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await host.operations.listSources(request));
    },
  );
  if (host.taskCompletion) {
    const completion = host.taskCompletion;
    server.tool(
      "kontext_assess_completion",
      "Host-only: assess completion for an exact registered Task and integrated schedule. Recaptures current sources, verifies the actual clean integrated commit and derives invariant evidence from persisted runs. Writes manifest audit artifacts, but never executes verifiers or models, grants approval, or publishes code. The returned verdict is an observation, not an immutable Task state.",
      {
        ...taskCompletionAssessmentRequestSchema.shape,
        hostToken: z.string().regex(/^[a-f0-9]{64}$/),
      },
      { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
      async ({ hostToken, ...request }) => {
        authorize(hostToken);
        return workflowToolResult(await completion.assess(request));
      },
    );
  }
  if (host.taskCreation) {
    const taskCreation = host.taskCreation;
    server.tool(
      "kontext_create_task",
      "Host-only: finalize a user-reviewed personal-context Task plan against an exact clean Git revision. Recaptures selected sources, loads trusted local normative rules, and atomically registers the Task. Does not plan with a model, grant sharing, start workers, or mark work complete.",
      { ...localTaskCreationRequestSchema.shape, hostToken: z.string().regex(/^[a-f0-9]{64}$/) },
      { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
      async ({ hostToken, ...request }) => {
        authorize(hostToken);
        return workflowToolResult(await taskCreation.createTask(request));
      },
    );
  }
  server.tool(
    "kontext_register_session_source",
    "Host-only: recapture a user-reviewed native session from its owning app and register raw evidence. Requires its exact reviewed digest; never grants provider sharing or approves normative decisions.",
    { ...nativeSessionRegistrationSchema.shape, hostToken: z.string().regex(/^[a-f0-9]{64}$/) },
    { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await host.operations.registerSessionSource(request));
    },
  );
  server.tool(
    "kontext_register_markdown_source",
    "Host-only: synchronize a user-selected Markdown file into the private local knowledge graph. Does not grant provider sharing or approve normative decisions.",
    {
      hostToken: z.string().regex(/^[a-f0-9]{64}$/),
      workspacePath: z.string().min(1),
      relativePath: z.string().min(1),
    },
    { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await host.operations.registerMarkdownSource(request));
    },
  );
  server.tool(
    "kontext_inspect_source",
    "Host-only: inspect a registered source locator, captured revision and explicit provider-sharing configuration. Returns no source body.",
    { hostToken: z.string().regex(/^[a-f0-9]{64}$/), resourceId: sourceResourceIdSchema },
    { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await host.operations.inspectSource(request));
    },
  );
  server.tool(
    "kontext_refresh_source",
    "Host-only: recapture a registered source by Resource ID; changed content clears previous provider grants. Unavailable origins are stale, never a current cached copy. Does not grant normative approval.",
    { hostToken: z.string().regex(/^[a-f0-9]{64}$/), resourceId: sourceResourceIdSchema },
    { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await host.operations.refreshSource(request));
    },
  );
  server.tool(
    "kontext_set_source_sharing",
    "Host-only: apply the user's explicit provider-sharing choice to an exact inspected source revision. Empty providers revoke sharing. Never infers consent from source content or approves normative decisions.",
    { ...sourceSharingRequestSchema.shape, hostToken: z.string().regex(/^[a-f0-9]{64}$/) },
    { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: false },
    async ({ hostToken, ...request }) => {
      authorize(hostToken);
      return workflowToolResult(await host.operations.setSourceSharing(request));
    },
  );
}
