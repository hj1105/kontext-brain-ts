import type { TaskContextWorkflow } from "@kontext-brain/context";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  type KontextCompletionOperations,
  KontextCompletionToolRouter,
  checkChangeToolShape,
  proposeTransitionToolShape,
  submitChangeBundleToolShape,
} from "./completion-workflow-tools.js";
import {
  cancelScheduleToolShape,
  getScheduleToolShape,
  inspectRuntimesToolShape,
  integrateScheduleToolShape,
  scheduleLogicToolShape,
} from "./runtime-schedule-contract.js";
import {
  type KontextRuntimeOperations,
  KontextRuntimeToolRouter,
} from "./runtime-workflow-tools.js";
import {
  KontextTaskWorkflowToolRouter,
  type WriteAuthorizationBindingStore,
  authorizeWriteToolShape,
  beginLogicToolShape,
  prepareTaskToolShape,
  refreshTaskContextToolShape,
  workflowToolResult,
} from "./task-workflow-tools.js";

export function registerTaskWorkflowTools(
  server: McpServer,
  workflow: Pick<TaskContextWorkflow, "prepareTask" | "beginLogic" | "refreshTaskContext">,
  bindings?: WriteAuthorizationBindingStore,
  completionOperations?: KontextCompletionOperations,
  runtimeOperations?: KontextRuntimeOperations,
): void {
  const router = new KontextTaskWorkflowToolRouter(workflow, () => new Date(), bindings);
  server.tool(
    "kontext_prepare_task",
    "Validate a Task Contract against sidecar-owned current state and freeze its Task Context Snapshot.",
    prepareTaskToolShape,
    async (input) => workflowToolResult(await router.prepareTask(input)),
  );
  server.tool(
    "kontext_begin_logic",
    "Compile mandatory symbol context and issue a Context Receipt before editing one Logic Work Item.",
    beginLogicToolShape,
    async (input) => workflowToolResult(await router.beginLogic(input)),
  );
  server.tool(
    "kontext_authorize_write",
    "Hook-only guard for Codex apply_patch and Claude Write/Edit. Revalidates context and exact sidecar-owned file paths.",
    authorizeWriteToolShape,
    async (input) => workflowToolResult(await router.authorizeWrite(input)),
  );
  server.tool(
    "kontext_refresh_task_context",
    "Show the normative and Evidence revision diff, then replace a stale Task Context Snapshot.",
    refreshTaskContextToolShape,
    async (input) => workflowToolResult(await router.refreshTaskContext(input)),
  );
  if (completionOperations) {
    const completion = new KontextCompletionToolRouter(completionOperations);
    server.tool(
      "kontext_check_change",
      "Resynchronize affected symbols and execute the required verification tier for the exact revision and context digest.",
      checkChangeToolShape,
      async (input) => workflowToolResult(await completion.checkChange(input)),
    );
    server.tool(
      "kontext_submit_change_bundle",
      "Validate an immutable worker handoff against observed code, Context Receipts, current proof, and quarantine.",
      submitChangeBundleToolShape,
      async (input) => workflowToolResult(await completion.submitChangeBundle(input)),
    );
    server.tool(
      "kontext_propose_transition",
      "Compute Task state from sidecar-owned proof. Direct state writes and caller-supplied Accuracy Manifests are not accepted.",
      proposeTransitionToolShape,
      async (input) => workflowToolResult(await completion.proposeTransition(input)),
    );
  }
  if (runtimeOperations) {
    const runtime = new KontextRuntimeToolRouter(runtimeOperations);
    server.tool(
      "kontext_inspect_runtimes",
      "Inspect installed Codex and Claude CLIs, authentication, billing path, and frozen runtime capabilities.",
      inspectRuntimesToolShape,
      async (input) => workflowToolResult(await runtime.inspectRuntimes(input)),
    );
    server.tool(
      "kontext_schedule_logic",
      "Queue sidecar-planned Logic Work Items for asynchronous execution in isolated worktrees with provider-bound Context Receipts, leases, checkpoints, and bounded retries.",
      scheduleLogicToolShape,
      async (input) => workflowToolResult(await runtime.scheduleLogic(input)),
    );
    server.tool(
      "kontext_get_schedule",
      "Read queued, running, cancelling, completed, failed, interrupted, or cancelled durable schedule state and terminal result.",
      getScheduleToolShape,
      async (input) => workflowToolResult(await runtime.getSchedule(input)),
    );
    server.tool(
      "kontext_cancel_schedule",
      "Request cancellation, terminate active runtime workers, release their leases, and report durable schedule state.",
      cancelScheduleToolShape,
      async (input) => workflowToolResult(await runtime.cancelSchedule(input)),
    );
    server.tool(
      "kontext_integrate_schedule",
      "Semantically validate accepted Change Bundles, apply them in dependency order to an isolated integration worktree, run same-revision full verification, and obtain risk-based independent review.",
      integrateScheduleToolShape,
      async (input) => workflowToolResult(await runtime.integrateSchedule(input)),
    );
  }
}

/** A task-only MCP server used by the Codex plugin and local sidecar. */
export class KontextTaskToolServer {
  private readonly server = new McpServer({ name: "kontext-brain-task", version: "0.1.0" });

  constructor(
    workflow: TaskContextWorkflow,
    bindings?: WriteAuthorizationBindingStore,
    completionOperations?: KontextCompletionOperations,
    runtimeOperations?: KontextRuntimeOperations,
  ) {
    registerTaskWorkflowTools(
      this.server,
      workflow,
      bindings,
      completionOperations,
      runtimeOperations,
    );
  }

  async start(): Promise<void> {
    process.stderr.write("kontext-brain task MCP server starting (stdio mode)\n");
    await this.server.connect(new StdioServerTransport());
  }
}
