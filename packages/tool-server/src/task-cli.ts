import { existsSync } from "node:fs";
import path from "node:path";
import { TaskContextWorkflow } from "@kontext-brain/context";
import {
  FileQuarantineStore,
  FileTaskCompletionArtifactStore,
  FileTaskContextRepository,
  FileVerificationRetryQueue,
  registerWorkspaceCommandVerifiers,
  resolvePluginDataDirectory,
} from "@kontext-brain/local";
import {
  DurableVerificationCoordinator,
  VerificationCoordinator,
  VerifierRegistry,
} from "@kontext-brain/orchestrator";
import { ClaudeCodeRuntimeAdapter } from "@kontext-brain/runtime-claude";
import { CodexRuntimeAdapter, type RuntimeMcpServer } from "@kontext-brain/runtime-codex";
import { FileIntegratedTaskStateStore } from "./file-integrated-task-state-store.js";
import { FileWriteAuthorizationBindingStore } from "./file-write-authorization-binding-store.js";
import { FileWriteAuthorizationEventStore } from "./file-write-authorization-event-store.js";
import { takeHostKnowledgeCapability } from "./host-knowledge-tools.js";
import { LocalKontextCompletionOperations } from "./local-completion-operations.js";
import { LocalKnowledgeOperations } from "./local-knowledge-operations.js";
import { LocalPostWriteObserver } from "./local-post-write-observer.js";
import { LocalRegisteredIntegrationOperations } from "./local-registered-integration.js";
import { LocalRegisteredScheduleOperations } from "./local-registered-schedules.js";
import { LocalTaskCompletionAssessment } from "./local-task-completion-assessment.js";
import { LocalTaskCreationOperations } from "./local-task-creation.js";
import { LocalTaskFinalizationOperations } from "./local-task-finalization.js";
import { LocalTaskInventoryOperations } from "./local-task-inventory.js";
import { LocalTaskPlanningOperations } from "./local-task-planning.js";
import { LocalVerificationRecoveryService } from "./local-verification-recovery.js";
import { LocalWorkspaceObservationService } from "./local-workspace-observation-service.js";
import { RegisteredTaskContextProvider } from "./registered-task-context.js";
import { RegisteredTaskContextWorkflow } from "./registered-task-workflow.js";
import { LocalKontextRuntimeOperations } from "./runtime-workflow-tools.js";
import { BoundWorkspaceChangeEvidenceProvider } from "./sidecar-change-evidence.js";
import { subscriptionRuntimeEnvironment } from "./subscription-runtime-environment.js";
import { KontextTaskToolServer } from "./task-tool-server.js";
import {
  KontextTaskWorkflowToolRouter,
  type WriteAuthorizationResult,
  extractWritePaths,
} from "./task-workflow-tools.js";

async function main(): Promise<void> {
  // Verifiers and Git subprocesses must never inherit host-management authority.
  const hostToken = takeHostKnowledgeCapability(process.env);
  const dataDirectory = resolvePluginDataDirectory();
  const repository = new FileTaskContextRepository(dataDirectory);
  const currentState = new RegisteredTaskContextProvider(dataDirectory, repository);
  const workflow = new RegisteredTaskContextWorkflow(dataDirectory, repository, currentState);
  const bindings = new FileWriteAuthorizationBindingStore(dataDirectory);
  const authorizationEvents = new FileWriteAuthorizationEventStore(dataDirectory);
  if (process.argv.includes("--authorize-write-hook")) {
    await authorizeWriteHook(workflow, bindings, authorizationEvents);
    return;
  }
  const quarantine = new FileQuarantineStore(dataDirectory);
  const postWriteObserver = new LocalPostWriteObserver(
    currentState,
    repository,
    bindings,
    authorizationEvents,
    quarantine,
  );
  if (process.argv.includes("--observe-write-hook")) {
    await observeWriteHook(postWriteObserver);
    return;
  }
  const artifacts = new FileTaskCompletionArtifactStore(dataDirectory);
  const integratedTasks = new FileIntegratedTaskStateStore(dataDirectory);
  const retryQueue = new FileVerificationRetryQueue(dataDirectory);
  const verifierRegistry = new VerifierRegistry();
  registerWorkspaceCommandVerifiers(verifierRegistry);
  const durableVerification = new DurableVerificationCoordinator(
    new VerificationCoordinator(verifierRegistry),
    retryQueue,
  );
  const changeEvidence = new BoundWorkspaceChangeEvidenceProvider(bindings);
  const completion = new LocalKontextCompletionOperations(
    currentState,
    repository,
    artifacts,
    quarantine,
    durableVerification,
    changeEvidence,
    integratedTasks,
  );
  const runtimeOperations = new LocalKontextRuntimeOperations(
    currentState,
    repository,
    workflow,
    bindings,
    dataDirectory,
    [
      new CodexRuntimeAdapter({
        environment: subscriptionRuntimeEnvironment(dataDirectory),
        mcpServer: workerToolServer(dataDirectory),
      }),
      new ClaudeCodeRuntimeAdapter({
        pluginPath: currentPluginRoot(),
        environment: subscriptionRuntimeEnvironment(dataDirectory),
      }),
    ],
    artifacts,
    quarantine,
    durableVerification,
    changeEvidence,
    integratedTasks,
  );
  const completionAssessment = new LocalTaskCompletionAssessment(dataDirectory, completion);
  new LocalVerificationRecoveryService(
    currentState,
    repository,
    artifacts,
    retryQueue,
    durableVerification,
  ).start(30_000, (error) => {
    process.stderr.write(
      `kontext verification recovery error: ${error instanceof Error ? error.message : String(error)}\n`,
    );
  });
  new LocalWorkspaceObservationService(bindings, postWriteObserver).start(2_000, (error) => {
    process.stderr.write(
      `kontext workspace observation error: ${error instanceof Error ? error.message : String(error)}\n`,
    );
  });
  process.stderr.write(`kontext-brain private data: ${dataDirectory}\n`);
  await new KontextTaskToolServer(
    workflow,
    bindings,
    completion,
    runtimeOperations,
    hostToken
      ? {
          token: hostToken,
          operations: new LocalKnowledgeOperations(dataDirectory),
          taskCreation: new LocalTaskCreationOperations(dataDirectory),
          taskCompletion: completionAssessment,
          taskInventory: new LocalTaskInventoryOperations(dataDirectory),
          registeredSchedules: new LocalRegisteredScheduleOperations(
            dataDirectory,
            runtimeOperations,
          ),
          registeredIntegration: new LocalRegisteredIntegrationOperations(
            dataDirectory,
            new LocalRegisteredScheduleOperations(dataDirectory, runtimeOperations),
            runtimeOperations,
          ),
          taskFinalization: new LocalTaskFinalizationOperations(
            dataDirectory,
            completionAssessment,
          ),
          taskPlanning: new LocalTaskPlanningOperations(dataDirectory, [
            new CodexRuntimeAdapter({
              environment: subscriptionRuntimeEnvironment(dataDirectory),
              timeoutMilliseconds: 15 * 60_000,
            }),
            new ClaudeCodeRuntimeAdapter({
              environment: subscriptionRuntimeEnvironment(dataDirectory),
              timeoutMilliseconds: 15 * 60_000,
            }),
          ]),
        }
      : undefined,
  ).start();
}

async function authorizeWriteHook(
  workflow: TaskContextWorkflow,
  bindings: FileWriteAuthorizationBindingStore,
  authorizationEvents: FileWriteAuthorizationEventStore,
): Promise<void> {
  try {
    const event = JSON.parse(await readStandardInput()) as Record<string, unknown>;
    const router = new KontextTaskWorkflowToolRouter(workflow, () => new Date(), bindings);
    const decision = await router.authorizeWrite({
      cwd: event.cwd,
      toolName: event.tool_name,
      toolInput: event.tool_input,
    });
    if (
      decision.hookSpecificOutput.permissionDecision === "allow" &&
      typeof event.tool_use_id === "string" &&
      typeof event.cwd === "string" &&
      isRecord(event.tool_input) &&
      isWriteToolName(event.tool_name)
    ) {
      const workspacePath = path.resolve(event.cwd);
      const binding = await bindings.get(workspacePath);
      if (!binding) throw new Error("Authorized write lost its persisted Kontext binding");
      const authorizedPaths = extractWritePaths(event.tool_name, event.tool_input)
        .map((filePath) => path.resolve(workspacePath, filePath))
        .map((filePath) => path.relative(workspacePath, filePath).replaceAll("\\", "/"))
        .sort((left, right) => left.localeCompare(right));
      await authorizationEvents.put({
        toolUseId: event.tool_use_id,
        workspacePath,
        taskId: binding.request.taskId,
        workItemId: binding.request.logic.workItemId,
        receiptId: binding.receipt.receiptId,
        contextDigest: binding.receipt.contextDigest,
        baselineRevision: binding.baseline.revision,
        authorizedPaths,
        authorizedAt: new Date().toISOString(),
      });
    }
    process.stdout.write(`${JSON.stringify(decision)}\n`);
  } catch (error) {
    const decision: WriteAuthorizationResult = {
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: `Kontext write guard failed closed: ${
          error instanceof Error ? error.message : String(error)
        }`,
      },
    };
    process.stdout.write(`${JSON.stringify(decision)}\n`);
  }
}

async function observeWriteHook(observer: LocalPostWriteObserver): Promise<void> {
  try {
    const event = JSON.parse(await readStandardInput()) as Record<string, unknown>;
    const observation = await observer.observe({
      cwd: typeof event.cwd === "string" ? event.cwd : "",
      toolName: typeof event.tool_name === "string" ? event.tool_name : "",
      toolUseId: typeof event.tool_use_id === "string" ? event.tool_use_id : undefined,
      toolInput: event.tool_input,
      observedAt: new Date().toISOString(),
    });
    const quarantineId = observation.assessment.record?.quarantineId;
    const additionalContext = observation.changed
      ? `Kontext observed ${observation.changedPaths.length} changed path(s) at ${observation.codeRevision}.`
      : "Kontext observed no workspace content change.";
    const output = quarantineId
      ? {
          decision: "block",
          reason: `Write was quarantined as ${quarantineId}: ${observation.assessment.record?.reasons.join(", ")}`,
          hookSpecificOutput: {
            hookEventName: "PostToolUse",
            additionalContext,
          },
        }
      : {
          hookSpecificOutput: {
            hookEventName: "PostToolUse",
            additionalContext,
          },
        };
    process.stdout.write(`${JSON.stringify(output)}\n`);
  } catch (error) {
    process.stdout.write(
      `${JSON.stringify({
        decision: "block",
        reason: `Kontext post-write observation failed closed: ${error instanceof Error ? error.message : String(error)}`,
        hookSpecificOutput: {
          hookEventName: "PostToolUse",
          additionalContext: "The completed write could not be reconciled with Kontext state.",
        },
      })}\n`,
    );
  }
}

async function readStandardInput(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString("utf8");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isWriteToolName(value: unknown): value is "apply_patch" | "Write" | "Edit" {
  return value === "apply_patch" || value === "Write" || value === "Edit";
}

/**
 * A worker must reach the same task tool server this process is, over the same
 * data directory, so its Context Receipt and Change Bundle land where the
 * scheduler reads them. Claude gets that through the plugin directory; Codex
 * only loads servers from the user's config, so the worker is handed this entry.
 */
function workerToolServer(dataDirectory: string): RuntimeMcpServer | undefined {
  const entry = process.argv[1] ? path.resolve(process.argv[1]) : undefined;
  if (!entry) return undefined;
  return {
    name: "kontext_brain",
    command: process.execPath,
    args: [entry],
    // Why: under Electron the executable is Electron itself; this flag makes it Node
    // and is ignored by a real Node binary. No host-management token travels here.
    env: { KONTEXT_PLUGIN_DATA: dataDirectory, ELECTRON_RUN_AS_NODE: "1" },
    startupTimeoutSeconds: 30,
  };
}

function currentPluginRoot(): string | undefined {
  const candidate =
    process.env.KONTEXT_PLUGIN_ROOT ??
    (process.argv[1] ? path.dirname(path.resolve(process.argv[1])) : undefined);
  return candidate && existsSync(path.join(candidate, ".codex-plugin", "plugin.json"))
    ? candidate
    : undefined;
}

main().catch((error) => {
  process.stderr.write(
    `kontext-task-tool-server error: ${error instanceof Error ? error.message : String(error)}\n`,
  );
  process.exit(1);
});
