import { mkdtemp, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type {
  BeginLogicRequest,
  CompiledTaskContext,
  PrepareTaskRequest,
  PreparedTaskContext,
  RefreshTaskContextRequest,
  TaskContextRefreshResult,
} from "@kontext-brain/context";
import { afterEach, describe, expect, it } from "vitest";
import {
  FileWriteAuthorizationBindingStore,
  type KontextTaskWorkflowOperations,
  KontextTaskWorkflowToolRouter,
} from "../src/index.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("FileWriteAuthorizationBindingStore", () => {
  it("lets a separate command-hook router revalidate an exact persisted path", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "kontext-write-capability-"));
    temporaryDirectories.push(directory);
    const workspace = path.join(directory, "workspace");
    const bindings = new FileWriteAuthorizationBindingStore(directory);
    const workflow = new CurrentWorkflow();
    const mcpRouter = new KontextTaskWorkflowToolRouter(
      workflow,
      () => new Date("2026-08-28T00:00:00.000Z"),
      bindings,
    );
    await mcpRouter.beginLogic({
      taskId: "task:persisted-capability",
      workspacePath: workspace,
      logic: {
        workItemId: "work-item:handler",
        plannedSymbolIds: ["planned-symbol:handler"],
      },
      runtimeProvider: "codex",
      receiptTtlSeconds: 600,
      totalTokenBudget: 10_000,
      optionalEvidenceTokenBudget: 1_000,
    });

    const hookRouter = new KontextTaskWorkflowToolRouter(
      workflow,
      () => new Date("2026-08-28T00:01:00.000Z"),
      new FileWriteAuthorizationBindingStore(directory),
    );
    const decision = await hookRouter.authorizeWrite({
      cwd: workspace,
      toolName: "apply_patch",
      toolInput: {
        command: "*** Begin Patch\n*** Update File: src/handler.ts\n*** End Patch",
      },
    });

    expect(decision.hookSpecificOutput.permissionDecision).toBe("allow");
    expect((await stat(bindings.filePath(workspace))).mode & 0o777).toBe(0o600);
  });

  it("does not let a stale observer overwrite a newer Logic Work Item binding", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "kontext-write-capability-race-"));
    temporaryDirectories.push(directory);
    const workspace = path.join(directory, "workspace");
    const bindings = new FileWriteAuthorizationBindingStore(directory);
    const workflow = new CurrentWorkflow();
    const firstRouter = new KontextTaskWorkflowToolRouter(
      workflow,
      () => new Date("2026-08-28T00:00:00.000Z"),
      bindings,
    );
    await firstRouter.beginLogic({
      taskId: "task:persisted-capability",
      workspacePath: workspace,
      logic: {
        workItemId: "work-item:first",
        plannedSymbolIds: ["planned-symbol:first"],
      },
      runtimeProvider: "codex",
      receiptTtlSeconds: 600,
      totalTokenBudget: 10_000,
      optionalEvidenceTokenBudget: 1_000,
    });
    const stale = await bindings.get(workspace);
    if (!stale) throw new Error("expected first binding");

    const secondRouter = new KontextTaskWorkflowToolRouter(
      workflow,
      () => new Date("2026-08-28T00:01:00.000Z"),
      bindings,
    );
    await secondRouter.beginLogic({
      taskId: "task:persisted-capability",
      workspacePath: workspace,
      logic: {
        workItemId: "work-item:second",
        plannedSymbolIds: ["planned-symbol:second"],
      },
      runtimeProvider: "codex",
      receiptTtlSeconds: 600,
      totalTokenBudget: 10_000,
      optionalEvidenceTokenBudget: 1_000,
    });

    expect(await bindings.putIfUnchanged(workspace, stale, stale)).toBe(false);
    expect((await bindings.get(workspace))?.request.logic.workItemId).toBe("work-item:second");
  });
});

class CurrentWorkflow implements KontextTaskWorkflowOperations {
  async prepareTask(request: PrepareTaskRequest): Promise<PreparedTaskContext> {
    return { contract: request.contract } as unknown as PreparedTaskContext;
  }

  async beginLogic(request: BeginLogicRequest): Promise<CompiledTaskContext> {
    return {
      status: "current",
      editingAllowed: true,
      contextDigest: "sha256:context",
      receipt: {
        receiptId: "context-receipt:persisted",
        taskId: request.taskId,
        workItemId: request.logic.workItemId,
        plannedSymbolIds: request.logic.plannedSymbolIds,
        allowedPaths: ["src/handler.ts"],
        contextDigest: "sha256:context",
        normativeRevisions: [],
        evidenceIds: [],
        issuedAt: request.issuedAt,
        expiresAt: request.expiresAt,
      },
    } as unknown as CompiledTaskContext;
  }

  async refreshTaskContext(_request: RefreshTaskContextRequest): Promise<TaskContextRefreshResult> {
    return { changed: false } as unknown as TaskContextRefreshResult;
  }
}
