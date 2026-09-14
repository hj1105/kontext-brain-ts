import path from "node:path";
import type {
  BeginLogicRequest,
  CompiledTaskContext,
  PrepareTaskRequest,
  PreparedTaskContext,
  RefreshTaskContextRequest,
  TaskContextRefreshResult,
} from "@kontext-brain/context";
import type { ContextReceipt } from "@kontext-brain/spec";
import { z } from "zod";
import {
  type WorkspaceObservationSnapshot,
  captureWorkspaceSnapshot,
} from "./workspace-change-observer.js";
import {
  type WorkspaceCodeSymbolSnapshot,
  captureWorkspaceCodeSymbols,
} from "./workspace-code-symbol-observer.js";

const verifierSchema = z
  .object({
    kind: z.enum(["test", "typecheck", "build", "lint", "query", "manual_review"]),
    ref: z.string().min(1),
  })
  .strict();

const taskContractSchema = z
  .object({
    taskId: z.string().min(1),
    intent: z.string().min(1),
    acceptance: z
      .array(
        z
          .object({
            criterionId: z.string().min(1),
            statement: z.string().min(1),
            verifier: verifierSchema,
          })
          .strict(),
      )
      .min(1),
    nonGoals: z.array(z.string()),
    targets: z.array(z.string().min(1)).min(1),
    risk: z.enum(["low", "medium", "high"]),
  })
  .strict();

export const prepareTaskToolShape = {
  contract: taskContractSchema,
  additionalRequiredEvidenceIds: z.array(z.string().min(1)).optional(),
  createdAt: z.string().datetime(),
};

export const beginLogicToolShape = {
  taskId: z.string().min(1),
  workspacePath: z.string().min(1),
  logic: z
    .object({
      workItemId: z.string().min(1),
      plannedSymbolIds: z.array(z.string().min(1)).min(1),
    })
    .strict(),
  runtimeProvider: z.string().min(1),
  receiptTtlSeconds: z.number().int().min(60).max(3600).optional(),
  totalTokenBudget: z.number().int().positive(),
  optionalEvidenceTokenBudget: z.number().int().nonnegative(),
};

export const refreshTaskContextToolShape = {
  taskId: z.string().min(1),
  createdAt: z.string().datetime(),
};

const writeToolNameSchema = z.enum(["apply_patch", "Write", "Edit"]);

export const authorizeWriteToolShape = {
  cwd: z.string().min(1),
  toolName: writeToolNameSchema,
  toolInput: z
    .object({
      command: z.string().min(1).optional(),
      file_path: z.string().min(1).optional(),
    })
    .passthrough(),
};

export interface WriteAuthorizationResult {
  readonly hookSpecificOutput: {
    readonly hookEventName: "PreToolUse";
    readonly permissionDecision: "allow" | "deny";
    readonly permissionDecisionReason: string;
  };
}

export interface KontextTaskWorkflowOperations {
  prepareTask(request: PrepareTaskRequest): Promise<PreparedTaskContext>;
  beginLogic(request: BeginLogicRequest): Promise<CompiledTaskContext>;
  refreshTaskContext(request: RefreshTaskContextRequest): Promise<TaskContextRefreshResult>;
}

export interface WriteAuthorizationBinding {
  readonly request: BeginLogicRequest;
  readonly allowedPaths: readonly string[];
  readonly receipt: ContextReceipt;
  readonly initialBaseline: WorkspaceObservationSnapshot;
  readonly baseline: WorkspaceObservationSnapshot;
  readonly symbolBaseline: WorkspaceCodeSymbolSnapshot;
}

export interface WriteAuthorizationBindingStore {
  get(workspacePath: string): Promise<WriteAuthorizationBinding | undefined>;
  list(): Promise<
    readonly {
      readonly workspacePath: string;
      readonly binding: WriteAuthorizationBinding;
    }[]
  >;
  put(workspacePath: string, binding: WriteAuthorizationBinding): Promise<void>;
  putIfUnchanged(
    workspacePath: string,
    expected: WriteAuthorizationBinding,
    binding: WriteAuthorizationBinding,
  ): Promise<boolean>;
  delete(workspacePath: string): Promise<void>;
}

export class InMemoryWriteAuthorizationBindingStore implements WriteAuthorizationBindingStore {
  private readonly bindings = new Map<string, WriteAuthorizationBinding>();

  async get(workspacePath: string): Promise<WriteAuthorizationBinding | undefined> {
    return this.bindings.get(workspacePath);
  }

  async list() {
    return Array.from(this.bindings.entries())
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([workspacePath, binding]) => ({ workspacePath, binding }));
  }

  async put(workspacePath: string, binding: WriteAuthorizationBinding): Promise<void> {
    this.bindings.set(workspacePath, binding);
  }

  async putIfUnchanged(
    workspacePath: string,
    expected: WriteAuthorizationBinding,
    binding: WriteAuthorizationBinding,
  ): Promise<boolean> {
    if (!sameBindingGeneration(this.bindings.get(workspacePath), expected)) return false;
    this.bindings.set(workspacePath, binding);
    return true;
  }

  async delete(workspacePath: string): Promise<void> {
    this.bindings.delete(workspacePath);
  }
}

export function sameBindingGeneration(
  current: WriteAuthorizationBinding | undefined,
  expected: WriteAuthorizationBinding,
): boolean {
  return (
    current?.request.taskId === expected.request.taskId &&
    current.request.logic.workItemId === expected.request.logic.workItemId &&
    current.request.issuedAt === expected.request.issuedAt &&
    current.receipt.receiptId === expected.receipt.receiptId &&
    current.initialBaseline.revision === expected.initialBaseline.revision
  );
}

export class KontextTaskWorkflowToolRouter {
  constructor(
    private readonly workflow: KontextTaskWorkflowOperations,
    private readonly now: () => Date = () => new Date(),
    private readonly bindings: WriteAuthorizationBindingStore = new InMemoryWriteAuthorizationBindingStore(),
  ) {}

  async prepareTask(input: unknown): Promise<PreparedTaskContext> {
    return this.workflow.prepareTask(z.object(prepareTaskToolShape).strict().parse(input));
  }

  async beginLogic(input: unknown): Promise<CompiledTaskContext> {
    const parsed = z.object(beginLogicToolShape).strict().parse(input);
    const issuedAt = this.now();
    const request: BeginLogicRequest = {
      taskId: parsed.taskId,
      logic: parsed.logic,
      runtimeProvider: parsed.runtimeProvider,
      issuedAt: issuedAt.toISOString(),
      expiresAt: new Date(
        issuedAt.getTime() + (parsed.receiptTtlSeconds ?? 900) * 1000,
      ).toISOString(),
      totalTokenBudget: parsed.totalTokenBudget,
      optionalEvidenceTokenBudget: parsed.optionalEvidenceTokenBudget,
    };
    const result = await this.workflow.beginLogic(request);
    const workspacePath = path.resolve(parsed.workspacePath);
    const allowedPaths = result.receipt
      ? normalizeReceiptPaths(workspacePath, result.receipt.allowedPaths)
      : undefined;
    if (result.receipt && allowedPaths) {
      const baseline = await captureWorkspaceSnapshot(workspacePath, allowedPaths);
      const symbolBaseline = await captureWorkspaceCodeSymbols(
        workspacePath,
        allowedPaths,
        baseline,
      );
      const confirmedBaseline = await captureWorkspaceSnapshot(workspacePath, allowedPaths);
      if (symbolBaseline.workspaceRevision !== confirmedBaseline.revision) {
        throw new Error(
          `Workspace changed while the initial Code Symbol baseline was captured (${symbolBaseline.workspaceRevision} != ${confirmedBaseline.revision})`,
        );
      }
      await this.bindings.put(workspacePath, {
        request,
        allowedPaths,
        receipt: result.receipt,
        initialBaseline: confirmedBaseline,
        baseline: confirmedBaseline,
        symbolBaseline,
      });
    } else {
      await this.bindings.delete(workspacePath);
    }
    return result;
  }

  async refreshTaskContext(input: unknown): Promise<TaskContextRefreshResult> {
    return this.workflow.refreshTaskContext(
      z.object(refreshTaskContextToolShape).strict().parse(input),
    );
  }

  async authorizeWrite(input: unknown): Promise<WriteAuthorizationResult> {
    const parsed = z.object(authorizeWriteToolShape).strict().parse(input);
    const workspacePath = path.resolve(parsed.cwd);
    const binding = await this.bindings.get(workspacePath);
    if (!binding) {
      return writeDecision(
        "deny",
        "No current Kontext Context Receipt is bound to this workspace.",
      );
    }
    const touchedPaths = extractWritePaths(parsed.toolName, parsed.toolInput).map((filePath) =>
      path.resolve(workspacePath, filePath),
    );
    if (
      touchedPaths.length === 0 ||
      touchedPaths.some(
        (filePath) =>
          !isWithin(workspacePath, filePath) || !binding.allowedPaths.includes(filePath),
      )
    ) {
      return writeDecision(
        "deny",
        "Patch targets are missing or outside the Logic Work Item's exact allowed paths.",
      );
    }

    const current = await this.workflow.beginLogic(binding.request);
    const currentAllowedPaths = current.receipt
      ? normalizeReceiptPaths(workspacePath, current.receipt.allowedPaths)
      : undefined;
    if (
      !current.editingAllowed ||
      !current.receipt ||
      !currentAllowedPaths ||
      !samePaths(currentAllowedPaths, binding.allowedPaths) ||
      Date.parse(current.receipt.expiresAt) <= this.now().getTime()
    ) {
      await this.bindings.delete(workspacePath);
      return writeDecision(
        "deny",
        `Kontext context is ${current.status}; refresh and begin this logic again.`,
      );
    }
    await this.bindings.put(workspacePath, {
      ...binding,
      receipt: current.receipt,
    });
    return writeDecision(
      "allow",
      `Authorized exact paths by Context Receipt ${current.receipt.receiptId}.`,
    );
  }
}

export function workflowToolResult(value: unknown) {
  const structuredContent = isRecord(value) ? value : { value };
  return {
    content: [
      {
        type: "text" as const,
        text: JSON.stringify(value, null, 2),
      },
    ],
    structuredContent,
  };
}

function normalizeReceiptPaths(
  workspacePath: string,
  allowedPaths: readonly string[],
): readonly string[] | undefined {
  const normalized = allowedPaths.map((allowedPath) => path.resolve(workspacePath, allowedPath));
  return normalized.length > 0 &&
    normalized.every((allowedPath) => isWithin(workspacePath, allowedPath))
    ? Array.from(new Set(normalized)).sort((left, right) => left.localeCompare(right))
    : undefined;
}

export function extractPatchPaths(command: string): readonly string[] {
  const paths: string[] = [];
  for (const line of command.split(/\r?\n/)) {
    const match = /^\*\*\* (?:Add|Update|Delete) File: (.+)$/.exec(line);
    const move = /^\*\*\* Move to: (.+)$/.exec(line);
    const filePath = match?.[1] ?? move?.[1];
    if (filePath?.trim()) paths.push(filePath.trim());
  }
  return Array.from(new Set(paths));
}

export function extractWritePaths(
  toolName: "apply_patch" | "Write" | "Edit",
  toolInput: { readonly command?: string; readonly file_path?: string },
): readonly string[] {
  if (toolName === "apply_patch") {
    return toolInput.command ? extractPatchPaths(toolInput.command) : [];
  }
  return toolInput.file_path ? [toolInput.file_path] : [];
}

function isWithin(workspacePath: string, filePath: string): boolean {
  const relative = path.relative(workspacePath, filePath);
  return relative !== "" && relative !== ".." && !relative.startsWith(`..${path.sep}`);
}

function samePaths(left: readonly string[], right: readonly string[]): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function writeDecision(
  permissionDecision: "allow" | "deny",
  permissionDecisionReason: string,
): WriteAuthorizationResult {
  return {
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision,
      permissionDecisionReason,
    },
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
