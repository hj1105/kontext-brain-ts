import { createHash } from "node:crypto";
import type { AgentRuntimeProvider, LogicWorkItem } from "@kontext-brain/spec";

export type RuntimeProvider = AgentRuntimeProvider;
export type RuntimeBillingPath = "subscription" | "api" | "unknown";

export interface RuntimeCapabilitySnapshot {
  readonly snapshotId: string;
  readonly provider: RuntimeProvider;
  readonly cliPath: string;
  readonly cliVersion?: string;
  readonly installed: boolean;
  readonly authenticated: boolean;
  readonly billingPath: RuntimeBillingPath;
  readonly model?: string;
  readonly supports: {
    readonly structuredOutput: boolean;
    readonly sessionResume: boolean;
    readonly mcp: boolean;
    readonly hooks: boolean;
    readonly workspaceSandbox: boolean;
  };
  readonly inspectedAt: string;
  readonly diagnostic?: string;
}

export type RuntimeCapabilitySnapshotInput = Omit<RuntimeCapabilitySnapshot, "snapshotId">;

export interface RuntimeCheckpoint {
  readonly checkpointId: string;
  readonly taskId: string;
  readonly workItemId: string;
  readonly provider: RuntimeProvider;
  readonly providerSessionId?: string;
  readonly workspacePath: string;
  readonly codeRevision: string;
  readonly contextDigest: string;
  readonly createdAt: string;
}

export interface RuntimeWorkInput {
  readonly taskId: string;
  readonly workItem: LogicWorkItem;
  readonly workspacePath: string;
  readonly prompt: string;
  readonly codeRevision: string;
  readonly contextDigest: string;
  readonly executionRole?: "implementation" | "independent_review";
  readonly checkpoint?: RuntimeCheckpoint;
  readonly signal?: AbortSignal;
}

export interface RuntimeSession {
  readonly sessionId: string;
  readonly provider: RuntimeProvider;
  readonly providerSessionId?: string;
  readonly status: "completed" | "failed" | "terminated";
  readonly output?: string;
  readonly events: readonly unknown[];
  readonly startedAt: string;
  readonly completedAt: string;
  readonly diagnostic?: string;
}

export interface AgentRuntimePort {
  readonly provider: RuntimeProvider;
  inspectCapabilities(): Promise<RuntimeCapabilitySnapshot>;
  start(input: RuntimeWorkInput): Promise<RuntimeSession>;
  resume(providerSessionId: string, input: RuntimeWorkInput): Promise<RuntimeSession>;
  terminate(providerSessionId: string): Promise<void>;
}

export interface RuntimeWorktree {
  readonly worktreeId: string;
  readonly workspacePath: string;
  readonly branchName: string;
  readonly baseRevision: string;
}

export interface RuntimeWorktreePort {
  prepare(input: {
    readonly taskId: string;
    readonly workItem: LogicWorkItem;
    readonly baseRevision: string;
  }): Promise<RuntimeWorktree>;
}

export interface RuntimeWorkPreparationPort {
  prepare(input: {
    readonly taskId: string;
    readonly workItem: LogicWorkItem;
    readonly worktree: RuntimeWorktree;
    readonly provider: RuntimeProvider;
    readonly attempt: number;
    readonly totalTokenBudget: number;
    readonly optionalEvidenceTokenBudget: number;
    readonly receiptTtlSeconds: number;
  }): Promise<void>;
}

export interface RuntimeLease {
  readonly leaseId: string;
  readonly taskId: string;
  readonly workItemId: string;
  readonly provider: RuntimeProvider;
  readonly workspacePath: string;
  readonly symbolIds: readonly string[];
  readonly paths: readonly string[];
  readonly acquiredAt: string;
  readonly expiresAt: string;
  readonly releasedAt?: string;
}

export interface RuntimeLeaseStore {
  acquire(lease: RuntimeLease): Promise<boolean>;
  release(leaseId: string, releasedAt: string): Promise<void>;
  listActive(now: string): Promise<readonly RuntimeLease[]>;
}

export function createRuntimeCapabilitySnapshot(
  input: RuntimeCapabilitySnapshotInput,
): RuntimeCapabilitySnapshot {
  return Object.freeze({
    ...input,
    snapshotId: `runtime-capability:${sha256(stableJson(input))}`,
  });
}

export function createRuntimeCheckpoint(
  input: Omit<RuntimeCheckpoint, "checkpointId">,
): RuntimeCheckpoint {
  return Object.freeze({
    ...input,
    checkpointId: `runtime-checkpoint:${sha256(stableJson(input))}`,
  });
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function stableJson(value: unknown): string {
  return JSON.stringify(stableValue(value));
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, nested]) => [key, stableValue(nested)]),
    );
  }
  return value;
}
