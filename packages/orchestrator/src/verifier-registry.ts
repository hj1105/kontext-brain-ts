import { createHash } from "node:crypto";
import type { VerificationRun, VerifierRef } from "@kontext-brain/spec";
import type {
  VerificationBinding,
  VerificationExecution,
  VerificationPlan,
  VerificationRequirement,
  VerifierAdapter,
} from "./domain.js";

export class VerifierInfrastructureError extends Error {
  override readonly name = "VerifierInfrastructureError";
}

export class VerifierRegistry {
  private readonly adapters = new Map<string, VerifierAdapter>();
  private readonly fallbackAdapters = new Map<VerifierRef["kind"], VerifierAdapter>();

  register(verifier: VerifierRef, adapter: VerifierAdapter): void {
    const key = verifierKey(verifier);
    if (this.adapters.has(key)) {
      throw new Error(`Verifier ${verifier.kind}:${verifier.ref} is already registered`);
    }
    this.adapters.set(key, adapter);
  }

  registerFallback(kind: VerifierRef["kind"], adapter: VerifierAdapter): void {
    if (this.fallbackAdapters.has(kind)) {
      throw new Error(`Fallback verifier for ${kind} is already registered`);
    }
    this.fallbackAdapters.set(kind, adapter);
  }

  resolve(requirement: VerificationRequirement): VerifierAdapter | undefined {
    return (
      this.adapters.get(verifierKey(requirement.verifier)) ??
      this.fallbackAdapters.get(requirement.verifier.kind)
    );
  }
}

export class VerificationCoordinator {
  constructor(private readonly registry: VerifierRegistry) {}

  async executePlan(
    plan: VerificationPlan,
    binding: VerificationBinding,
  ): Promise<readonly VerificationExecution[]> {
    return Promise.all(plan.requirements.map((requirement) => this.execute(requirement, binding)));
  }

  async execute(
    requirement: VerificationRequirement,
    binding: VerificationBinding,
  ): Promise<VerificationExecution> {
    const adapter = this.registry.resolve(requirement);
    if (!adapter) {
      return inconclusiveExecution(
        requirement,
        binding,
        "unregistered_verifier",
        `No adapter is registered for ${requirement.verifier.kind}:${requirement.verifier.ref}`,
      );
    }

    try {
      const result = await adapter.execute({ requirement, ...binding });
      return {
        run: createVerificationRun(requirement, binding, result.result, result.output),
        disposition: "settled",
      };
    } catch (error) {
      const diagnostic = error instanceof Error ? error.message : "Verifier execution failed";
      const reason =
        error instanceof VerifierInfrastructureError
          ? "verifier_infrastructure_failure"
          : "verifier_execution_failure";
      return inconclusiveExecution(requirement, binding, reason, diagnostic);
    }
  }
}

function inconclusiveExecution(
  requirement: VerificationRequirement,
  binding: VerificationBinding,
  reason: string,
  diagnostic: string,
): VerificationExecution {
  return {
    run: createVerificationRun(requirement, binding, "inconclusive", { reason, diagnostic }),
    disposition: "retryable",
    diagnostic,
  };
}

export function createVerificationRun(
  requirement: VerificationRequirement,
  binding: Pick<VerificationBinding, "codeRevision" | "contextDigest" | "observedAt">,
  result: VerificationRun["result"],
  output?: unknown,
): VerificationRun {
  const subjectIds = uniqueSorted(requirement.subjectIds);
  const outputDigest = output === undefined ? undefined : digest(output);
  const identity = {
    tier: requirement.tier,
    verifierKind: requirement.verifier.kind,
    verifierRef: requirement.verifier.ref,
    codeRevision: binding.codeRevision,
    contextDigest: binding.contextDigest,
    subjectIds,
    result,
    outputDigest,
    observedAt: binding.observedAt,
  };
  return Object.freeze({
    verificationRunId: `verification-run:${sha256(stableJson(identity))}`,
    ...identity,
  });
}

function verifierKey(verifier: VerifierRef): string {
  return `${verifier.kind}\u0000${verifier.ref}`;
}

function digest(value: unknown): string {
  return `sha256:${sha256(stableJson(value))}`;
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

function uniqueSorted(values: readonly string[]): readonly string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}
