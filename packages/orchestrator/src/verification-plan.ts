import type { VerificationTier, VerifierRef } from "@kontext-brain/spec";
import type {
  FastVerificationPlanInput,
  FullVerificationPlanInput,
  TargetedVerificationPlanInput,
  VerificationPlan,
  VerificationRequirement,
} from "./domain.js";

const FAST_VERIFIERS: readonly VerifierRef[] = [
  { kind: "query", ref: "kontext:semantic-sync" },
  { kind: "query", ref: "kontext:stable-symbol-identity" },
  { kind: "query", ref: "kontext:domain-term-check" },
  { kind: "query", ref: "kontext:graph-query-check" },
];

const FULL_VERIFIERS: readonly VerifierRef[] = [
  { kind: "typecheck", ref: "workspace:typecheck" },
  { kind: "test", ref: "workspace:test" },
  { kind: "build", ref: "workspace:build" },
  { kind: "lint", ref: "workspace:lint" },
  { kind: "manual_review", ref: "kontext:independent-review" },
];

export function createFastVerificationPlan(input: FastVerificationPlanInput): VerificationPlan {
  return plan("fast", FAST_VERIFIERS, input.affectedSymbolIds);
}

export function createTargetedVerificationPlan(
  input: TargetedVerificationPlanInput,
): VerificationPlan {
  return plan(
    "targeted",
    [...input.workItem.requiredVerifiers, ...(input.boundInvariantVerifiers ?? [])],
    [input.workItem.workItemId, ...input.workItem.plannedSymbolIds],
  );
}

export function createFullVerificationPlan(input: FullVerificationPlanInput): VerificationPlan {
  return plan(
    "full",
    [
      ...FULL_VERIFIERS,
      ...input.contract.acceptance.map((criterion) => criterion.verifier),
      ...(input.boundInvariantVerifiers ?? []),
    ],
    [input.contract.taskId, ...input.contract.targets],
  );
}

function plan(
  tier: VerificationTier,
  verifiers: readonly VerifierRef[],
  subjectIds: readonly string[],
): VerificationPlan {
  const subjects = uniqueSorted(subjectIds);
  const requirements = new Map<string, VerificationRequirement>();
  for (const verifier of verifiers) {
    const key = `${verifier.kind}\u0000${verifier.ref}`;
    requirements.set(key, { tier, verifier, subjectIds: subjects });
  }
  return {
    tier,
    requirements: Array.from(requirements.entries())
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([, requirement]) => requirement),
  };
}

function uniqueSorted(values: readonly string[]): readonly string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}
