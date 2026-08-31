import type { TaskContract } from "./domain.js";

export type TaskContractIssueCode =
  | "missing_intent"
  | "missing_acceptance"
  | "missing_targets"
  | "duplicate_criterion_id"
  | "invalid_acceptance_criterion";

export interface TaskContractIssue {
  readonly code: TaskContractIssueCode;
  readonly message: string;
  readonly ref?: string;
}

export function validateTaskContract(contract: TaskContract): readonly TaskContractIssue[] {
  const issues: TaskContractIssue[] = [];
  if (!contract.intent.trim()) {
    issues.push({ code: "missing_intent", message: "Task intent is required" });
  }
  if (contract.acceptance.length === 0) {
    issues.push({
      code: "missing_acceptance",
      message: "At least one acceptance criterion is required",
    });
  }
  if (contract.targets.length === 0 || contract.targets.every((target) => !target.trim())) {
    issues.push({ code: "missing_targets", message: "At least one Task target is required" });
  }

  const criterionIds = new Set<string>();
  for (const criterion of contract.acceptance) {
    if (criterionIds.has(criterion.criterionId)) {
      issues.push({
        code: "duplicate_criterion_id",
        message: `Duplicate acceptance criterion ID: ${criterion.criterionId}`,
        ref: criterion.criterionId,
      });
    }
    criterionIds.add(criterion.criterionId);
    if (
      !criterion.criterionId.trim() ||
      !criterion.statement.trim() ||
      !criterion.verifier.ref.trim()
    ) {
      issues.push({
        code: "invalid_acceptance_criterion",
        message: "Acceptance criteria require an ID, statement, and verifier reference",
        ref: criterion.criterionId,
      });
    }
  }
  return issues;
}
