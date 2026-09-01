import { validateAccuracyManifestForTask } from "./accuracy-manifest.js";
import type {
  ApprovalRole,
  EvaluateTaskStateInput,
  InvariantEvaluation,
  TaskStateEvaluation,
  TaskTransitionIssue,
  TaskTransitionIssueCode,
  VerificationRun,
} from "./domain.js";
import { isTaskContextSnapshotValid } from "./task-context.js";
import { validateTaskContract } from "./task-contract.js";

export function evaluateTaskState(input: EvaluateTaskStateInput): TaskStateEvaluation {
  const issues: TaskTransitionIssue[] = [];
  const hardBlockers = new Set<TaskTransitionIssueCode>();
  const addIssue = (
    code: TaskTransitionIssueCode,
    message: string,
    ref?: string,
    hard = false,
  ): void => {
    if (!issues.some((issue) => issue.code === code && issue.ref === ref)) {
      issues.push({ code, message, ref });
    }
    if (hard) hardBlockers.add(code);
  };

  assessContext(input, addIssue);
  assessReviewFindings(input, addIssue);

  if (!input.workStarted) {
    return { state: hardBlockers.size > 0 ? "blocked" : "planned", issues };
  }
  if (!input.completionRequested) {
    return { state: hardBlockers.size > 0 ? "blocked" : "in_progress", issues };
  }

  const contractIssues = validateTaskContract(input.contract);
  for (const issue of contractIssues) {
    addIssue("invalid_task_contract", issue.message, issue.ref ?? issue.code);
  }

  const currentRuns = input.verificationRuns.filter(
    (run) =>
      run.codeRevision === input.currentCodeRevision &&
      run.contextDigest === input.snapshot.contextDigest,
  );

  const hasCurrentCommit = input.evidence.some(
    (evidence) =>
      evidence.kind === "commit" &&
      evidence.codeRevision === input.currentCodeRevision &&
      evidence.contextDigest === input.snapshot.contextDigest,
  );
  if (!hasCurrentCommit) {
    addIssue("missing_commit", "Current code revision and context require commit Evidence");
  }

  for (const criterion of input.contract.acceptance) {
    const matching = currentRuns.filter(
      (run) =>
        run.tier === "full" &&
        run.verifierKind === criterion.verifier.kind &&
        run.verifierRef === criterion.verifier.ref,
    );
    assessRequiredRuns(matching, criterion.criterionId, addIssue);
  }

  const requiredInvariants = input.snapshot.normativeRevisions.filter(
    (revision) => revision.kind === "invariant",
  );
  for (const required of requiredInvariants) {
    const evaluation = input.invariantEvaluations.find(
      (candidate) =>
        candidate.invariantId === required.recordId && candidate.revisionId === required.revisionId,
    );
    if (!evaluation) {
      addIssue(
        "missing_invariant_evaluation",
        `Invariant ${required.revisionId} has no current evaluation`,
        required.revisionId,
      );
      continue;
    }
    assessInvariant(evaluation, currentRuns, addIssue);
  }

  assessApprovals(input, addIssue);
  assessAccuracyManifest(input, addIssue);

  if (hardBlockers.size > 0) return { state: "blocked", issues };
  if (issues.length > 0) return { state: "awaiting_evidence", issues };
  return { state: "done", issues: [] };
}

function assessAccuracyManifest(input: EvaluateTaskStateInput, addIssue: AddIssue): void {
  if (!input.accuracyManifest) {
    addIssue("missing_accuracy_manifest", "Current completion requires an Accuracy Manifest");
    return;
  }

  const manifestIssues = validateAccuracyManifestForTask({
    manifest: input.accuracyManifest,
    contract: input.contract,
    snapshot: input.snapshot,
    currentCodeRevision: input.currentCodeRevision,
    changeBundles: input.changeBundles,
    verificationRuns: input.verificationRuns,
    reviewFindings: input.reviewFindings,
  });
  for (const issue of manifestIssues) {
    addIssue("invalid_accuracy_manifest", issue.message, issue.ref ?? issue.code, true);
  }
}

type AddIssue = (
  code: TaskTransitionIssueCode,
  message: string,
  ref?: string,
  hard?: boolean,
) => void;

function assessContext(input: EvaluateTaskStateInput, addIssue: AddIssue): void {
  if (!isTaskContextSnapshotValid(input.snapshot)) {
    addIssue(
      "context_digest_mismatch",
      "Task Context Snapshot content does not match its digest",
      undefined,
      true,
    );
  }
  if (input.context.status !== "current") {
    const code = `context_${input.context.status}` as TaskTransitionIssueCode;
    addIssue(code, `Mandatory context is ${input.context.status}`, undefined, true);
    return;
  }
  if (input.context.contextDigest !== input.snapshot.contextDigest) {
    addIssue(
      "context_digest_mismatch",
      "Current context digest does not match the Task Context Snapshot",
      undefined,
      true,
    );
  }
}

function assessReviewFindings(input: EvaluateTaskStateInput, addIssue: AddIssue): void {
  for (const finding of input.reviewFindings) {
    if (finding.status === "open") {
      addIssue(
        "unresolved_review_finding",
        `Review Finding ${finding.findingId} is unresolved`,
        finding.findingId,
        true,
      );
    }
  }
}

function assessRequiredRuns(
  matching: readonly VerificationRun[],
  criterionId: string,
  addIssue: AddIssue,
): void {
  if (matching.some((run) => run.result === "failed")) {
    addIssue(
      "failed_verification",
      `Acceptance criterion ${criterionId} has a failed Verification Run`,
      criterionId,
      true,
    );
    return;
  }
  if (matching.some((run) => run.result === "inconclusive")) {
    addIssue(
      "inconclusive_verification",
      `Acceptance criterion ${criterionId} is inconclusive`,
      criterionId,
    );
    return;
  }
  if (!matching.some((run) => run.result === "passed")) {
    addIssue(
      "missing_acceptance_verification",
      `Acceptance criterion ${criterionId} lacks a passing current Verification Run`,
      criterionId,
    );
  }
}

function assessInvariant(
  evaluation: InvariantEvaluation,
  currentRuns: readonly VerificationRun[],
  addIssue: AddIssue,
): void {
  switch (evaluation.status) {
    case "violated":
      addIssue(
        "violated_invariant",
        `Invariant ${evaluation.revisionId} is violated`,
        evaluation.revisionId,
        true,
      );
      return;
    case "unguarded":
      addIssue(
        "unguarded_invariant",
        `Invariant ${evaluation.revisionId} is unguarded`,
        evaluation.revisionId,
      );
      return;
    case "inconclusive":
      addIssue(
        "inconclusive_invariant",
        `Invariant ${evaluation.revisionId} is inconclusive`,
        evaluation.revisionId,
      );
      return;
    case "retired":
      addIssue(
        "retired_invariant",
        `Snapshot references retired Invariant ${evaluation.revisionId}`,
        evaluation.revisionId,
        true,
      );
      return;
    case "guarded": {
      const currentPassingRunIds = new Set(
        currentRuns
          .filter((run) => run.tier === "full" && run.result === "passed")
          .map((run) => run.verificationRunId),
      );
      if (
        evaluation.verificationRunIds.length === 0 ||
        !evaluation.verificationRunIds.every((runId) => currentPassingRunIds.has(runId))
      ) {
        addIssue(
          "missing_invariant_verification",
          `Invariant ${evaluation.revisionId} lacks current passing verifier Evidence`,
          evaluation.revisionId,
        );
      }
    }
  }
}

function assessApprovals(input: EvaluateTaskStateInput, addIssue: AddIssue): void {
  const approvals = new Set<ApprovalRole>();
  for (const evidence of input.evidence) {
    if (
      evidence.kind === "approval" &&
      evidence.codeRevision === input.currentCodeRevision &&
      evidence.contextDigest === input.snapshot.contextDigest
    ) {
      approvals.add(evidence.role);
    }
  }

  if (input.contract.risk === "medium" || input.contract.risk === "high") {
    if (!approvals.has("code_owner")) {
      addIssue("missing_code_owner_approval", "Current completion requires Code Owner approval");
    }
  }
  if (input.contract.risk === "high" && !approvals.has("domain_owner")) {
    addIssue(
      "missing_domain_owner_approval",
      "High-risk completion requires Domain Owner approval",
    );
  }
}
