import { describe, expect, it } from "vitest";
import {
  type AccuracyManifest,
  type ChangeBundle,
  type EvaluateTaskStateInput,
  REQUIRED_COMPLETION_VERIFIERS,
  type TaskContextSnapshot,
  type TaskContract,
  type VerificationRun,
  computeTaskContextDigest,
  createAccuracyManifest,
  createChangeBundle,
  evaluateTaskState,
  taskContractDigest,
  validateTaskContract,
} from "../src/index.js";

const contract: TaskContract = {
  taskId: "task:accurate-code",
  intent: "Keep accepted Decisions enforced while writing code.",
  acceptance: [
    {
      criterionId: "acceptance:tests",
      statement: "The transition tests pass.",
      verifier: { kind: "test", ref: "spec:task-transition" },
    },
  ],
  nonGoals: ["Implement provider runtimes"],
  targets: ["planned-symbol:evaluate-task-state"],
  risk: "low",
};

const digestInput = {
  taskId: contract.taskId,
  baseCodeRevision: "base-commit",
  effectiveScopes: [{ kind: "workspace" as const, workspaceId: "workspace:kontext" }],
  normativeRevisions: [
    {
      kind: "invariant" as const,
      recordId: "invariant:proof",
      revisionId: "invariant:proof@1",
    },
  ],
  requiredEvidenceIds: ["evidence:decision"],
  sourceFreshnessDigest: "freshness:1",
};

const contextDigest = computeTaskContextDigest(digestInput);
const snapshot: TaskContextSnapshot = {
  ...digestInput,
  contextDigest,
  createdAt: "2026-08-28T01:00:00.000Z",
};

function passingRun(overrides: Partial<VerificationRun> = {}): VerificationRun {
  return {
    verificationRunId: "verification:tests",
    tier: "full",
    verifierKind: "test",
    verifierRef: "spec:task-transition",
    codeRevision: "commit:current",
    contextDigest,
    subjectIds: [contract.taskId],
    result: "passed",
    observedAt: "2026-08-28T01:01:00.000Z",
    ...overrides,
  };
}

function fullRuns(): readonly VerificationRun[] {
  return [
    passingRun(),
    ...REQUIRED_COMPLETION_VERIFIERS.map((verifier, index) =>
      passingRun({
        verificationRunId: `verification:completion:${index}`,
        verifierKind: verifier.kind,
        verifierRef: verifier.ref,
      }),
    ),
  ];
}

function changeBundle(overrides: Partial<Omit<ChangeBundle, "bundleId">> = {}): ChangeBundle {
  return createChangeBundle({
    taskId: contract.taskId,
    workItemId: "work-item:evaluate-task-state",
    baseRevision: snapshot.baseCodeRevision,
    resultRevision: "commit:current",
    taskContextDigest: contextDigest,
    patchDigest: "sha256:patch",
    changedSymbolIds: ["symbol:evaluateTaskState"],
    changedPaths: ["packages/spec/src/task-transition.ts"],
    contextReceiptIds: ["context-receipt:evaluate-task-state"],
    evidenceIds: [...snapshot.requiredEvidenceIds],
    normativeRevisions: [...snapshot.normativeRevisions],
    verificationRunIds: ["verification:tests"],
    proposals: [],
    unresolved: [],
    submittedAt: "2026-08-28T01:02:00.000Z",
    ...overrides,
  });
}

function manifestFor(input: Omit<EvaluateTaskStateInput, "accuracyManifest">): AccuracyManifest {
  const taskBundles = input.changeBundles.filter(
    (bundle) => bundle.taskId === input.contract.taskId,
  );
  return createAccuracyManifest({
    taskId: input.contract.taskId,
    taskContractDigest: taskContractDigest(input.contract),
    contextDigest: input.snapshot.contextDigest,
    baseCodeRevision: input.snapshot.baseCodeRevision,
    resultCodeRevision: input.currentCodeRevision,
    normativeRevisions: input.snapshot.normativeRevisions,
    evidenceIds: [
      ...input.snapshot.requiredEvidenceIds,
      ...taskBundles.flatMap((bundle) => bundle.evidenceIds),
    ],
    workItemIds: taskBundles.map((bundle) => bundle.workItemId),
    changeBundleIds: taskBundles.map((bundle) => bundle.bundleId),
    changedSymbolIds: taskBundles.flatMap((bundle) => bundle.changedSymbolIds),
    verificationRunIds: input.verificationRuns
      .filter(
        (run) =>
          run.codeRevision === input.currentCodeRevision &&
          run.contextDigest === input.snapshot.contextDigest &&
          run.result === "passed",
      )
      .map((run) => run.verificationRunId),
    reviewFindingIds: input.reviewFindings.map((finding) => finding.findingId),
    emergencyBypassIds: [],
    createdAt: "2026-08-28T01:03:00.000Z",
  });
}

function completionInput(overrides: Partial<EvaluateTaskStateInput> = {}): EvaluateTaskStateInput {
  const base: Omit<EvaluateTaskStateInput, "accuracyManifest"> = {
    currentState: "in_progress",
    workStarted: true,
    completionRequested: true,
    contract,
    snapshot,
    context: { status: "current", contextDigest },
    currentCodeRevision: "commit:current",
    evidence: [
      {
        kind: "commit",
        ref: "commit:current",
        codeRevision: "commit:current",
        contextDigest,
        observedAt: "2026-08-28T01:01:00.000Z",
      },
    ],
    verificationRuns: fullRuns(),
    invariantEvaluations: [
      {
        invariantId: "invariant:proof",
        revisionId: "invariant:proof@1",
        status: "guarded",
        verificationRunIds: ["verification:tests"],
      },
    ],
    reviewFindings: [],
    changeBundles: [changeBundle()],
  };
  const { accuracyManifest: overrideManifest, ...otherOverrides } = overrides;
  const merged = { ...base, ...otherOverrides };
  return {
    ...merged,
    accuracyManifest: "accuracyManifest" in overrides ? overrideManifest : manifestFor(merged),
  };
}

describe("Task Contract validation", () => {
  it("requires intent, acceptance, and targets before completion", () => {
    expect(
      validateTaskContract({ ...contract, intent: " ", acceptance: [], targets: [] }).map(
        (issue) => issue.code,
      ),
    ).toEqual(["missing_intent", "missing_acceptance", "missing_targets"]);
  });
});

describe("Task state evaluation", () => {
  it("reaches done only with current proof for every acceptance criterion and Invariant", () => {
    expect(evaluateTaskState(completionInput())).toEqual({ state: "done", issues: [] });
  });

  it("keeps completion awaiting evidence until an Accuracy Manifest exists", () => {
    const result = evaluateTaskState(completionInput({ accuracyManifest: undefined }));

    expect(result.state).toBe("awaiting_evidence");
    expect(result.issues).toContainEqual(
      expect.objectContaining({ code: "missing_accuracy_manifest" }),
    );
  });

  it("blocks a tampered Accuracy Manifest", () => {
    const valid = completionInput().accuracyManifest;
    if (!valid) throw new Error("expected completion fixture to include an Accuracy Manifest");
    const result = evaluateTaskState(
      completionInput({
        accuracyManifest: { ...valid, resultCodeRevision: "commit:other" },
      }),
    );

    expect(result.state).toBe("blocked");
    expect(result.issues).toContainEqual(
      expect.objectContaining({ code: "invalid_accuracy_manifest" }),
    );
  });

  it("blocks a Manifest that omits a required full completion run", () => {
    const valid = completionInput().accuracyManifest;
    if (!valid) throw new Error("expected completion fixture to include an Accuracy Manifest");
    const omittedRunId = "verification:completion:0";
    const incomplete = createAccuracyManifest({
      ...valid,
      verificationRunIds: valid.verificationRunIds.filter((runId) => runId !== omittedRunId),
    });
    const result = evaluateTaskState(completionInput({ accuracyManifest: incomplete }));

    expect(result.state).toBe("blocked");
    expect(result.issues).toContainEqual(
      expect.objectContaining({ code: "invalid_accuracy_manifest" }),
    );
  });

  it("does not reuse a passing Verification Run from another code revision", () => {
    const result = evaluateTaskState(
      completionInput({
        verificationRuns: [passingRun({ codeRevision: "commit:old" })],
        accuracyManifest: undefined,
      }),
    );

    expect(result.state).toBe("awaiting_evidence");
    expect(result.issues).toContainEqual(
      expect.objectContaining({ code: "missing_acceptance_verification" }),
    );
  });

  it("does not reuse Evidence from another context digest", () => {
    const result = evaluateTaskState(
      completionInput({
        evidence: [
          {
            kind: "commit",
            ref: "commit:current",
            codeRevision: "commit:current",
            contextDigest: "context:old",
            observedAt: "2026-08-28T01:01:00.000Z",
          },
        ],
      }),
    );

    expect(result.state).toBe("awaiting_evidence");
    expect(result.issues).toContainEqual(expect.objectContaining({ code: "missing_commit" }));
  });

  it("blocks when snapshot content no longer matches its digest", () => {
    const result = evaluateTaskState(
      completionInput({
        snapshot: {
          ...snapshot,
          normativeRevisions: [
            {
              kind: "invariant",
              recordId: "invariant:proof",
              revisionId: "invariant:proof@2",
            },
          ],
        },
      }),
    );

    expect(result.state).toBe("blocked");
    expect(result.issues).toContainEqual(
      expect.objectContaining({ code: "context_digest_mismatch" }),
    );
  });

  it("blocks when mandatory context is stale, conflicting, inaccessible, or unavailable", () => {
    for (const status of ["stale", "conflict", "inaccessible", "unavailable"] as const) {
      const result = evaluateTaskState(completionInput({ context: { status, contextDigest } }));

      expect(result.state).toBe("blocked");
      expect(result.issues).toContainEqual(expect.objectContaining({ code: `context_${status}` }));
    }
  });

  it("blocks on a failed required verifier", () => {
    const result = evaluateTaskState(
      completionInput({ verificationRuns: [passingRun({ result: "failed" })] }),
    );

    expect(result.state).toBe("blocked");
    expect(result.issues).toContainEqual(expect.objectContaining({ code: "failed_verification" }));
  });

  it("keeps inconclusive verification awaiting evidence instead of passing", () => {
    const result = evaluateTaskState(
      completionInput({
        verificationRuns: [passingRun({ result: "inconclusive" })],
        accuracyManifest: undefined,
      }),
    );

    expect(result.state).toBe("awaiting_evidence");
    expect(result.issues).toContainEqual(
      expect.objectContaining({ code: "inconclusive_verification" }),
    );
  });

  it("distinguishes an unguarded Invariant from a violated Invariant", () => {
    const unguarded = evaluateTaskState(
      completionInput({
        invariantEvaluations: [
          {
            invariantId: "invariant:proof",
            revisionId: "invariant:proof@1",
            status: "unguarded",
            verificationRunIds: [],
          },
        ],
      }),
    );
    expect(unguarded.state).toBe("awaiting_evidence");
    expect(unguarded.issues).toContainEqual(
      expect.objectContaining({ code: "unguarded_invariant" }),
    );

    const violated = evaluateTaskState(
      completionInput({
        invariantEvaluations: [
          {
            invariantId: "invariant:proof",
            revisionId: "invariant:proof@1",
            status: "violated",
            verificationRunIds: ["verification:tests"],
          },
        ],
      }),
    );
    expect(violated.state).toBe("blocked");
    expect(violated.issues).toContainEqual(expect.objectContaining({ code: "violated_invariant" }));
  });

  it("blocks completion while an independent Review Finding remains open", () => {
    const result = evaluateTaskState(
      completionInput({
        reviewFindings: [
          {
            findingId: "finding:1",
            status: "open",
            codeRevision: "commit:current",
            contextDigest,
            symbolId: "symbol:evaluateTaskState",
            evidenceIds: ["evidence:review"],
          },
        ],
      }),
    );

    expect(result.state).toBe("blocked");
    expect(result.issues).toContainEqual(
      expect.objectContaining({ code: "unresolved_review_finding" }),
    );
  });

  it("requires risk-based human approvals for medium and high risk completion", () => {
    const medium = evaluateTaskState(
      completionInput({ contract: { ...contract, risk: "medium" } }),
    );
    expect(medium.state).toBe("awaiting_evidence");
    expect(medium.issues).toContainEqual(
      expect.objectContaining({ code: "missing_code_owner_approval" }),
    );

    const highWithCodeOwner = evaluateTaskState(
      completionInput({
        contract: { ...contract, risk: "high" },
        evidence: [
          ...completionInput().evidence,
          {
            kind: "approval",
            role: "code_owner",
            ref: "approval:code-owner",
            codeRevision: "commit:current",
            contextDigest,
            observedAt: "2026-08-28T01:02:00.000Z",
          },
        ],
      }),
    );
    expect(highWithCodeOwner.state).toBe("awaiting_evidence");
    expect(highWithCodeOwner.issues).toContainEqual(
      expect.objectContaining({ code: "missing_domain_owner_approval" }),
    );
  });
});
