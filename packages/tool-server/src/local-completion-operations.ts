import { createHash } from "node:crypto";
import path from "node:path";
import type {
  CurrentTaskContextState,
  PreparedTaskContext,
  PreparedTaskContextStore,
  TaskContextStateProvider,
} from "@kontext-brain/context";
import {
  type DurableVerificationCoordinator,
  type QuarantineStore,
  type TaskCompletionArtifactStore,
  assembleAccuracyManifest,
  auditAccuracyManifestCandidate,
  createFastVerificationPlan,
  createFullVerificationPlan,
  createTargetedVerificationPlan,
  createVerificationRun,
  validateChangeBundle,
} from "@kontext-brain/orchestrator";
import {
  type ContextAssessment,
  type GovernanceScope,
  type LogicWorkItem,
  type NormativeRevisionRef,
  createChangeBundle,
  evaluateTaskState,
} from "@kontext-brain/spec";
import type {
  CheckChangeRequest,
  KontextCompletionOperations,
  ProposeTransitionRequest,
  SubmitChangeBundleRequest,
} from "./completion-workflow-tools.js";

export class LocalKontextCompletionOperations implements KontextCompletionOperations {
  constructor(
    private readonly currentState: TaskContextStateProvider,
    private readonly preparedTasks: PreparedTaskContextStore,
    private readonly artifacts: TaskCompletionArtifactStore,
    private readonly quarantine: QuarantineStore,
    private readonly verification: DurableVerificationCoordinator,
  ) {}

  async checkChange(request: CheckChangeRequest): Promise<unknown> {
    const { prepared, current, workItem } = await this.loadTask(request.taskId, request.workItemId);
    assertBinding(
      request.codeRevision,
      request.contextDigest,
      current.codeRevision,
      prepared.snapshot.contextDigest,
    );
    const invariantVerifiers = boundInvariantVerifiers(current, prepared);
    const plan =
      request.tier === "fast"
        ? createFastVerificationPlan({ affectedSymbolIds: request.affectedSymbolIds })
        : request.tier === "targeted"
          ? createTargetedVerificationPlan({
              workItem,
              boundInvariantVerifiers: invariantVerifiers,
            })
          : createFullVerificationPlan({
              contract: prepared.contract,
              boundInvariantVerifiers: invariantVerifiers,
            });
    const executions = await this.verification.executePlan({
      taskId: request.taskId,
      workItemId: request.workItemId,
      plan,
      binding: {
        workspacePath: path.resolve(request.workspacePath),
        codeRevision: request.codeRevision,
        contextDigest: request.contextDigest,
        observedAt: request.observedAt,
      },
      nextAttemptAt: request.nextAttemptAt,
    });
    await this.artifacts.putVerificationRuns(
      request.taskId,
      executions.map((execution) => execution.run),
    );
    return { plan, executions };
  }

  async submitChangeBundle(request: SubmitChangeBundleRequest): Promise<unknown> {
    const bundle = createChangeBundle(request.bundle);
    const { prepared, current, workItem } = await this.loadTask(bundle.taskId, bundle.workItemId);
    const validation = validateChangeBundle({
      bundle,
      workItem,
      snapshot: prepared.snapshot,
      currentCodeRevision: current.codeRevision,
      observedPatch: request.observedPatch,
      receipts: request.receipts,
      verificationRuns: await this.artifacts.listVerificationRuns(bundle.taskId),
      boundInvariantVerifiers: boundInvariantVerifiers(current, prepared),
      quarantineRecords: await this.quarantine.list("active"),
    });
    if (validation.accepted) await this.artifacts.putChangeBundle(bundle);
    return { ...validation, bundle };
  }

  async proposeTransition(request: ProposeTransitionRequest): Promise<unknown> {
    const prepared = await this.requirePrepared(request.taskId);
    const current = await this.currentState.getCurrent(request.taskId);
    if (request.currentCodeRevision !== current.codeRevision) {
      throw new Error("Requested Task transition code revision is not sidecar-current");
    }
    let verificationRuns = await this.artifacts.listVerificationRuns(request.taskId);
    const changeBundles = await this.artifacts.listChangeBundles(request.taskId);
    let accuracyManifest = await this.artifacts.getAccuracyManifest(request.taskId);
    let accuracyManifestError: string | undefined;
    if (request.completionRequested) {
      try {
        const candidateRuns = verificationRuns.filter(
          (run) =>
            !(
              run.tier === "full" &&
              run.verifierKind === "query" &&
              run.verifierRef === "kontext:manifest-audit"
            ),
        );
        const auditInput = {
          contract: prepared.contract,
          snapshot: prepared.snapshot,
          currentCodeRevision: current.codeRevision,
          changeBundles,
          verificationRuns: candidateRuns,
          reviewFindings: request.reviewFindings,
          additionalEvidenceIds: request.evidence.map((evidence) => evidence.ref),
          createdAt: request.requestedAt,
        };
        const audit = auditAccuracyManifestCandidate(auditInput);
        const manifestAuditRun = createVerificationRun(
          {
            tier: "full",
            verifier: { kind: "query", ref: "kontext:manifest-audit" },
            subjectIds: [prepared.contract.taskId, ...prepared.contract.targets],
          },
          {
            codeRevision: current.codeRevision,
            contextDigest: prepared.snapshot.contextDigest,
            observedAt: request.requestedAt,
          },
          audit.passed ? "passed" : "failed",
          {
            candidateManifestId: audit.candidate.manifestId,
            blockingIssues: audit.blockingIssues,
            selfEvidenceIssues: audit.selfEvidenceIssues,
          },
        );
        await this.artifacts.putVerificationRuns(request.taskId, [manifestAuditRun]);
        verificationRuns = [...candidateRuns, manifestAuditRun];
        if (audit.passed) {
          accuracyManifest = assembleAccuracyManifest({
            ...auditInput,
            verificationRuns,
          });
          await this.artifacts.putAccuracyManifest(accuracyManifest);
        } else {
          accuracyManifest = undefined;
        }
      } catch (error) {
        accuracyManifest = undefined;
        accuracyManifestError = error instanceof Error ? error.message : String(error);
      }
    }
    const context = assessCurrentContext(prepared, current);
    const evaluation = evaluateTaskState({
      currentState: request.currentState,
      workStarted: request.workStarted,
      completionRequested: request.completionRequested,
      contract: prepared.contract,
      snapshot: prepared.snapshot,
      context,
      currentCodeRevision: current.codeRevision,
      evidence: request.evidence,
      verificationRuns,
      invariantEvaluations: request.invariantEvaluations,
      reviewFindings: request.reviewFindings,
      changeBundles,
      accuracyManifest,
    });
    return {
      ...evaluation,
      context,
      reportedContextMatched:
        request.context.status === context.status &&
        request.context.contextDigest === context.contextDigest,
      accuracyManifest,
      accuracyManifestError,
    };
  }

  private async loadTask(
    taskId: string,
    workItemId: string,
  ): Promise<{
    readonly prepared: PreparedTaskContext;
    readonly current: CurrentTaskContextState;
    readonly workItem: LogicWorkItem;
  }> {
    const prepared = await this.requirePrepared(taskId);
    const current = await this.currentState.getCurrent(taskId);
    const plan = current.logicPlans.find((candidate) => candidate.workItemId === workItemId);
    if (!plan) throw new Error(`Logic Work Item "${workItemId}" is not sidecar-planned`);
    return {
      prepared,
      current,
      workItem: {
        workItemId: plan.workItemId,
        taskId,
        plannedSymbolIds: plan.plannedSymbolIds,
        dependsOn: plan.dependsOn ?? [],
        allowedPaths: plan.allowedPaths,
        requiredVerifiers: plan.requiredVerifiers ?? [],
        capabilityId:
          plan.capabilityId ??
          `capability:${createHash("sha256")
            .update(
              JSON.stringify([taskId, plan.workItemId, plan.plannedSymbolIds, plan.allowedPaths]),
            )
            .digest("hex")}`,
      },
    };
  }

  private async requirePrepared(taskId: string): Promise<PreparedTaskContext> {
    const prepared = await this.preparedTasks.get(taskId);
    if (!prepared) throw new Error(`Task "${taskId}" has no prepared context`);
    return prepared;
  }
}

function assertBinding(
  requestedCodeRevision: string,
  requestedContextDigest: string,
  currentCodeRevision: string,
  currentContextDigest: string,
): void {
  if (requestedCodeRevision !== currentCodeRevision) {
    throw new Error("Requested Verification Run code revision is not sidecar-current");
  }
  if (requestedContextDigest !== currentContextDigest) {
    throw new Error("Requested Verification Run context digest is not the prepared Snapshot");
  }
}

function boundInvariantVerifiers(current: CurrentTaskContextState, prepared: PreparedTaskContext) {
  const revisions = new Set(
    prepared.snapshot.normativeRevisions
      .filter((revision) => revision.kind === "invariant")
      .map(revisionKey),
  );
  return current.normativeRecords.flatMap((record) =>
    record.revision.kind === "invariant" && revisions.has(revisionKey(record.revision))
      ? record.revision.verifiers
      : [],
  );
}

function assessCurrentContext(
  prepared: PreparedTaskContext,
  current: CurrentTaskContextState,
): ContextAssessment {
  const snapshot = prepared.snapshot;
  if (current.conflicts.length > 0) {
    return { status: "conflict", contextDigest: snapshot.contextDigest };
  }
  const requiredEvidence = new Set(snapshot.requiredEvidenceIds);
  const evidence = new Map(current.evidence.map((item) => [item.evidenceId, item]));
  for (const evidenceId of requiredEvidence) {
    const item = evidence.get(evidenceId);
    if (!item) return { status: "unavailable", contextDigest: snapshot.contextDigest };
    if (item.availability !== "current") {
      return { status: item.availability, contextDigest: snapshot.contextDigest };
    }
  }
  const currentRevisions = current.normativeRecords.map((record) => ({
    kind: record.revision.kind,
    recordId: record.revision.recordId,
    revisionId: record.revision.revisionId,
  }));
  if (
    snapshot.sourceFreshnessDigest !== current.sourceFreshnessDigest ||
    !sameValues(snapshot.effectiveScopes, current.effectiveScopes) ||
    !sameValues(snapshot.normativeRevisions.map(revisionKey), currentRevisions.map(revisionKey))
  ) {
    return { status: "stale", contextDigest: snapshot.contextDigest };
  }
  return { status: "current", contextDigest: snapshot.contextDigest };
}

function revisionKey(revision: NormativeRevisionRef): string {
  return JSON.stringify([revision.kind, revision.recordId, revision.revisionId]);
}

function sameValues(
  left: readonly GovernanceScope[] | readonly string[],
  right: readonly GovernanceScope[] | readonly string[],
): boolean {
  const normalize = (values: readonly unknown[]) =>
    values.map((value) => JSON.stringify(value)).sort();
  return JSON.stringify(normalize(left)) === JSON.stringify(normalize(right));
}
