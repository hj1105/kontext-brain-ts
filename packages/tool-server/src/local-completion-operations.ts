import { createHash } from "node:crypto";
import path from "node:path";
import type {
  CurrentTaskContextState,
  PreparedTaskContext,
  PreparedTaskContextStore,
  TaskContextStateProvider,
} from "@kontext-brain/context";
import { validateContextReceipt } from "@kontext-brain/context";
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
import type {
  IntegratedTaskState,
  IntegratedTaskStateStore,
} from "./file-integrated-task-state-store.js";
import type { SidecarChangeEvidenceProvider } from "./sidecar-change-evidence.js";
import { captureWorkspaceSnapshot } from "./workspace-change-observer.js";

export class LocalKontextCompletionOperations implements KontextCompletionOperations {
  constructor(
    private readonly currentState: TaskContextStateProvider,
    private readonly preparedTasks: PreparedTaskContextStore,
    private readonly artifacts: TaskCompletionArtifactStore,
    private readonly quarantine: QuarantineStore,
    private readonly verification: DurableVerificationCoordinator,
    private readonly changeEvidence: SidecarChangeEvidenceProvider,
    private readonly integratedTasks?: IntegratedTaskStateStore,
  ) {}

  async checkChange(request: CheckChangeRequest): Promise<unknown> {
    const {
      prepared,
      current,
      workItem,
      plan: logicPlan,
    } = await this.loadTask(request.taskId, request.workItemId);
    const evidence = await this.changeEvidence.observe({
      workspacePath: request.workspacePath,
      taskId: request.taskId,
      workItem,
      plannedSymbols: logicPlan.plannedSymbols,
    });
    assertCheckEvidence(evidence, workItem, prepared, request.observedAt);
    const invariantVerifiers = boundInvariantVerifiers(current, prepared);
    const plan =
      request.tier === "fast"
        ? createFastVerificationPlan({
            affectedSymbolIds: evidence.observedPatch.changedSymbolIds,
          })
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
        codeRevision: evidence.currentCodeRevision,
        contextDigest: prepared.snapshot.contextDigest,
        observedAt: request.observedAt,
      },
      nextAttemptAt: request.nextAttemptAt,
    });
    await this.artifacts.putVerificationRuns(
      request.taskId,
      executions.map((execution) => execution.run),
    );
    return { plan, executions, observedPatch: evidence.observedPatch };
  }

  async submitChangeBundle(request: SubmitChangeBundleRequest): Promise<unknown> {
    const bundle = createChangeBundle(request.bundle);
    const { prepared, current, workItem, plan } = await this.loadTask(
      bundle.taskId,
      bundle.workItemId,
    );
    const evidence = await this.changeEvidence.observe({
      workspacePath: request.workspacePath,
      taskId: bundle.taskId,
      workItem,
      plannedSymbols: plan.plannedSymbols,
    });
    const validation = validateChangeBundle({
      bundle,
      workItem,
      snapshot: prepared.snapshot,
      currentCodeRevision: evidence.currentCodeRevision,
      observedPatch: evidence.observedPatch,
      receipts: evidence.receipts,
      plannedSymbolIssues: evidence.plannedSymbolIssues,
      unauthorizedChangedSymbolIds: evidence.unauthorizedChangedSymbolIds,
      verificationRuns: await this.artifacts.listVerificationRuns(bundle.taskId),
      boundInvariantVerifiers: boundInvariantVerifiers(current, prepared),
      quarantineRecords: await this.quarantine.list("active"),
    });
    if (validation.accepted) await this.artifacts.putChangeBundle(bundle);
    return {
      ...validation,
      bundle,
      observedPatch: evidence.observedPatch,
      plannedSymbolBindings: evidence.plannedSymbolBindings,
    };
  }

  async proposeTransition(request: ProposeTransitionRequest): Promise<unknown> {
    const prepared = await this.requirePrepared(request.taskId);
    const current = await this.currentState.getCurrent(request.taskId);
    let verificationRuns = await this.artifacts.listVerificationRuns(request.taskId);
    const changeBundles = await this.artifacts.listChangeBundles(request.taskId);
    const integration = await this.integratedTasks?.get(request.taskId);
    const currentCodeRevision = integration
      ? await validateIntegratedTaskState(integration, prepared, changeBundles)
      : completionCodeRevision(changeBundles, current.codeRevision);
    const reviewFindings = (await this.artifacts.listReviewFindings(request.taskId)).filter(
      (finding) =>
        finding.codeRevision === currentCodeRevision &&
        finding.contextDigest === prepared.snapshot.contextDigest,
    );
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
          currentCodeRevision,
          changeBundles,
          verificationRuns: candidateRuns,
          reviewFindings,
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
            codeRevision: currentCodeRevision,
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
      currentCodeRevision,
      evidence: request.evidence,
      verificationRuns,
      invariantEvaluations: request.invariantEvaluations,
      reviewFindings,
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
      integration,
    };
  }

  private async loadTask(
    taskId: string,
    workItemId: string,
  ): Promise<{
    readonly prepared: PreparedTaskContext;
    readonly current: CurrentTaskContextState;
    readonly workItem: LogicWorkItem;
    readonly plan: CurrentTaskContextState["logicPlans"][number];
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
      plan,
    };
  }

  private async requirePrepared(taskId: string): Promise<PreparedTaskContext> {
    const prepared = await this.preparedTasks.get(taskId);
    if (!prepared) throw new Error(`Task "${taskId}" has no prepared context`);
    return prepared;
  }
}

async function validateIntegratedTaskState(
  integration: IntegratedTaskState,
  prepared: PreparedTaskContext,
  bundles: readonly { readonly bundleId: string }[],
): Promise<string> {
  if (integration.taskId !== prepared.contract.taskId) {
    throw new Error("Integrated Task state belongs to another Task");
  }
  if (integration.contextDigest !== prepared.snapshot.contextDigest) {
    throw new Error("Integrated Task state uses stale Task context");
  }
  if (
    !sameStringSet(
      integration.changeBundleIds,
      bundles.map((bundle) => bundle.bundleId),
    )
  ) {
    throw new Error("Integrated Task state does not include every accepted Change Bundle");
  }
  const observed = await captureWorkspaceSnapshot(
    integration.workspacePath,
    integration.changedPaths,
  );
  if (observed.revision !== integration.resultRevision) {
    throw new Error("Integrated workspace changed after sidecar integration");
  }
  return observed.revision;
}

function completionCodeRevision(
  bundles: readonly { readonly resultRevision: string }[],
  baseRevision: string,
): string {
  const revisions = Array.from(new Set(bundles.map((bundle) => bundle.resultRevision)));
  if (revisions.length > 1) {
    throw new Error("Accepted Change Bundles do not describe one integrated code revision");
  }
  return revisions[0] ?? baseRevision;
}

function assertCheckEvidence(
  evidence: Awaited<ReturnType<SidecarChangeEvidenceProvider["observe"]>>,
  workItem: LogicWorkItem,
  prepared: PreparedTaskContext,
  observedAt: string,
): void {
  if (evidence.plannedSymbolIssues.length > 0) {
    throw new Error(
      `Cannot verify unbound Planned Symbols: ${evidence.plannedSymbolIssues
        .map((issue) => issue.plannedSymbolId)
        .join(", ")}`,
    );
  }
  if (evidence.unauthorizedChangedSymbolIds.length > 0) {
    throw new Error(
      `Cannot verify out-of-scope Code Symbols: ${evidence.unauthorizedChangedSymbolIds.join(", ")}`,
    );
  }
  const allowedPaths = new Set(workItem.allowedPaths.map(canonicalPath));
  const outsidePaths = evidence.observedPatch.changedPaths
    .map(canonicalPath)
    .filter((changedPath) => !allowedPaths.has(changedPath));
  if (outsidePaths.length > 0) {
    throw new Error(`Cannot verify out-of-scope paths: ${outsidePaths.join(", ")}`);
  }
  const receipt = evidence.receipts[0];
  if (!receipt || evidence.receipts.length !== 1) {
    throw new Error("Cannot verify without one sidecar-owned Context Receipt");
  }
  const receiptIssues = validateContextReceipt({
    receipt,
    snapshot: prepared.snapshot,
    logic: workItem,
    allowedPaths: workItem.allowedPaths,
    now: observedAt,
  });
  if (receiptIssues.length > 0) {
    throw new Error(`Cannot verify invalid Context Receipt: ${receiptIssues.join(", ")}`);
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

export function assessCurrentContext(
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

function canonicalPath(value: string): string {
  return value.replaceAll("\\", "/").replace(/^\.\//, "");
}

function sameStringSet(left: readonly string[], right: readonly string[]): boolean {
  const normalize = (values: readonly string[]) => Array.from(new Set(values)).sort();
  return JSON.stringify(normalize(left)) === JSON.stringify(normalize(right));
}
