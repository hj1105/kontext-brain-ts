import { createHash } from "node:crypto";
import path from "node:path";
import type { PreparedTaskContextStore, TaskContextStateProvider } from "@kontext-brain/context";
import { GitChangeBundleIntegrator } from "@kontext-brain/local";
import {
  type AgentRuntimePort,
  type DurableVerificationCoordinator,
  IndependentReviewCoordinator,
  type QuarantineStore,
  type TaskCompletionArtifactStore,
  createFullVerificationPlan,
  planChangeBundleIntegration,
  validateChangeBundle,
} from "@kontext-brain/orchestrator";
import type { LogicWorkItem, VerifierRef } from "@kontext-brain/spec";
import type {
  IntegratedTaskState,
  IntegratedTaskStateStore,
} from "./file-integrated-task-state-store.js";
import type { RuntimeScheduleJobView } from "./file-runtime-schedule-job-store.js";
import { assessCurrentContext } from "./local-completion-operations.js";
import type { IntegrateScheduleRequest } from "./runtime-schedule-contract.js";
import type { SidecarChangeEvidenceProvider } from "./sidecar-change-evidence.js";
import { captureWorkspaceSnapshot, observeWorkspacePatch } from "./workspace-change-observer.js";
import {
  captureWorkspaceCodeSymbols,
  changedWorkspaceCodeSymbolIds,
} from "./workspace-code-symbol-observer.js";

const maxReviewPacketBytes = 2 * 1024 * 1024;

export class LocalScheduleIntegrator {
  constructor(
    private readonly currentState: TaskContextStateProvider,
    private readonly preparedTasks: PreparedTaskContextStore,
    private readonly artifacts: TaskCompletionArtifactStore,
    private readonly quarantine: QuarantineStore,
    private readonly verification: DurableVerificationCoordinator,
    private readonly changeEvidence: SidecarChangeEvidenceProvider,
    private readonly integratedTasks: IntegratedTaskStateStore,
    private readonly runtimes: readonly AgentRuntimePort[],
    private readonly dataDirectory: string,
    private readonly now: () => Date = () => new Date(),
  ) {}

  async integrate(
    job: RuntimeScheduleJobView,
    request: IntegrateScheduleRequest,
  ): Promise<unknown> {
    if (job.status !== "completed" || !job.result) {
      throw new Error(`Runtime schedule ${job.jobId} is not completed`);
    }
    if (job.result.results.some((result) => result.status !== "completed")) {
      throw new Error(`Runtime schedule ${job.jobId} contains failed Logic Work Items`);
    }
    const prepared = await this.preparedTasks.get(job.taskId);
    if (!prepared) throw new Error(`Task "${job.taskId}" has no prepared context`);
    if (
      prepared.snapshot.contextDigest !== job.contextDigest ||
      prepared.snapshot.baseCodeRevision !== job.codeRevision
    ) {
      throw new Error("Runtime schedule no longer matches the prepared Task Context Snapshot");
    }
    const existingIntegration = await this.integratedTasks.get(job.taskId);
    if (existingIntegration?.scheduleJobId === job.jobId) {
      const observed = await captureWorkspaceSnapshot(
        existingIntegration.workspacePath,
        existingIntegration.changedPaths,
      );
      if (observed.revision !== existingIntegration.resultRevision) {
        throw new Error("Previously integrated workspace changed after sidecar integration");
      }
      return {
        state: existingIntegration,
        executions: (await this.artifacts.listVerificationRuns(job.taskId)).filter(
          (run) => run.codeRevision === existingIntegration.resultRevision,
        ),
        reviewFindings: (await this.artifacts.listReviewFindings(job.taskId)).filter(
          (finding) => finding.codeRevision === existingIntegration.resultRevision,
        ),
        reused: true,
      };
    }
    const current = await this.currentState.getCurrent(job.taskId);
    const context = assessCurrentContext(prepared, current);
    if (context.status !== "current") {
      throw new Error(`Task context is ${context.status}; refresh before integration`);
    }
    const plans = new Map(current.logicPlans.map((plan) => [plan.workItemId, plan] as const));
    const storedBundles = await this.artifacts.listChangeBundles(job.taskId);
    const verificationRuns = await this.artifacts.listVerificationRuns(job.taskId);
    const quarantineRecords = await this.quarantine.list("active");
    const workItems: LogicWorkItem[] = [];
    const bundles = [];
    const authors = [];
    const sources = new Map<string, string>();

    for (const result of job.result.results) {
      const plan = plans.get(result.workItemId);
      if (!plan || !result.provider || !result.worktree) {
        throw new Error(`Schedule result ${result.workItemId} lacks sidecar planning provenance`);
      }
      const workItem = logicWorkItem(job.taskId, plan);
      const evidence = await this.changeEvidence.observe({
        workspacePath: result.worktree.workspacePath,
        taskId: job.taskId,
        workItem,
        plannedSymbols: plan.plannedSymbols,
      });
      const candidates = storedBundles.filter(
        (bundle) =>
          bundle.workItemId === workItem.workItemId &&
          bundle.resultRevision === evidence.currentCodeRevision &&
          bundle.patchDigest === evidence.observedPatch.patchDigest,
      );
      if (candidates.length !== 1 || !candidates[0]) {
        throw new Error(
          `Schedule result ${result.workItemId} requires one current accepted Change Bundle`,
        );
      }
      const bundle = candidates[0];
      const validation = validateChangeBundle({
        bundle,
        workItem,
        snapshot: prepared.snapshot,
        currentCodeRevision: evidence.currentCodeRevision,
        observedPatch: evidence.observedPatch,
        plannedSymbolIssues: evidence.plannedSymbolIssues,
        unauthorizedChangedSymbolIds: evidence.unauthorizedChangedSymbolIds,
        receipts: evidence.receipts,
        verificationRuns,
        boundInvariantVerifiers: boundInvariantVerifiers(
          current,
          prepared.snapshot.normativeRevisions,
        ),
        quarantineRecords,
      });
      if (!validation.accepted) {
        throw new Error(
          `Change Bundle ${bundle.bundleId} is no longer acceptable: ${validation.issues
            .map((issue) => issue.code)
            .join(", ")}`,
        );
      }
      workItems.push(workItem);
      bundles.push(bundle);
      authors.push({ workItemId: result.workItemId, provider: result.provider });
      sources.set(result.workItemId, result.worktree.workspacePath);
    }

    const plan = planChangeBundleIntegration({
      taskId: job.taskId,
      workItems,
      changeBundles: bundles,
      authors,
    });
    const repositoryPath = path.resolve(job.repositoryPath);
    const integrator = new GitChangeBundleIntegrator(
      repositoryPath,
      path.join(
        this.dataDirectory,
        "integration-worktrees",
        createHash("sha256").update(repositoryPath).digest("hex"),
      ),
    );
    const workspace = await integrator.prepare({
      taskId: job.taskId,
      scheduleJobId: job.jobId,
      baseRevision: prepared.snapshot.baseCodeRevision,
    });
    const baseline = await captureWorkspaceSnapshot(workspace.workspacePath, plan.changedPaths);
    const baselineSymbols = await captureWorkspaceCodeSymbols(
      workspace.workspacePath,
      plan.changedPaths,
      baseline,
    );
    const gitResult = await integrator.apply(
      workspace,
      plan.orderedChangeBundles.map((bundle) => ({
        bundleId: bundle.bundleId,
        workItemId: bundle.workItemId,
        sourceWorkspacePath: requireSource(sources, bundle.workItemId),
        baseRevision: bundle.baseRevision,
        changedPaths: bundle.changedPaths,
      })),
    );
    const integrated = await captureWorkspaceSnapshot(workspace.workspacePath, plan.changedPaths);
    const integratedSymbols = await captureWorkspaceCodeSymbols(
      workspace.workspacePath,
      plan.changedPaths,
      integrated,
    );
    const observedPatch = observeWorkspacePatch(baseline, integrated);
    const integratedChangedSymbols = changedWorkspaceCodeSymbolIds(
      baselineSymbols,
      integratedSymbols,
    );
    if (!sameStrings(observedPatch.changedPaths, plan.changedPaths)) {
      throw new Error("Integrated paths do not match the accepted Change Bundles");
    }
    if (!sameStrings(integratedChangedSymbols, plan.changedSymbolIds)) {
      throw new Error("Integrated Code Symbols do not match the accepted Change Bundles");
    }

    const fullPlan = createFullVerificationPlan({
      contract: prepared.contract,
      boundInvariantVerifiers: boundInvariantVerifiers(
        current,
        prepared.snapshot.normativeRevisions,
      ),
    });
    const deterministicPlan = {
      ...fullPlan,
      requirements: fullPlan.requirements.filter(
        (requirement) => requirement.verifier.ref !== "kontext:independent-review",
      ),
    };
    const executions = await this.verification.executePlan({
      taskId: job.taskId,
      plan: deterministicPlan,
      binding: {
        workspacePath: workspace.workspacePath,
        codeRevision: integrated.revision,
        contextDigest: prepared.snapshot.contextDigest,
        observedAt: request.observedAt,
      },
      nextAttemptAt: request.nextAttemptAt,
    });
    await this.artifacts.putVerificationRuns(
      job.taskId,
      executions.map((execution) => execution.run),
    );

    let review: Awaited<ReturnType<IndependentReviewCoordinator["review"]>> | undefined;
    if (prepared.contract.risk !== "low") {
      const diff = await integrator.diff(workspace);
      if (Buffer.byteLength(diff) > maxReviewPacketBytes) {
        throw new Error(`Independent review packet exceeds ${maxReviewPacketBytes} bytes`);
      }
      const reviewEvidenceIds = uniqueSorted([
        ...prepared.snapshot.requiredEvidenceIds,
        ...bundles.flatMap((bundle) => bundle.evidenceIds),
      ]);
      review = await new IndependentReviewCoordinator(this.runtimes).review({
        contract: prepared.contract,
        snapshot: prepared.snapshot,
        workspacePath: workspace.workspacePath,
        codeRevision: integrated.revision,
        authorProviders: plan.authorProviders,
        eligibleProviders: reviewEligibleProviders(
          current,
          prepared.snapshot.normativeRevisions,
          reviewEvidenceIds,
        ),
        changedSymbolIds: plan.changedSymbolIds,
        changedPaths: plan.changedPaths,
        allowedRuleRefs: [
          ...prepared.contract.acceptance.map((criterion) => criterion.criterionId),
          ...prepared.snapshot.normativeRevisions.flatMap((revision) => [
            revision.recordId,
            revision.revisionId,
          ]),
        ],
        allowedEvidenceIds: reviewEvidenceIds,
        reviewPacket: JSON.stringify({
          taskContract: prepared.contract,
          taskContextSnapshot: prepared.snapshot,
          normativeRecords: current.normativeRecords.filter((record) =>
            prepared.snapshot.normativeRevisions.some(
              (revision) => revision.revisionId === record.revision.revisionId,
            ),
          ),
          evidence: current.evidence.filter((item) => reviewEvidenceIds.includes(item.evidenceId)),
          changeBundles: plan.orderedChangeBundles,
          deterministicVerificationRuns: executions.map((execution) => execution.run),
          diff,
        }),
        reviewedAt: request.observedAt,
      });
      await this.artifacts.putReviewFindings(job.taskId, review.findings);
      await this.artifacts.putVerificationRuns(job.taskId, [review.verificationRun]);
    }

    const state: IntegratedTaskState = {
      taskId: job.taskId,
      scheduleJobId: job.jobId,
      repositoryPath,
      workspacePath: workspace.workspacePath,
      baseRevision: workspace.baseRevision,
      gitCommit: gitResult.gitCommit,
      resultRevision: integrated.revision,
      contextDigest: prepared.snapshot.contextDigest,
      changeBundleIds: plan.orderedChangeBundles.map((bundle) => bundle.bundleId),
      workItemIds: plan.orderedChangeBundles.map((bundle) => bundle.workItemId),
      changedPaths: plan.changedPaths,
      changedSymbolIds: plan.changedSymbolIds,
      authorProviders: plan.authorProviders,
      createdAt: this.now().toISOString(),
    };
    await this.integratedTasks.put(state);
    return { state, executions, review };
  }
}

function logicWorkItem(
  taskId: string,
  plan: {
    readonly workItemId: string;
    readonly plannedSymbolIds: readonly string[];
    readonly allowedPaths: readonly string[];
    readonly dependsOn?: readonly string[];
    readonly requiredVerifiers?: readonly VerifierRef[];
    readonly capabilityId?: string;
  },
): LogicWorkItem {
  return {
    workItemId: plan.workItemId,
    taskId,
    plannedSymbolIds: plan.plannedSymbolIds,
    allowedPaths: plan.allowedPaths,
    dependsOn: plan.dependsOn ?? [],
    requiredVerifiers: plan.requiredVerifiers ?? [],
    capabilityId:
      plan.capabilityId ??
      `capability:${createHash("sha256")
        .update(JSON.stringify([taskId, plan.workItemId, plan.plannedSymbolIds, plan.allowedPaths]))
        .digest("hex")}`,
  };
}

function boundInvariantVerifiers(
  current: Awaited<ReturnType<TaskContextStateProvider["getCurrent"]>>,
  revisions: readonly { readonly kind: string; readonly revisionId: string }[],
): readonly VerifierRef[] {
  const revisionIds = new Set(
    revisions
      .filter((revision) => revision.kind === "invariant")
      .map((revision) => revision.revisionId),
  );
  return current.normativeRecords.flatMap((record) =>
    record.revision.kind === "invariant" && revisionIds.has(record.revision.revisionId)
      ? record.revision.verifiers
      : [],
  );
}

function requireSource(values: ReadonlyMap<string, string>, workItemId: string): string {
  const value = values.get(workItemId);
  if (!value) throw new Error(`Missing source worktree for ${workItemId}`);
  return value;
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return JSON.stringify(uniqueSorted(left)) === JSON.stringify(uniqueSorted(right));
}

function uniqueSorted(values: readonly string[]): string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}

function reviewEligibleProviders(
  current: Awaited<ReturnType<TaskContextStateProvider["getCurrent"]>>,
  revisions: readonly { readonly revisionId: string }[],
  evidenceIds: readonly string[],
): readonly ("codex" | "claude")[] {
  const allowed = new Set<"codex" | "claude">(["codex", "claude"]);
  const revisionIds = new Set(revisions.map((revision) => revision.revisionId));
  for (const record of current.normativeRecords) {
    if (!revisionIds.has(record.revision.revisionId)) continue;
    for (const provider of [...allowed]) {
      if (!record.revision.egress.allowedRuntimeProviders.includes(provider)) {
        allowed.delete(provider);
      }
    }
  }
  const evidence = new Map(current.evidence.map((item) => [item.evidenceId, item] as const));
  for (const evidenceId of evidenceIds) {
    const item = evidence.get(evidenceId);
    if (!item || item.availability !== "current") return [];
    for (const provider of [...allowed]) {
      if (!item.allowedRuntimeProviders.includes(provider)) allowed.delete(provider);
    }
  }
  return Array.from(allowed).sort();
}
