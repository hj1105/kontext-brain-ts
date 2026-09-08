import { realpath } from "node:fs/promises";
import type { CurrentTaskContextState, PreparedTaskContext } from "@kontext-brain/context";
import { FileQuarantineStore, FileTaskCompletionArtifactStore } from "@kontext-brain/local";
import type { InvariantEvaluation, VerificationRun } from "@kontext-brain/spec";
import { z } from "zod";
import { FileIntegratedTaskStateStore } from "./file-integrated-task-state-store.js";
import {
  type LocalKontextCompletionOperations,
  assessCurrentContext,
} from "./local-completion-operations.js";
import { requireLocalTaskOwner } from "./local-task-owner.js";
import { inspectTaskWorkspace } from "./local-task-preparation.js";
import { RegisteredTaskContextProvider } from "./registered-task-context.js";
import { taskPlanningDigest } from "./task-planning-contract.js";
import { captureWorkspaceSnapshot } from "./workspace-change-observer.js";

export const taskCompletionAssessmentRequestSchema = z
  .object({
    taskId: z.string().trim().min(1),
    jobId: z.string().trim().min(1),
  })
  .strict();

/** Host-derived evidence only; completion assessment never grants owner approval. */
export class LocalTaskCompletionAssessment {
  constructor(
    private readonly dataDirectory: string,
    private readonly completion: Pick<LocalKontextCompletionOperations, "proposeTransition">,
    private readonly environment: NodeJS.ProcessEnv = process.env,
  ) {}

  async assess(input: unknown) {
    const request = taskCompletionAssessmentRequestSchema.parse(input);
    const { repository, registration } = await requireLocalTaskOwner(
      this.dataDirectory,
      request.taskId,
    );
    const prepared = await repository.get(request.taskId);
    const store = new FileIntegratedTaskStateStore(this.dataDirectory);
    const integration = await store.get(request.taskId);
    if (!prepared || !integration || integration.scheduleJobId !== request.jobId)
      throw new Error("This Task and schedule have no matching sidecar integration");
    if (
      (await realpath(integration.repositoryPath)) !==
      (await realpath(registration.owner.workspacePath))
    )
      throw new Error("Integrated repository does not match the registered Task workspace");
    const assertCode = async () => {
      await inspectTaskWorkspace(
        this.dataDirectory,
        {
          workspacePath: integration.workspacePath,
          expectedCodeRevision: integration.gitCommit,
        },
        this.environment,
      );
      const observed = await captureWorkspaceSnapshot(
        integration.workspacePath,
        integration.changedPaths,
      );
      if (
        observed.revision !== integration.resultRevision ||
        integration.contextDigest !== prepared.snapshot.contextDigest ||
        JSON.stringify(await store.get(request.taskId)) !== JSON.stringify(integration)
      )
        throw new Error("Integrated code or context changed; completion must be reassessed");
      const quarantine = await new FileQuarantineStore(this.dataDirectory).list("active");
      if (
        quarantine.some(
          (record) =>
            record.taskId === request.taskId ||
            (record.taskId === undefined && record.codeRevision === integration.resultRevision),
        )
      )
        throw new Error("Active quarantine prevents completion assessment");
    };
    await assertCode();
    const currentProvider = new RegisteredTaskContextProvider(this.dataDirectory, repository);
    const current = await currentProvider.getCurrent(request.taskId);
    const artifacts = new FileTaskCompletionArtifactStore(this.dataDirectory);
    const runs = await artifacts.listVerificationRuns(request.taskId);
    const auditInputs = async (verificationRuns: typeof runs) =>
      JSON.stringify({
        verificationRuns: verificationRuns.filter(
          (run) =>
            !(
              run.tier === "full" &&
              run.verifierKind === "query" &&
              run.verifierRef === "kontext:manifest-audit"
            ),
        ),
        changeBundles: await artifacts.listChangeBundles(request.taskId),
        reviewFindings: await artifacts.listReviewFindings(request.taskId),
      });
    const initialAuditInputs = await auditInputs(runs);
    const completionBasisDigest = taskPlanningDigest({
      taskId: request.taskId,
      jobId: request.jobId,
      owner: registration.owner,
      contract: prepared.contract,
      snapshot: prepared.snapshot,
      sourceFreshnessDigest: current.sourceFreshnessDigest,
      integration,
      artifacts: JSON.parse(initialAuditInputs),
    });
    const observedAt = new Date().toISOString();
    const invariantEvaluations = evaluateBoundCompletionInvariants(
      prepared,
      current,
      runs,
      integration.resultRevision,
    );
    const result = await this.completion.proposeTransition({
      taskId: request.taskId,
      currentState: "in_progress",
      workStarted: true,
      completionRequested: true,
      context: assessCurrentContext(prepared, current),
      evidence: [
        {
          kind: "commit",
          ref: integration.gitCommit,
          codeRevision: integration.resultRevision,
          contextDigest: prepared.snapshot.contextDigest,
          observedAt,
        },
      ],
      invariantEvaluations,
      requestedAt: observedAt,
    });
    await assertCode();
    const after = await currentProvider.getCurrent(request.taskId);
    const latestPrepared = await repository.get(request.taskId);
    const latestRuns = await artifacts.listVerificationRuns(request.taskId);
    if (
      initialAuditInputs !== (await auditInputs(latestRuns)) ||
      after.sourceFreshnessDigest !== current.sourceFreshnessDigest ||
      JSON.stringify(latestPrepared) !== JSON.stringify(prepared) ||
      JSON.stringify(result.integration) !== JSON.stringify(integration) ||
      (result.state === "done" && assessCurrentContext(prepared, after).status !== "current")
    )
      throw new Error("Completion inputs changed during assessment; request a fresh assessment");
    return {
      taskId: request.taskId,
      jobId: request.jobId,
      completionBasisDigest,
      observedAt,
      risk: prepared.contract.risk,
      state: result.state,
      issues: result.issues,
      context: result.context,
      gitCommit: integration.gitCommit,
      codeRevision: integration.resultRevision,
      workspacePath: integration.workspacePath,
      invariantEvaluations,
      verificationRuns: latestRuns
        .filter(
          (run) =>
            run.codeRevision === integration.resultRevision &&
            run.contextDigest === prepared.snapshot.contextDigest,
        )
        .map(({ verificationRunId, tier, verifierKind, verifierRef, result, observedAt }) => ({
          verificationRunId,
          tier,
          verifierKind,
          verifierRef,
          result,
          observedAt,
        })),
      accuracyManifest: result.accuracyManifest,
      accuracyManifestError: result.accuracyManifestError,
    };
  }
}

export function evaluateBoundCompletionInvariants(
  prepared: PreparedTaskContext,
  current: CurrentTaskContextState,
  runs: readonly VerificationRun[],
  codeRevision: string,
): InvariantEvaluation[] {
  const currentRuns = runs.filter(
    (run) =>
      run.tier === "full" &&
      run.codeRevision === codeRevision &&
      run.contextDigest === prepared.snapshot.contextDigest,
  );
  return prepared.snapshot.normativeRevisions.flatMap((ref): InvariantEvaluation[] => {
    if (ref.kind !== "invariant") return [];
    const record = current.normativeRecords.find(
      ({ revision }) =>
        revision.kind === ref.kind &&
        revision.recordId === ref.recordId &&
        revision.revisionId === ref.revisionId,
    );
    if (record?.revision.kind !== "invariant") return [];
    const groups = record.revision.verifiers.map((verifier) =>
      currentRuns.filter(
        (run) => run.verifierKind === verifier.kind && run.verifierRef === verifier.ref,
      ),
    );
    const matching = groups.flat();
    const status = matching.some((run) => run.result === "failed")
      ? "violated"
      : matching.some((run) => run.result === "inconclusive")
        ? "inconclusive"
        : groups.length === 0 ||
            groups.some((group) => !group.some((run) => run.result === "passed"))
          ? "unguarded"
          : "guarded";
    return [
      {
        invariantId: ref.recordId,
        revisionId: ref.revisionId,
        status,
        verificationRunIds: matching.map((run) => run.verificationRunId),
      },
    ];
  });
}
