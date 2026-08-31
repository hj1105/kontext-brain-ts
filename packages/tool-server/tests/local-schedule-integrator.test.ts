import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  InMemoryPreparedTaskContextStore,
  InMemoryTaskContextStateProvider,
  TaskContextWorkflow,
} from "@kontext-brain/context";
import {
  FileQuarantineStore,
  FileTaskCompletionArtifactStore,
  GitRuntimeWorktreeManager,
} from "@kontext-brain/local";
import {
  DurableVerificationCoordinator,
  InMemoryVerificationRetryQueue,
  VerificationCoordinator,
  VerifierRegistry,
} from "@kontext-brain/orchestrator";
import { createChangeBundle } from "@kontext-brain/spec";
import { afterEach, describe, expect, it } from "vitest";
import {
  BoundWorkspaceChangeEvidenceProvider,
  FileIntegratedTaskStateStore,
  InMemoryWriteAuthorizationBindingStore,
  KontextTaskWorkflowToolRouter,
  LocalKontextCompletionOperations,
  LocalScheduleIntegrator,
  captureWorkspaceCodeSymbols,
  captureWorkspaceSnapshot,
} from "../src/index.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("LocalScheduleIntegrator", () => {
  it("revalidates an accepted worker bundle, integrates it, and verifies one final revision", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "kontext-schedule-integration-"));
    temporaryDirectories.push(root);
    const repositoryPath = path.join(root, "repository");
    await mkdir(path.join(repositoryPath, "src"), { recursive: true });
    await writeFile(
      path.join(repositoryPath, "src/handler.ts"),
      "export function handler(value: number): number { return value + 1; }\n",
    );
    initializeGit(repositoryPath);
    const baseRevision = git(repositoryPath, ["rev-parse", "HEAD"]).trim();
    const taskId = "task:integrate-schedule";
    const workItemId = "work-item:handler";
    const source = await new GitRuntimeWorktreeManager(
      repositoryPath,
      path.join(root, "worker-worktrees"),
    ).prepare({
      taskId,
      workItem: {
        workItemId,
        taskId,
        plannedSymbolIds: ["placeholder"],
        dependsOn: [],
        allowedPaths: ["src/handler.ts"],
        requiredVerifiers: [{ kind: "test", ref: "handler:test" }],
        capabilityId: "capability:handler",
      },
      baseRevision,
    });
    const sourceSnapshot = await captureWorkspaceSnapshot(source.workspacePath, ["src/handler.ts"]);
    const sourceSymbols = await captureWorkspaceCodeSymbols(
      source.workspacePath,
      ["src/handler.ts"],
      sourceSnapshot,
    );
    const symbol = sourceSymbols.symbols.find(
      (candidate) => candidate.identity.qualifiedName === "handler",
    );
    if (!symbol) throw new Error("expected handler Code Symbol");
    const plannedSymbolId = "planned-symbol:handler";
    const plannedSymbols = [
      {
        plannedSymbolId,
        taskId,
        intendedIdentity: {
          relativePath: symbol.identity.relativePath,
          language: symbol.identity.language,
          kind: symbol.identity.kind,
          qualifiedName: symbol.identity.qualifiedName,
          signatureDiscriminator: symbol.identity.signatureDiscriminator,
        },
        responsibility: "Increment the handler result twice.",
      },
    ];

    const current = new InMemoryTaskContextStateProvider();
    current.set(taskId, {
      codeRevision: baseRevision,
      sourceFreshnessDigest: "freshness:current",
      effectiveScopes: [{ kind: "personal", subjectId: "user:test" }],
      normativeRecords: [],
      normativeRevisionCatalog: [],
      conflicts: [],
      evidence: [],
      logicPlans: [
        {
          workItemId,
          plannedSymbolIds: [plannedSymbolId],
          plannedSymbols,
          allowedPaths: ["src/handler.ts"],
          dependsOn: [],
          requiredVerifiers: [{ kind: "test", ref: "handler:test" }],
          capabilityId: "capability:handler",
        },
      ],
    });
    const prepared = new InMemoryPreparedTaskContextStore();
    const workflow = new TaskContextWorkflow(current, prepared);
    const contract = {
      taskId,
      intent: "Integrate one independently implemented handler change.",
      acceptance: [
        {
          criterionId: "acceptance:handler",
          statement: "The handler verifier passes.",
          verifier: { kind: "test" as const, ref: "handler:test" },
        },
      ],
      nonGoals: [],
      targets: [symbol.symbolId],
      risk: "low" as const,
    };
    const preparedTask = await workflow.prepareTask({
      contract,
      createdAt: "2026-08-31T04:00:00.000Z",
    });
    const bindings = new InMemoryWriteAuthorizationBindingStore();
    const workflowRouter = new KontextTaskWorkflowToolRouter(
      workflow,
      () => new Date("2026-08-31T04:01:00.000Z"),
      bindings,
    );
    const compiled = await workflowRouter.beginLogic({
      taskId,
      workspacePath: source.workspacePath,
      logic: { workItemId, plannedSymbolIds: [plannedSymbolId] },
      runtimeProvider: "codex",
      receiptTtlSeconds: 900,
      totalTokenBudget: 20_000,
      optionalEvidenceTokenBudget: 2_000,
    });
    if (!compiled.receipt) throw new Error("expected Context Receipt");
    await writeFile(
      path.join(source.workspacePath, "src/handler.ts"),
      "export function handler(value: number): number { return value + 2; }\n",
    );

    const artifacts = new FileTaskCompletionArtifactStore(path.join(root, "data"));
    const quarantine = new FileQuarantineStore(path.join(root, "data"));
    const verification = passingVerification();
    const changeEvidence = new BoundWorkspaceChangeEvidenceProvider(bindings);
    const integratedTasks = new FileIntegratedTaskStateStore(path.join(root, "data"));
    const diagnosticEvidence = await changeEvidence.observe({
      workspacePath: source.workspacePath,
      taskId,
      workItem: {
        workItemId,
        taskId,
        plannedSymbolIds: [plannedSymbolId],
        dependsOn: [],
        allowedPaths: ["src/handler.ts"],
        requiredVerifiers: [{ kind: "test", ref: "handler:test" }],
        capabilityId: "capability:handler",
      },
      plannedSymbols,
    });
    expect(diagnosticEvidence.plannedSymbolIssues).toEqual([]);
    const completion = new LocalKontextCompletionOperations(
      current,
      prepared,
      artifacts,
      quarantine,
      verification,
      changeEvidence,
      integratedTasks,
    );
    for (const tier of ["fast", "targeted"] as const) {
      await completion.checkChange({
        taskId,
        workItemId,
        workspacePath: source.workspacePath,
        tier,
        observedAt: "2026-08-31T04:02:00.000Z",
        nextAttemptAt: "2026-08-31T04:05:00.000Z",
      });
    }
    const observed = await changeEvidence.observe({
      workspacePath: source.workspacePath,
      taskId,
      workItem: {
        workItemId,
        taskId,
        plannedSymbolIds: [plannedSymbolId],
        dependsOn: [],
        allowedPaths: ["src/handler.ts"],
        requiredVerifiers: [{ kind: "test", ref: "handler:test" }],
        capabilityId: "capability:handler",
      },
      plannedSymbols,
    });
    const bundle = createChangeBundle({
      taskId,
      workItemId,
      baseRevision,
      resultRevision: observed.currentCodeRevision,
      taskContextDigest: preparedTask.snapshot.contextDigest,
      patchDigest: observed.observedPatch.patchDigest,
      changedSymbolIds: observed.observedPatch.changedSymbolIds,
      changedPaths: observed.observedPatch.changedPaths,
      contextReceiptIds: observed.receipts.map((receipt) => receipt.receiptId),
      evidenceIds: [],
      normativeRevisions: [],
      verificationRunIds: (await artifacts.listVerificationRuns(taskId)).map(
        (run) => run.verificationRunId,
      ),
      proposals: [],
      unresolved: [],
      submittedAt: "2026-08-31T04:03:00.000Z",
    });
    const { bundleId: _bundleId, ...bundleDraft } = bundle;
    const submitted = (await completion.submitChangeBundle({
      workspacePath: source.workspacePath,
      bundle: bundleDraft,
    })) as { readonly accepted: boolean; readonly issues: readonly unknown[] };
    expect(submitted.accepted, JSON.stringify(submitted.issues)).toBe(true);

    const integration = (await new LocalScheduleIntegrator(
      current,
      prepared,
      artifacts,
      quarantine,
      verification,
      changeEvidence,
      integratedTasks,
      [],
      path.join(root, "data"),
      () => new Date("2026-08-31T04:04:00.000Z"),
    ).integrate(
      {
        jobId: "runtime-schedule:one",
        taskId,
        repositoryPath,
        codeRevision: baseRevision,
        contextDigest: preparedTask.snapshot.contextDigest,
        status: "completed",
        requestedAt: "2026-08-31T04:01:00.000Z",
        result: {
          capabilities: [],
          results: [
            {
              workItemId,
              status: "completed",
              provider: "codex",
              worktree: source,
              attempts: 1,
              checkpoints: [],
              diagnostics: [],
            },
          ],
        },
      },
      {
        jobId: "runtime-schedule:one",
        observedAt: "2026-08-31T04:04:00.000Z",
        nextAttemptAt: "2026-08-31T04:10:00.000Z",
      },
    )) as { readonly state: { readonly workspacePath: string; readonly resultRevision: string } };

    expect(
      await readFile(path.join(integration.state.workspacePath, "src/handler.ts"), "utf8"),
    ).toContain("value + 2");
    expect(await integratedTasks.get(taskId)).toMatchObject({
      resultRevision: integration.state.resultRevision,
      changeBundleIds: [bundle.bundleId],
      authorProviders: ["codex"],
    });
    expect(
      (await artifacts.listVerificationRuns(taskId)).filter(
        (run) => run.tier === "full" && run.codeRevision === integration.state.resultRevision,
      ),
    ).toHaveLength(5);
    const transition = (await completion.proposeTransition({
      taskId,
      currentState: "in_progress",
      workStarted: true,
      completionRequested: true,
      context: { status: "current", contextDigest: preparedTask.snapshot.contextDigest },
      evidence: [
        {
          kind: "commit",
          ref: "commit:integration",
          codeRevision: integration.state.resultRevision,
          contextDigest: preparedTask.snapshot.contextDigest,
          observedAt: "2026-08-31T04:05:00.000Z",
        },
      ],
      invariantEvaluations: [],
      requestedAt: "2026-08-31T04:05:00.000Z",
    })) as { readonly state: string; readonly issues: readonly unknown[] };
    expect(transition.state, JSON.stringify(transition.issues)).toBe("done");
  });
});

function passingVerification(): DurableVerificationCoordinator {
  const registry = new VerifierRegistry();
  for (const kind of ["test", "typecheck", "build", "lint", "query", "manual_review"] as const) {
    registry.registerFallback(kind, {
      execute: async () => ({ result: "passed", output: { kind } }),
    });
  }
  return new DurableVerificationCoordinator(
    new VerificationCoordinator(registry),
    new InMemoryVerificationRetryQueue(),
  );
}

function initializeGit(repositoryPath: string): void {
  git(repositoryPath, ["init", "-q"]);
  git(repositoryPath, ["add", "."]);
  git(repositoryPath, [
    "-c",
    "user.name=Kontext Test",
    "-c",
    "user.email=kontext@example.invalid",
    "commit",
    "-qm",
    "baseline",
  ]);
}

function git(cwd: string, args: readonly string[]): string {
  return execFileSync("git", args, { cwd, encoding: "utf8" });
}
