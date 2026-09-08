import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  FileQuarantineStore,
  FileTaskCompletionArtifactStore,
  FileTaskContextRepository,
} from "@kontext-brain/local";
import {
  DurableVerificationCoordinator,
  InMemoryVerificationRetryQueue,
  VerificationCoordinator,
  VerifierRegistry,
  createFullVerificationPlan,
  createQuarantineRecord,
  createVerificationRun,
} from "@kontext-brain/orchestrator";
import { createChangeBundle } from "@kontext-brain/spec";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { afterEach, expect, it, vi } from "vitest";
import { FileIntegratedTaskStateStore } from "../src/file-integrated-task-state-store.js";
import { LocalKontextCompletionOperations } from "../src/local-completion-operations.js";
import { LocalKnowledgeOperations } from "../src/local-knowledge-operations.js";
import {
  LocalTaskCompletionAssessment,
  evaluateBoundCompletionInvariants,
} from "../src/local-task-completion-assessment.js";
import { LocalTaskCreationOperations } from "../src/local-task-creation.js";
import { LocalTaskFinalizationOperations } from "../src/local-task-finalization.js";
import { LocalTaskInventoryOperations } from "../src/local-task-inventory.js";
import { RegisteredTaskContextProvider } from "../src/registered-task-context.js";
import { captureWorkspaceSnapshot } from "../src/workspace-change-observer.js";

const directories: string[] = [];
afterEach(async () => {
  await Promise.all(directories.splice(0).map((dir) => rm(dir, { recursive: true })));
});
async function fixture(risk: "low" | "medium" = "low") {
  const root = await mkdtemp(path.join(tmpdir(), "kontext-completion-assessment-"));
  directories.push(root);
  const workspacePath = path.join(root, "workspace");
  const data = path.join(root, "data");
  const home = path.join(root, "home");
  await Promise.all([workspacePath, home].map((dir) => mkdir(dir)));
  const environment = {
    PATH: process.env.PATH,
    HOME: home,
    USERPROFILE: home,
    GIT_CONFIG_NOSYSTEM: "1",
    XDG_CONFIG_HOME: home,
    SYSTEMROOT: process.env.SYSTEMROOT,
  };
  const git = (...args: string[]) =>
    execFileSync("git", args, { cwd: workspacePath, env: environment, encoding: "utf8" }).trim();
  git("init", "--quiet", `--template=${home}`);
  await writeFile(path.join(workspacePath, "index.ts"), "export function total() { return 1; }\n");
  git("add", "index.ts");
  git(
    "-c",
    "user.name=Fixture",
    "-c",
    "user.email=fixture@example.invalid",
    "-c",
    "commit.gpgsign=false",
    "commit",
    "--no-verify",
    "--quiet",
    "-m",
    "fixture",
  );
  const gitCommit = git("rev-parse", "HEAD");
  const knowledge = new LocalKnowledgeOperations(data);
  await writeFile(path.join(root, "notes.md"), "## Terminology\nKeep the existing total term.\n");
  const source = await knowledge.registerMarkdownSource({
    workspacePath: root,
    relativePath: "notes.md",
  });
  await knowledge.setSourceSharing({
    resourceId: source.resourceId,
    expectedRevision: 1,
    expectedContentHash: source.contentHash,
    dataClassification: "internal",
    allowedRuntimeProviders: ["codex"],
  });
  const creation = await new LocalTaskCreationOperations(data, environment).createTask({
    requestId: "5825c99a-805b-4ffc-b0f1-bd4c812e900a",
    workspacePath,
    workspaceId: "workspace:test",
    expectedCodeRevision: gitCommit,
    sourceResourceIds: [source.resourceId],
    contract: {
      intent: "Verify total",
      risk,
      targets: ["symbol:total"],
      nonGoals: [],
      acceptance: [
        {
          criterionId: "total",
          statement: "Total works",
          verifier: { kind: "test", ref: "workspace:test" },
        },
      ],
    },
    logicPlans: [
      { workItemId: "logic:total", plannedSymbolIds: ["symbol:total"], allowedPaths: ["index.ts"] },
    ],
  });
  const taskId = creation.taskId;
  const repository = new FileTaskContextRepository(data);
  const prepared = await repository.get(taskId);
  if (!prepared) throw new Error("Missing fixture Task");
  const current = new RegisteredTaskContextProvider(data, repository);
  const artifacts = new FileTaskCompletionArtifactStore(data);
  const quarantine = new FileQuarantineStore(data);
  const integrations = new FileIntegratedTaskStateStore(data);
  const revision = (await captureWorkspaceSnapshot(workspacePath, ["index.ts"])).revision;
  const observedAt = new Date().toISOString();
  const runs = createFullVerificationPlan({ contract: prepared.contract }).requirements.map(
    (requirement) =>
      createVerificationRun(
        requirement,
        { codeRevision: revision, contextDigest: prepared.snapshot.contextDigest, observedAt },
        "passed",
      ),
  );
  // Seed trusted verification artifacts; this fixture never executes project commands or a model.
  await artifacts.putVerificationRuns(taskId, runs);
  const bundle = createChangeBundle({
    taskId,
    workItemId: "logic:total",
    baseRevision: gitCommit,
    resultRevision: revision,
    taskContextDigest: prepared.snapshot.contextDigest,
    patchDigest: "sha256:fixture",
    changedSymbolIds: ["symbol:total"],
    changedPaths: ["index.ts"],
    contextReceiptIds: ["receipt:fixture"],
    evidenceIds: prepared.snapshot.requiredEvidenceIds,
    normativeRevisions: [],
    verificationRunIds: [],
    proposals: [],
    unresolved: [],
    submittedAt: observedAt,
  });
  await artifacts.putChangeBundle(bundle);
  const integration = await integrations.put({
    taskId,
    scheduleJobId: "job:fixture",
    repositoryPath: workspacePath,
    workspacePath,
    baseRevision: gitCommit,
    gitCommit,
    resultRevision: revision,
    contextDigest: prepared.snapshot.contextDigest,
    changeBundleIds: [bundle.bundleId],
    workItemIds: ["logic:total"],
    changedPaths: ["index.ts"],
    changedSymbolIds: ["symbol:total"],
    authorProviders: ["codex"],
    createdAt: observedAt,
  });
  const completion = new LocalKontextCompletionOperations(
    current,
    repository,
    artifacts,
    quarantine,
    new DurableVerificationCoordinator(
      new VerificationCoordinator(new VerifierRegistry()),
      new InMemoryVerificationRetryQueue(),
    ),
    {
      observe: async () => {
        throw new Error("Must not invoke worker verification");
      },
    },
    integrations,
  );
  const assessment = new LocalTaskCompletionAssessment(data, completion, environment);
  return {
    root,
    data,
    taskId,
    repository,
    prepared,
    current,
    artifacts,
    quarantine,
    integrations,
    completion,
    assessment,
    integration,
    runs,
    git,
    request: { taskId, jobId: integration.scheduleJobId },
  };
}

it("derives a real integrated commit and invokes the existing manifest/transition audit", async () => {
  const h = await fixture();
  const result = await h.assessment.assess(h.request);
  expect(result).toMatchObject({
    taskId: h.taskId,
    jobId: "job:fixture",
    state: "done",
    issues: [],
    gitCommit: h.integration.gitCommit,
    context: { status: "current" },
  });
  expect(result.accuracyManifest?.manifestId).toMatch(/^accuracy-manifest:/);
  expect(await h.artifacts.getAccuracyManifest(h.taskId)).toEqual(result.accuracyManifest);
  expect(
    result.verificationRuns.some(
      (run) => run.verifierRef === "kontext:manifest-audit" && run.result === "passed",
    ),
  ).toBe(true);
  expect(JSON.stringify(result)).not.toContain("export function total");
});

it("does not synthesize Code Owner approval from a passing review or local ownership", async () => {
  const h = await fixture("medium");
  const result = await h.assessment.assess(h.request);
  expect(result.state).not.toBe("done");
  expect(result.issues).toContainEqual(
    expect.objectContaining({ code: "missing_code_owner_approval" }),
  );
  await expect(
    h.assessment.assess({ ...h.request, evidence: [{ kind: "approval" }] }),
  ).rejects.toThrow();
});

it("rejects wrong Task/schedule, dirty or different commits, and does not reset an earlier verdict", async () => {
  const h = await fixture();
  await h.assessment.assess(h.request);
  await expect(h.assessment.assess({ ...h.request, taskId: "task:other" })).rejects.toThrow(
    "principal",
  );
  await expect(h.assessment.assess({ ...h.request, jobId: "other" })).rejects.toThrow(
    "matching sidecar integration",
  );
  await writeFile(path.join(h.integration.workspacePath, "index.ts"), "changed\n");
  await expect(h.assessment.assess(h.request)).rejects.toThrow("clean Git commit");
  h.git("add", "index.ts");
  h.git(
    "-c",
    "user.name=Fixture",
    "-c",
    "user.email=fixture@example.invalid",
    "-c",
    "commit.gpgsign=false",
    "commit",
    "--no-verify",
    "--quiet",
    "-m",
    "changed",
  );
  await expect(h.assessment.assess(h.request)).rejects.toThrow("revision changed");
});

it("refuses quarantine and invalidates a result if code changes during the audit", async () => {
  const h = await fixture();
  const original = h.completion.proposeTransition.bind(h.completion);
  vi.spyOn(h.completion, "proposeTransition").mockImplementation(async (request) => {
    const result = await original(request);
    await writeFile(path.join(h.integration.workspacePath, "untracked.md"), "changed\n");
    return result;
  });
  await expect(h.assessment.assess(h.request)).rejects.toThrow("clean Git commit");
  const q = await fixture();
  await q.quarantine.put(
    createQuarantineRecord({
      taskId: q.taskId,
      codeRevision: q.integration.resultRevision,
      paths: ["index.ts"],
      symbolIds: ["symbol:total"],
      reasons: ["unobserved_write"],
      observedAt: new Date().toISOString(),
    }),
  );
  await expect(q.assessment.assess(q.request)).rejects.toThrow("quarantine");
});

it("rechecks actual Markdown provenance and refuses completion after its content changes", async () => {
  const h = await fixture();
  expect((await h.assessment.assess(h.request)).state).toBe("done");
  await writeFile(
    path.join(h.root, "notes.md"),
    "## Terminology\nA new decision supersedes the previous context.\n",
  );
  const result = await h.assessment.assess(h.request);
  expect(result.state).toBe("blocked");
  expect(result.context.status).not.toBe("current");
});

it("refuses a successful observation when verification evidence changes during the audit", async () => {
  const h = await fixture();
  const original = h.completion.proposeTransition.bind(h.completion);
  vi.spyOn(h.completion, "proposeTransition").mockImplementation(async (request) => {
    const result = await original(request);
    await h.artifacts.putVerificationRuns(h.taskId, [
      createVerificationRun(
        {
          tier: "full",
          verifier: { kind: "test", ref: "workspace:test" },
          subjectIds: [h.taskId],
        },
        {
          codeRevision: h.integration.resultRevision,
          contextDigest: h.prepared.snapshot.contextDigest,
          observedAt: new Date().toISOString(),
        },
        "failed",
      ),
    ]);
    return result;
  });
  await expect(h.assessment.assess(h.request)).rejects.toThrow("inputs changed");
});

it("assesses through the actual bundled host MCP and refuses unauthorized callers", async () => {
  const h = await fixture();
  const client = new Client({ name: "completion-fixture", version: "1" });
  const hostToken = "c".repeat(64);
  try {
    await client.connect(
      new StdioClientTransport({
        command: process.execPath,
        args: [path.resolve("plugins/kontext-brain/server.mjs")],
        cwd: h.root,
        env: {
          KONTEXT_PLUGIN_DATA: h.data,
          KONTEXT_HOST_MANAGEMENT_TOKEN: hostToken,
          HOME: path.join(h.root, "home"),
          USERPROFILE: path.join(h.root, "home"),
          XDG_CONFIG_HOME: path.join(h.root, "home"),
          PATH: process.env.PATH ?? "",
          ...(process.env.SYSTEMROOT ? { SYSTEMROOT: process.env.SYSTEMROOT } : {}),
        },
        stderr: "pipe",
      }),
    );
    const denied = await client.callTool({
      name: "kontext_assess_completion",
      arguments: { ...h.request, hostToken: "d".repeat(64) },
    });
    expect(denied.isError).toBe(true);
    const result = await client.callTool({
      name: "kontext_assess_completion",
      arguments: { ...h.request, hostToken },
    });
    expect(result.isError).not.toBe(true);
    expect(result.structuredContent).toMatchObject({
      taskId: h.taskId,
      jobId: h.request.jobId,
      state: "done",
      gitCommit: h.integration.gitCommit,
    });
    expect(JSON.stringify(result)).not.toContain(hostToken);
    expect(JSON.stringify(result)).not.toContain("Keep the existing total term");
    const basis = (await h.assessment.assess(h.request)).completionBasisDigest;
    const finalized = await client.callTool({
      name: "kontext_finalize_task",
      arguments: {
        ...h.request,
        hostToken,
        requestId: "c6f9ef06-51df-4e7d-a6c9-4a52b4c7e941",
        expectedCompletionBasisDigest: basis,
      },
    });
    expect(finalized.isError).not.toBe(true);
    expect(finalized.structuredContent).toMatchObject({
      created: true,
      record: { request: { taskId: h.taskId }, accuracyManifest: { taskId: h.taskId } },
      currentEvidence: "validated_at_recording",
    });
    const inspected = await client.callTool({
      name: "kontext_inspect_finalization",
      arguments: { taskId: h.taskId, hostToken },
    });
    expect(inspected.isError).not.toBe(true);
    expect(inspected.structuredContent).toMatchObject({
      taskId: h.taskId,
      currentEvidence: "not_revalidated",
    });
    const recordId = (
      await new LocalTaskFinalizationOperations(h.data, h.assessment).inspect({ taskId: h.taskId })
    ).record?.recordId;
    const revalidated = await client.callTool({
      name: "kontext_revalidate_finalization",
      arguments: { taskId: h.taskId, expectedRecordId: recordId, hostToken },
    });
    expect(revalidated.isError).not.toBe(true);
    expect(revalidated.structuredContent).toMatchObject({
      taskId: h.taskId,
      recordId,
      currentEvidence: "revalidated_current",
      state: "done",
    });
    const forbidden = await client.callTool({
      name: "kontext_revalidate_finalization",
      arguments: { taskId: h.taskId, expectedRecordId: recordId, hostToken: "d".repeat(64) },
    });
    expect(forbidden.isError).toBe(true);
    expect(h.git("status", "--porcelain=v1")).toBe("");
  } finally {
    await client.close();
  }
});

it("binds every invariant verifier to exact full-tier code and context evidence", async () => {
  const h = await fixture();
  const current = await h.current.getCurrent(h.taskId);
  const ref = {
    kind: "invariant" as const,
    recordId: "invariant:total",
    revisionId: "revision:total",
  };
  const prepared = {
    ...h.prepared,
    snapshot: { ...h.prepared.snapshot, normativeRevisions: [ref] },
  };
  const state = {
    ...current,
    normativeRecords: [
      {
        origin: "local" as const,
        revision: {
          ...ref,
          organizationId: "org",
          statement: "Total stays correct",
          scope: { kind: "personal" as const, subjectId: "user" },
          evidence: [],
          egress: { dataClassification: "internal" as const, allowedRuntimeProviders: [] },
          authoredBy: "user",
          authoredAt: new Date().toISOString(),
          verifiers: [
            { kind: "test" as const, ref: "workspace:test" },
            { kind: "lint" as const, ref: "workspace:lint" },
          ],
        },
        activation: {
          ...ref,
          organizationId: "org",
          scope: { kind: "personal" as const, subjectId: "user" },
          state: "accepted_local" as const,
          acceptedBy: "user",
          acceptedAt: new Date().toISOString(),
        },
      },
    ],
  };
  const evaluate = (runs = h.runs) =>
    evaluateBoundCompletionInvariants(prepared, state, runs, h.integration.resultRevision)[0];
  expect(evaluate()?.status).toBe("guarded");
  expect(evaluate(h.runs.filter((run) => run.verifierRef !== "workspace:lint"))?.status).toBe(
    "unguarded",
  );
  expect(evaluate(h.runs.map((run) => ({ ...run, codeRevision: "stale" })))?.status).toBe(
    "unguarded",
  );
  expect(evaluate(h.runs.map((run) => ({ ...run, tier: "targeted" as const })))?.status).toBe(
    "unguarded",
  );
  expect(evaluate(h.runs.map((run) => ({ ...run, result: "inconclusive" as const })))?.status).toBe(
    "inconclusive",
  );
  expect(evaluate(h.runs.map((run) => ({ ...run, result: "failed" as const })))?.status).toBe(
    "violated",
  );
});

it("records one explicit finalization across concurrent UUID replays and restart", async () => {
  const h = await fixture();
  const assessed = await h.assessment.assess(h.request);
  expect((await h.assessment.assess(h.request)).completionBasisDigest).toBe(
    assessed.completionBasisDigest,
  );
  const operations = new LocalTaskFinalizationOperations(h.data, h.assessment);
  const request = {
    ...h.request,
    requestId: "8825c99a-805b-4ffc-b0f1-bd4c812e900a",
    expectedCompletionBasisDigest: assessed.completionBasisDigest,
  };
  const results = await Promise.all(Array.from({ length: 8 }, () => operations.finalize(request)));
  expect(results.filter((result) => result.created)).toHaveLength(1);
  expect(new Set(results.map((result) => result.record.recordId)).size).toBe(1);
  const frozenManifest = results[0]?.record.accuracyManifest;
  expect(await new LocalTaskInventoryOperations(h.data).list({})).toMatchObject({
    currentEvidence: "not_revalidated",
    tasks: [
      {
        taskId: h.taskId,
        integration: { jobId: h.request.jobId },
        finalization: { recordId: results[0]?.record.recordId, jobId: h.request.jobId },
      },
    ],
  });
  await h.assessment.assess(h.request);
  const reloaded = new LocalTaskFinalizationOperations(h.data, h.assessment);
  expect(await reloaded.inspect({ taskId: h.taskId, requestId: request.requestId })).toMatchObject({
    record: results[0]?.record,
    currentEvidence: "not_revalidated",
  });
  expect((await reloaded.inspect({ taskId: h.taskId })).record?.accuracyManifest).toEqual(
    frozenManifest,
  );
  await expect(reloaded.finalize({ ...request, jobId: "different" })).rejects.toThrow("conflicts");
  await writeFile(path.join(h.integration.workspacePath, "index.ts"), "changed after completion\n");
  expect(await reloaded.finalize(request)).toMatchObject({
    created: false,
    currentEvidence: "not_revalidated",
  });
});

it("refuses finalization for changed reviewed evidence or missing owner approval", async () => {
  const h = await fixture("medium");
  const assessed = await h.assessment.assess(h.request);
  const operations = new LocalTaskFinalizationOperations(h.data, h.assessment);
  const request = {
    ...h.request,
    requestId: "8825c99a-805b-4ffc-b0f1-bd4c812e900a",
    expectedCompletionBasisDigest: assessed.completionBasisDigest,
  };
  await expect(operations.finalize(request)).rejects.toThrow("requirements are not satisfied");
  expect((await operations.inspect({ taskId: h.taskId })).record).toBeNull();
  await expect(
    operations.finalize({ ...request, expectedCompletionBasisDigest: `sha256:${"e".repeat(64)}` }),
  ).rejects.toThrow("evidence changed");
  await expect(operations.finalize({ ...request, approval: "code_owner" })).rejects.toThrow();
});

it("refuses corrupted finalization history instead of treating it as a missing record", async () => {
  const h = await fixture();
  const operations = new LocalTaskFinalizationOperations(h.data, h.assessment);
  const assessed = await h.assessment.assess(h.request);
  await operations.finalize({
    ...h.request,
    requestId: "8825c99a-805b-4ffc-b0f1-bd4c812e900a",
    expectedCompletionBasisDigest: assessed.completionBasisDigest,
  });
  const directory = path.join(h.data, "task-finalizations");
  const filename = (await readdir(directory)).find((name) => name.endsWith(".json"));
  if (!filename) throw new Error("Missing fixture finalization file");
  const filePath = path.join(directory, filename);
  const envelope = JSON.parse(await readFile(filePath, "utf8"));
  envelope.records[0].gitCommit = "forged";
  await writeFile(filePath, JSON.stringify(envelope));
  await expect(operations.inspect({ taskId: h.taskId })).rejects.toThrow("integrity");
});

it("revalidates current completion, detects changed sources and preserves immutable history", async () => {
  const h = await fixture();
  const operations = new LocalTaskFinalizationOperations(h.data, h.assessment);
  const assessed = await h.assessment.assess(h.request);
  const { record } = await operations.finalize({
    ...h.request,
    requestId: "6825c99a-805b-4ffc-b0f1-bd4c812e900a",
    expectedCompletionBasisDigest: assessed.completionBasisDigest,
  });
  const request = { taskId: h.taskId, expectedRecordId: record.recordId };
  expect(await operations.revalidate(request)).toMatchObject({
    currentEvidence: "revalidated_current",
    state: "done",
    recordId: record.recordId,
  });
  await writeFile(path.join(h.root, "notes.md"), "## Updated\nThe original source changed.\n");
  expect(await operations.revalidate(request)).toMatchObject({ currentEvidence: "changed" });
  expect((await operations.inspect({ taskId: h.taskId })).record).toEqual(record);
  await writeFile(path.join(h.integration.workspacePath, "index.ts"), "dirty code\n");
  await expect(operations.revalidate(request)).rejects.toThrow();
  expect((await operations.inspect({ taskId: h.taskId })).record).toEqual(record);
});

it("requires the latest reviewed record and refuses history changes during assessment", async () => {
  const h = await fixture();
  const assessment = { assess: vi.fn(h.assessment.assess.bind(h.assessment)) };
  const operations = new LocalTaskFinalizationOperations(h.data, assessment);
  await expect(
    operations.revalidate({ taskId: h.taskId, expectedRecordId: `sha256:${"a".repeat(64)}` }),
  ).rejects.toThrow("history changed");
  expect(assessment.assess).not.toHaveBeenCalled();
  const assessed = await h.assessment.assess(h.request);
  const original = await operations.finalize({
    ...h.request,
    requestId: "6825c99a-805b-4ffc-b0f1-bd4c812e900a",
    expectedCompletionBasisDigest: assessed.completionBasisDigest,
  });
  assessment.assess.mockImplementationOnce(async () => {
    await new LocalTaskFinalizationOperations(h.data, h.assessment).finalize({
      ...h.request,
      requestId: "7825c99a-805b-4ffc-b0f1-bd4c812e900a",
      expectedCompletionBasisDigest: assessed.completionBasisDigest,
    });
    return h.assessment.assess(h.request);
  });
  await expect(
    operations.revalidate({ taskId: h.taskId, expectedRecordId: original.record.recordId }),
  ).rejects.toThrow("history changed during");
  await expect(
    operations.revalidate({ taskId: h.taskId, expectedRecordId: original.record.recordId }),
  ).rejects.toThrow("history changed");
});
