import { execFile } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { TaskContextWorkflow } from "@kontext-brain/context";
import { FileLocalNormativeOverlayStore, FileTaskContextRepository } from "@kontext-brain/local";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { afterEach, expect, it } from "vitest";
import { z } from "zod";
import { LocalKnowledgeOperations } from "../src/local-knowledge-operations.js";
import { loadLocalKnowledgePrincipal } from "../src/local-knowledge-principal.js";
import {
  LocalTaskCreationOperations,
  type LocalTaskCreationRequest,
} from "../src/local-task-creation.js";
import { RegisteredTaskContextProvider } from "../src/registered-task-context.js";

const execute = promisify(execFile);
const dirs: string[] = [];
afterEach(async () => {
  await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true })));
});
async function fixture() {
  const dir = await mkdtemp(path.join(tmpdir(), "kontext-task-creation-"));
  dirs.push(dir);
  const workspace = path.join(
    dir,
    process.platform === "win32" ? "workspace with spaces" : "workspace with spaces ",
  );
  const home = path.join(dir, "home");
  const templates = path.join(dir, "empty-templates");
  await Promise.all([workspace, home, templates].map((item) => mkdir(item)));
  const environment = {
    PATH: process.env.PATH,
    HOME: home,
    USERPROFILE: home,
    XDG_CONFIG_HOME: home,
    SYSTEMROOT: process.env.SYSTEMROOT,
    TMPDIR: dir,
    GIT_CONFIG_NOSYSTEM: "1",
  };
  const git = async (args: string[]) =>
    (await execute("git", args, { cwd: workspace, env: environment })).stdout.trim();
  await git(["init", "--quiet", `--template=${templates}`]);
  await writeFile(path.join(workspace, "index.ts"), "export function total() { return 1; }\n");
  await git(["add", "--", "index.ts"]);
  await git([
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
  ]);
  const revision = await git(["rev-parse", "HEAD"]);
  const data = path.join(dir, "data");
  const knowledge = new LocalKnowledgeOperations(data);
  await writeFile(
    path.join(dir, "notes.md"),
    "## Terminology\nUse the existing total domain term.\n",
  );
  const source = await knowledge.registerMarkdownSource({
    workspacePath: dir,
    relativePath: "notes.md",
  });
  await knowledge.setSourceSharing({
    resourceId: source.resourceId,
    expectedRevision: 1,
    expectedContentHash: source.contentHash,
    dataClassification: "internal",
    allowedRuntimeProviders: ["codex"],
  });
  const request: LocalTaskCreationRequest = {
    requestId: "5825c99a-805b-4ffc-b0f1-bd4c812e900a",
    workspacePath: workspace,
    workspaceId: "workspace:fixture",
    expectedCodeRevision: revision,
    contract: {
      intent: "Implement total accurately",
      acceptance: [
        {
          criterionId: "total",
          statement: "Correct total",
          verifier: { kind: "test", ref: "workspace:test" },
        },
      ],
      nonGoals: ["Do not rename total"],
      targets: ["planned:total"],
      risk: "low",
    },
    sourceResourceIds: [source.resourceId],
    logicPlans: [
      {
        workItemId: "logic:total",
        plannedSymbolIds: ["planned:total"],
        allowedPaths: ["index.ts"],
        plannedSymbols: [
          {
            plannedSymbolId: "planned:total",
            intendedIdentity: {
              relativePath: "index.ts",
              kind: "function",
              qualifiedName: "total",
            },
            responsibility: "Compute total",
          },
        ],
      },
    ],
  };
  return {
    dir,
    data,
    workspace,
    knowledge,
    source,
    request,
    environment,
    creation: new LocalTaskCreationOperations(data, environment),
    repository: new FileTaskContextRepository(data),
  };
}

it("creates a real prepared Task from source-owned Evidence and exposes it through the existing workbench inspection", async () => {
  const h = await fixture();
  const result = await h.creation.createTask(h.request);
  expect(result.created).toBe(true);
  expect(result.inspection).toMatchObject({ status: "current", taskId: result.taskId });
  expect(JSON.stringify(result)).not.toContain("Use the existing total domain term");
  const initial = await h.repository.getInitialRegistration(result.taskId);
  expect(initial?.prepared.contract.taskId).toBe(result.taskId);
  expect(initial?.state.normativeRecords).toEqual([]);
  expect(initial?.state.logicPlans[0]?.plannedSymbols?.[0]?.taskId).toBe(result.taskId);
  const context = await new TaskContextWorkflow(
    new RegisteredTaskContextProvider(h.data, h.repository),
    h.repository,
  ).beginLogic({
    taskId: result.taskId,
    logic: { workItemId: "logic:total", plannedSymbolIds: ["planned:total"] },
    runtimeProvider: "codex",
    issuedAt: new Date().toISOString(),
    expiresAt: new Date(Date.now() + 60_000).toISOString(),
    totalTokenBudget: 10000,
    optionalEvidenceTokenBudget: 0,
  });
  expect(context.editingAllowed).toBe(true);
  expect(context.mandatory.evidence[0]?.provenance?.resourceId).toBe(h.source.resourceId);
});

it("recovers duplicate creation across restart without recapturing or resetting a changed task", async () => {
  const h = await fixture();
  const results = await Promise.all(
    Array.from({ length: 8 }, () =>
      new LocalTaskCreationOperations(h.data, h.environment).createTask(h.request),
    ),
  );
  expect(results.filter((result) => result.created)).toHaveLength(1);
  expect(new Set(results.map((result) => result.taskId)).size).toBe(1);
  const first = results[0];
  if (!first) throw new Error("missing fixture result");
  const observed = await h.repository.getCurrentVersion(first.taskId);
  await h.repository.publishCurrent(
    first.taskId,
    { ...observed.state, codeRevision: "later" },
    { expectedDigest: observed.digest },
  );
  await rm(path.join(h.dir, "notes.md"));
  expect(await h.creation.createTask(h.request)).toMatchObject({
    created: false,
    taskId: first.taskId,
    inspection: { status: "stale" },
  });
  await expect(
    h.creation.createTask({
      ...h.request,
      contract: { ...h.request.contract, intent: "Different task" },
    }),
  ).rejects.toThrow("conflicts");
});

it("preflights changed source content and cannot inherit permission for an older capture", async () => {
  const h = await fixture();
  await writeFile(path.join(h.dir, "notes.md"), "## Terminology\nUpdated terminology.\n");
  const result = await h.creation.createTask(h.request);
  const state = await h.repository.getCurrent(result.taskId);
  expect(state.evidence[0]?.allowedRuntimeProviders).toEqual([]);
  const prepared = await h.repository.get(result.taskId);
  expect(prepared?.snapshot.requiredEvidenceIds).toContain(state.evidence[0]?.evidenceId);
  expect((await h.knowledge.inspectSource(h.source)).sharing).toBeNull();
});

it("revalidates permissions and newly added source sections before logic work and after explicit snapshot refresh", async () => {
  const h = await fixture();
  const result = await h.creation.createTask(h.request);
  const current = new RegisteredTaskContextProvider(h.data, h.repository);
  const workflow = new TaskContextWorkflow(current, h.repository);
  const begin = () =>
    workflow.beginLogic({
      taskId: result.taskId,
      logic: { workItemId: "logic:total", plannedSymbolIds: ["planned:total"] },
      runtimeProvider: "codex",
      issuedAt: new Date().toISOString(),
      expiresAt: new Date(Date.now() + 60_000).toISOString(),
      totalTokenBudget: 10000,
      optionalEvidenceTokenBudget: 0,
    });
  expect((await begin()).editingAllowed).toBe(true);
  const inspected = await h.knowledge.inspectSource(h.source);
  await h.knowledge.setSourceSharing({
    resourceId: h.source.resourceId,
    expectedRevision: inspected.revision,
    expectedContentHash: inspected.contentHash,
    dataClassification: "internal",
    allowedRuntimeProviders: [],
  });
  expect((await begin()).editingAllowed).toBe(false);
  await writeFile(
    path.join(h.dir, "notes.md"),
    "## Terminology\nKeep total.\n\n## New constraint\nHandle negative numbers.\n",
  );
  expect((await begin()).editingAllowed).toBe(false);
  const changed = await h.knowledge.inspectSource(h.source);
  await h.knowledge.setSourceSharing({
    resourceId: h.source.resourceId,
    expectedRevision: changed.revision,
    expectedContentHash: changed.contentHash,
    dataClassification: "internal",
    allowedRuntimeProviders: ["codex"],
  });
  expect((await begin()).editingAllowed).toBe(false);
  await workflow.refreshTaskContext({ taskId: result.taskId, createdAt: new Date().toISOString() });
  const refreshed = await begin();
  expect(refreshed.editingAllowed).toBe(true);
  expect(refreshed.mandatory.evidence).toHaveLength(2);
  expect(
    refreshed.mandatory.evidence.some((item) => item.text.includes("Handle negative numbers")),
  ).toBe(true);
  await rm(path.join(h.dir, "notes.md"));
  expect((await begin()).editingAllowed).toBe(false);
});

it("loads accepted local rules from the owning overlay, not source-authored approvals", async () => {
  const h = await fixture();
  const principal = await loadLocalKnowledgePrincipal(h.data);
  const scope = { kind: "workspace" as const, workspaceId: h.request.workspaceId };
  const evidenceId = h.source.evidence[0]?.evidenceId;
  if (!evidenceId) throw new Error("missing fixture Evidence");
  await new FileLocalNormativeOverlayStore(h.data).save(
    {
      organizationId: principal.organizationId,
      subjectId: principal.subjectId,
      workspaceId: h.request.workspaceId,
    },
    {
      schemaVersion: 1,
      organizationId: principal.organizationId,
      revisions: [
        {
          kind: "decision",
          organizationId: principal.organizationId,
          recordId: "decision:total",
          revisionId: "revision:one",
          scope,
          statement: "Do not rename the total domain term",
          evidence: [{ evidenceId }],
          egress: { dataClassification: "internal", allowedRuntimeProviders: ["codex"] },
          authoredBy: principal.subjectId,
          authoredAt: "2026-09-06T00:00:00.000Z",
        },
      ],
      activations: [
        {
          organizationId: principal.organizationId,
          kind: "decision",
          recordId: "decision:total",
          revisionId: "revision:one",
          scope,
          state: "accepted_local",
          acceptedBy: principal.subjectId,
          acceptedAt: "2026-09-06T00:00:00.000Z",
        },
      ],
    },
  );
  const result = await h.creation.createTask(h.request);
  const state = await new RegisteredTaskContextProvider(h.data, h.repository).getCurrent(
    result.taskId,
  );
  expect(state.normativeRecords).toHaveLength(1);
  expect(state.normativeRecords[0]?.revision.revisionId).toBe("revision:one");
  expect(state.normativeRecords[0]?.activation.acceptedBy).toBe(principal.subjectId);
});

it("runs creation, protected preparation and permission revalidation through the actual executable host MCP", async () => {
  const h = await fixture();
  const client = new Client({ name: "task-creation-fixture", version: "1" });
  const hostToken = "d".repeat(64);
  try {
    await client.connect(
      new StdioClientTransport({
        command: process.env.KONTEXT_TEST_NODE_EXECUTABLE ?? process.execPath,
        args: [path.resolve("plugins/kontext-brain/server.mjs")],
        cwd: h.dir,
        env: Object.fromEntries(
          Object.entries({
            ...h.environment,
            KONTEXT_PLUGIN_DATA: h.data,
            KONTEXT_HOST_MANAGEMENT_TOKEN: hostToken,
            ELECTRON_RUN_AS_NODE: "1",
            CODEX_HOME: path.join(h.dir, "home"),
            CLAUDE_CONFIG_DIR: path.join(h.dir, "home"),
          }).flatMap(([key, value]) => (value === undefined ? [] : [[key, value]])),
        ),
        stderr: "pipe",
      }),
    );
    const denied = await client.callTool({
      name: "kontext_create_task",
      arguments: { ...h.request, hostToken: "e".repeat(64) },
    });
    expect(denied.isError).toBe(true);
    const created = await client.callTool({
      name: "kontext_create_task",
      arguments: { ...h.request, hostToken },
    });
    expect(created.isError).not.toBe(true);
    const result = z
      .object({ taskId: z.string(), created: z.literal(true) })
      .parse(created.structuredContent);
    expect(JSON.stringify(created)).not.toContain("Use the existing total domain term");
    const inventory = await client.callTool({
      name: "kontext_list_tasks",
      arguments: { hostToken },
    });
    expect(inventory.isError).not.toBe(true);
    expect(inventory.structuredContent).toMatchObject({
      observation: "saved_metadata_only",
      currentEvidence: "not_revalidated",
      tasks: [{ taskId: result.taskId, latestSchedule: null, finalization: null }],
    });
    expect(JSON.stringify(inventory)).not.toContain("Use the existing total domain term");
    expect(
      (
        await client.callTool({
          name: "kontext_list_tasks",
          arguments: { hostToken: "e".repeat(64) },
        })
      ).isError,
    ).toBe(true);
    const duplicate = await client.callTool({
      name: "kontext_create_task",
      arguments: { ...h.request, hostToken },
    });
    expect(duplicate.structuredContent).toMatchObject({ taskId: result.taskId, created: false });
    const changedContract = await client.callTool({
      name: "kontext_prepare_task",
      arguments: {
        contract: {
          ...h.request.contract,
          taskId: result.taskId,
          intent: "Unauthorized scope change",
        },
        createdAt: new Date().toISOString(),
      },
    });
    expect(changedContract.isError).toBe(true);
    const beginRequest = {
      taskId: result.taskId,
      workspacePath: h.workspace,
      logic: { workItemId: "logic:total", plannedSymbolIds: ["planned:total"] },
      runtimeProvider: "codex",
      totalTokenBudget: 10000,
      optionalEvidenceTokenBudget: 0,
    };
    const begin = await client.callTool({ name: "kontext_begin_logic", arguments: beginRequest });
    expect(begin.isError).not.toBe(true);
    expect(begin.structuredContent).toMatchObject({ editingAllowed: true });
    const source = await h.knowledge.inspectSource(h.source);
    await h.knowledge.setSourceSharing({
      resourceId: h.source.resourceId,
      expectedRevision: source.revision,
      expectedContentHash: source.contentHash,
      dataClassification: "internal",
      allowedRuntimeProviders: [],
    });
    const revoked = await client.callTool({ name: "kontext_begin_logic", arguments: beginRequest });
    expect(revoked.isError).not.toBe(true);
    expect(revoked.structuredContent).toMatchObject({ editingAllowed: false });
    expect(JSON.stringify(revoked)).not.toContain("Use the existing total domain term");
  } finally {
    await client.close();
  }
});

it("refuses unavailable mandatory sources, unreviewed working files and source-authored authority", async () => {
  const h = await fixture();
  await expect(
    h.creation.createTask({ ...h.request, expectedCodeRevision: "0".repeat(40) }),
  ).rejects.toThrow("revision changed");
  await writeFile(path.join(h.workspace, "index.ts"), "export function total() { return 2; }\n");
  await expect(h.creation.createTask(h.request)).rejects.toThrow("revision changed");
  await writeFile(path.join(h.workspace, "index.ts"), "export function total() { return 1; }\n");
  await expect(
    h.creation.createTask({ ...h.request, organizationId: "forged" } as LocalTaskCreationRequest),
  ).rejects.toThrow();
  await rm(path.join(h.dir, "notes.md"));
  await expect(h.creation.createTask(h.request)).rejects.toThrow();
});
