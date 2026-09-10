import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { FileTaskContextRepository } from "@kontext-brain/local";
import {
  type AgentRuntimePort,
  type RuntimePlanningInput,
  createRuntimeCapabilitySnapshot,
} from "@kontext-brain/orchestrator";
import { afterEach, expect, it, vi } from "vitest";
import { LocalKnowledgeOperations } from "../src/local-knowledge-operations.js";
import { LocalTaskPlanningOperations } from "../src/local-task-planning.js";
import { resolveTaskExecutionRepository } from "../src/task-execution-repository.js";
import {
  type TaskPlanProposal,
  type TaskPlanningRequest,
  taskPlanProposalSchema,
} from "../src/task-planning-contract.js";

const roots: string[] = [];
afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});
const proposal: TaskPlanProposal = {
  contract: {
    intent: "Implement total",
    risk: "low",
    nonGoals: ["Do not rename total"],
    targets: ["planned:total"],
    acceptance: [
      {
        criterionId: "total",
        statement: "Accurate total",
        verifier: { kind: "test", ref: "workspace:test" },
      },
    ],
  },
  logicPlans: [
    {
      workItemId: "logic:total",
      plannedSymbolIds: ["planned:total"],
      allowedPaths: ["index.ts"],
      plannedSymbols: [
        {
          plannedSymbolId: "planned:total",
          intendedIdentity: { relativePath: "index.ts", kind: "function", qualifiedName: "total" },
          responsibility: "Compute total",
        },
      ],
    },
  ],
};
async function fixture() {
  const root = await mkdtemp(path.join(tmpdir(), "kontext-plan-test-"));
  roots.push(root);
  const workspacePath = path.join(root, "workspace");
  const data = path.join(root, "data");
  const home = path.join(root, "home");
  const templates = path.join(root, "templates");
  await Promise.all([workspacePath, home, templates].map((dir) => mkdir(dir)));
  const environment = {
    PATH: process.env.PATH,
    HOME: home,
    USERPROFILE: home,
    XDG_CONFIG_HOME: home,
    SYSTEMROOT: process.env.SYSTEMROOT,
    TMPDIR: root,
    GIT_CONFIG_NOSYSTEM: "1",
  };
  const git = async (args: string[]) =>
    (
      await promisify(execFile)("git", args, { cwd: workspacePath, env: environment })
    ).stdout.trim();
  await git(["init", "--quiet", `--template=${templates}`]);
  await writeFile(path.join(workspacePath, "index.ts"), "export function total() { return 1; }\n");
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
  const knowledge = new LocalKnowledgeOperations(data);
  await writeFile(
    path.join(root, "notes.md"),
    "# Total\nKeep the established total domain term.\n",
  );
  const source = await knowledge.registerMarkdownSource({
    workspacePath: root,
    relativePath: "notes.md",
  });
  await knowledge.setSourceSharing({
    resourceId: source.resourceId,
    expectedRevision: 1,
    expectedContentHash: source.contentHash,
    dataClassification: "internal",
    allowedRuntimeProviders: ["codex", "claude"],
  });
  const plan = vi.fn(async (_input: RuntimePlanningInput) => ({
    sessionId: "fixture-session",
    provider: "codex" as const,
    status: "completed" as const,
    output: JSON.stringify(proposal),
    events: [],
    startedAt: new Date().toISOString(),
    completedAt: new Date().toISOString(),
  }));
  const adapter: AgentRuntimePort = {
    provider: "codex",
    plan,
    inspectCapabilities: vi.fn(async () =>
      createRuntimeCapabilitySnapshot({
        provider: "codex",
        cliPath: "fixture-only",
        installed: true,
        authenticated: true,
        billingPath: "subscription",
        inspectedAt: new Date().toISOString(),
        supports: {
          structuredOutput: true,
          sessionResume: true,
          mcp: true,
          hooks: true,
          workspaceSandbox: true,
        },
      }),
    ),
    start: vi.fn(async () => {
      throw new Error("Implementation must not start during planning");
    }),
    resume: vi.fn(async () => {
      throw new Error("No automatic resume");
    }),
    terminate: vi.fn(async () => undefined),
  };
  const operations = new LocalTaskPlanningOperations(data, [adapter], environment);
  const request: TaskPlanningRequest = {
    requestId: randomUUID(),
    goal: "Make total accurate",
    workspacePath,
    workspaceId: "workspace:fixture",
    sourceResourceIds: [source.resourceId],
    provider: "codex",
  };
  const settled = async (requestId = request.requestId) => {
    await vi.waitFor(
      async () => expect((await operations.inspectPlan({ requestId })).status).not.toBe("planning"),
      { timeout: 8_000, interval: 20 },
    );
    return operations.inspectPlan({ requestId });
  };
  return {
    root,
    data,
    workspacePath,
    environment,
    knowledge,
    source,
    plan,
    adapter,
    operations,
    request,
    settled,
    git,
  };
}

async function refinementFixture() {
  const h = await fixture();
  await h.operations.startPlan(h.request);
  const parent = await h.settled();
  expect(parent.status).toBe("review");
  return {
    ...h,
    parent,
    refinement: {
      requestId: randomUUID(),
      parentRequestId: h.request.requestId,
      expectedParentDigest: parent.planDigest ?? "missing",
      feedback: "Also cover zero inputs without renaming total.",
    },
  };
}

it("refines an exact unapproved draft in a new request without approving or mutating its parent", async () => {
  const h = await refinementFixture();
  expect((await h.operations.refinePlan(h.refinement)).created).toBe(true);
  const child = await h.settled(h.refinement.requestId);
  expect(child.status, child.diagnostic).toBe("review");
  expect(child.request).toEqual({ ...h.request, requestId: h.refinement.requestId });
  expect(child.refinement).toEqual({
    parentRequestId: h.refinement.parentRequestId,
    expectedParentDigest: h.refinement.expectedParentDigest,
    feedback: h.refinement.feedback,
  });
  expect(await h.operations.inspectPlan(h.request)).toEqual(h.parent);
  expect(child.planDigest).not.toBe(h.parent.planDigest);
  const sent = h.plan.mock.calls[1]?.[0];
  expect(sent?.prompt).toContain(h.refinement.feedback);
  expect(sent?.prompt).toContain(JSON.stringify(h.parent.proposal));
  expect(sent?.prompt).toContain(h.source.resourceId);
  expect(sent?.executionRole).toBe("planning");
  expect(h.adapter.start).not.toHaveBeenCalled();
  expect(h.adapter.resume).not.toHaveBeenCalled();
  await expect(
    h.operations.approvePlan({
      requestId: h.refinement.requestId,
      expectedPlanDigest: h.parent.planDigest ?? "missing",
    }),
  ).rejects.toThrow(/exact current proposal/);
  const approved = await h.operations.approvePlan({
    requestId: h.refinement.requestId,
    expectedPlanDigest: child.planDigest ?? "missing",
  });
  expect(approved.inspectionBasis).toBe("stored_context");
  expect((await h.operations.inspectPlan(h.request)).status).toBe("review");
});

it("replays a refinement once across concurrent calls and restart, including after parent approval", async () => {
  const h = await refinementFixture();
  const results = await Promise.all([
    h.operations.refinePlan(h.refinement),
    h.operations.refinePlan(h.refinement),
  ]);
  expect(results.filter((result) => result.created)).toHaveLength(1);
  await h.settled(h.refinement.requestId);
  expect(h.plan).toHaveBeenCalledTimes(2);
  await expect(
    h.operations.startPlan({ ...h.request, requestId: h.refinement.requestId }),
  ).rejects.toThrow(/different input/);
  await h.operations.approvePlan({
    requestId: h.request.requestId,
    expectedPlanDigest: h.refinement.expectedParentDigest,
  });
  const restarted = new LocalTaskPlanningOperations(h.data, [h.adapter], h.environment);
  expect((await restarted.refinePlan(h.refinement)).created).toBe(false);
  expect(h.plan).toHaveBeenCalledTimes(2);
  await expect(
    restarted.refinePlan({ ...h.refinement, feedback: "Different input" }),
  ).rejects.toThrow(/different input/);
  await expect(restarted.refinePlan({ ...h.refinement, requestId: randomUUID() })).rejects.toThrow(
    /unapproved/,
  );
});

it.each(["digest", "missing", "self", "override"] as const)(
  "refuses a %s refinement before dispatch",
  async (kind) => {
    const h = await refinementFixture();
    const input = {
      ...h.refinement,
      ...(kind === "digest" ? { expectedParentDigest: `sha256:${"0".repeat(64)}` } : {}),
      ...(kind === "missing" ? { parentRequestId: randomUUID() } : {}),
      ...(kind === "self" ? { requestId: h.request.requestId } : {}),
      ...(kind === "override" ? { provider: "claude", workspacePath: h.root } : {}),
    };
    await expect(h.operations.refinePlan(input)).rejects.toThrow();
    expect(h.plan).toHaveBeenCalledTimes(1);
  },
);

it.each(["code", "source", "sharing"] as const)(
  "does not retransmit a prior draft after its %s basis changes",
  async (kind) => {
    const h = await refinementFixture();
    if (kind === "code") {
      await writeFile(
        path.join(h.workspacePath, "index.ts"),
        "export function total() { return 2; }\n",
      );
    } else if (kind === "source") {
      await writeFile(path.join(h.root, "notes.md"), "# New source\nChanged rule.\n");
    } else {
      await h.knowledge.setSourceSharing({
        resourceId: h.source.resourceId,
        expectedRevision: 2,
        expectedContentHash: h.source.contentHash,
        dataClassification: "internal",
        allowedRuntimeProviders: [],
      });
    }
    await h.operations.refinePlan(h.refinement);
    expect((await h.settled(h.refinement.requestId)).status).toBe("failed");
    expect(h.plan).toHaveBeenCalledTimes(1);
    expect((await h.operations.inspectPlan(h.request)).planDigest).toBe(h.parent.planDigest);
  },
);

it("rechecks sharing after capability inspection before retransmitting the prior draft", async () => {
  const h = await refinementFixture();
  const capability = await h.adapter.inspectCapabilities();
  h.adapter.inspectCapabilities = vi.fn(async () => {
    await h.knowledge.setSourceSharing({
      resourceId: h.source.resourceId,
      expectedRevision: 2,
      expectedContentHash: h.source.contentHash,
      dataClassification: "internal",
      allowedRuntimeProviders: [],
    });
    return capability;
  });
  await h.operations.refinePlan(h.refinement);
  expect((await h.settled(h.refinement.requestId)).status).toBe("failed");
  expect(h.plan).toHaveBeenCalledTimes(1);
});

it("does not offer a refinement that completes after its parent was approved", async () => {
  const h = await refinementFixture();
  const completed = await h.plan.mock.results[0]?.value;
  if (!completed) throw new Error("Missing parent fixture result");
  h.plan.mockImplementationOnce(async () => {
    await h.operations.approvePlan({
      requestId: h.request.requestId,
      expectedPlanDigest: h.refinement.expectedParentDigest,
    });
    return completed;
  });
  await h.operations.refinePlan(h.refinement);
  const child = await h.settled(h.refinement.requestId);
  expect(child.status).toBe("failed");
  expect(child.proposal).toBeUndefined();
  expect((await h.operations.inspectPlan(h.request)).status).toBe("approved");
});

it.each([false, true])(
  "rechecks code after capability inspection before planning dispatch (refinement=%s)",
  async (refined) => {
    const h = await refinementFixture();
    const capability = await h.adapter.inspectCapabilities();
    h.adapter.inspectCapabilities = vi.fn(async () => {
      await writeFile(
        path.join(h.workspacePath, "index.ts"),
        "export function total() { return 99; }\n",
      );
      return capability;
    });
    const requestId = h.refinement.requestId;
    if (refined) {
      await h.operations.refinePlan(h.refinement);
    } else {
      await h.operations.startPlan({ ...h.request, requestId });
    }
    expect((await h.settled(requestId)).status).toBe("failed");
    expect(h.plan).toHaveBeenCalledTimes(1);
  },
);

it.each(["dirty", "unborn", "folder"] as const)(
  "plans from captured %s working files, approves the exact baseline and resolves only the owned seed",
  async (kind) => {
    const h = await fixture();
    const workspacePath = kind === "dirty" ? h.workspacePath : path.join(h.root, `${kind}-code`);
    if (kind !== "dirty") await mkdir(workspacePath);
    if (kind === "unborn")
      await promisify(execFile)("git", ["init", "-q"], { cwd: workspacePath, env: h.environment });
    const contents = "export function total() { return 42; }\r\n";
    await writeFile(path.join(workspacePath, "index.ts"), contents);
    const request = { ...h.request, workspacePath };
    await h.operations.startPlan(request);
    const reviewed = await h.settled();
    expect(reviewed.status, reviewed.diagnostic).toBe("review");
    const sent = h.plan.mock.calls[0]?.[0];
    if (!sent) throw new Error("Planner was not called");
    expect(sent.workspacePath).not.toBe(await realpath(workspacePath));
    expect(await readFile(path.join(sent.workspacePath, "index.ts"), "utf8")).toBe(contents);
    expect(sent.codeRevision).toBe(reviewed.codeRevision);
    const approved = await h.operations.approvePlan({
      requestId: request.requestId,
      expectedPlanDigest: reviewed.planDigest ?? "missing",
    });
    const executionPath = await resolveTaskExecutionRepository(
      h.data,
      approved.taskId,
      workspacePath,
    );
    expect(executionPath).toBe(sent.workspacePath);
    const registration = await new FileTaskContextRepository(h.data).getInitialRegistration(
      approved.taskId,
    );
    expect(registration?.owner.workspacePath).toBe(await realpath(workspacePath));
    expect(registration?.owner.workspaceSeed?.codeRevision).toBe(sent.codeRevision);
    await expect(resolveTaskExecutionRepository(h.data, approved.taskId, h.root)).rejects.toThrow(
      /registered Task workspace/,
    );
    expect(await readFile(path.join(workspacePath, "index.ts"), "utf8")).toBe(contents);
    await writeFile(path.join(executionPath, "index.ts"), "tampered seed");
    await expect(
      resolveTaskExecutionRepository(h.data, approved.taskId, workspacePath),
    ).rejects.toThrow(/seed.*changed/i);
  },
);

it("routes every manual_review to the independent review and keeps it out of work items", async () => {
  const h = await fixture();
  h.plan.mockResolvedValueOnce({
    sessionId: "fixture-session",
    provider: "codex" as const,
    status: "completed" as const,
    output: JSON.stringify({
      ...proposal,
      contract: {
        ...proposal.contract,
        acceptance: [
          ...proposal.contract.acceptance,
          {
            criterionId: "comment",
            statement: "The Why comment explains the override.",
            verifier: { kind: "manual_review", ref: "Review the comment wording." },
          },
        ],
      },
      logicPlans: proposal.logicPlans.map((plan) => ({
        ...plan,
        requiredVerifiers: [
          { kind: "manual_review", ref: "Review the comment wording." },
          { kind: "test", ref: "workspace:test" },
        ],
      })),
    }),
    events: [],
    startedAt: new Date().toISOString(),
    completedAt: new Date().toISOString(),
  });
  expect((await h.operations.startPlan(h.request)).created).toBe(true);
  const reviewed = await h.settled();
  expect(reviewed.status, reviewed.diagnostic).toBe("review");
  expect(reviewed.proposal?.contract.acceptance.map((criterion) => criterion.verifier)).toEqual([
    { kind: "test", ref: "workspace:test" },
    { kind: "manual_review", ref: "kontext:independent-review" },
  ]);
  expect(reviewed.proposal?.logicPlans[0]?.requiredVerifiers).toEqual([
    { kind: "test", ref: "workspace:test" },
  ]);
  expect(h.plan.mock.calls[0]?.[0]?.prompt).toContain(
    "never in a Logic Work Item's requiredVerifiers",
  );
});

it("tells the planner exactly which workspace verifiers exist", async () => {
  const h = await fixture();
  // Why: no declared verifiers means the contract must not invent lint/test commands.
  expect((await h.operations.startPlan(h.request)).created).toBe(true);
  await h.settled();
  const bare = h.plan.mock.calls[0]?.[0];
  expect(bare?.prompt).toContain("declares no verifiers in its manifests");

  await mkdir(path.join(h.workspacePath, ".kontext"));
  await writeFile(
    path.join(h.workspacePath, ".kontext", "verifiers.json"),
    JSON.stringify({
      schemaVersion: 1,
      verifiers: [{ kind: "lint", ref: "pnpm run check:code-quality:changed", command: "pnpm" }],
    }),
  );
  await h.git(["add", "--", ".kontext/verifiers.json"]);
  await h.git([
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
    "verifiers",
  ]);
  const declared = { ...h.request, requestId: randomUUID() };
  expect((await h.operations.startPlan(declared)).created).toBe(true);
  await h.settled(declared.requestId);
  const sent = h.plan.mock.calls[1]?.[0];
  expect(sent?.prompt).toContain('[{"kind":"lint","ref":"pnpm run check:code-quality:changed"}]');
  expect(sent?.prompt).toContain("do not list them");
});

it("generates a provenance-backed proposal and creates a Task only after exact explicit approval", async () => {
  const h = await fixture();
  expect((await h.operations.startPlan(h.request)).created).toBe(true);
  const reviewed = await h.settled();
  expect(reviewed.status, reviewed.diagnostic).toBe("review");
  expect(reviewed.taskId).toBeUndefined();
  expect(h.plan).toHaveBeenCalledTimes(1);
  const call = h.plan.mock.calls[0];
  if (!call) throw new Error("Expected the planning adapter to be called");
  const sent = call[0];
  expect(sent).toMatchObject({
    executionRole: "planning",
    workspacePath: await realpath(h.workspacePath),
  });
  expect(sent).not.toHaveProperty("workItem");
  expect(sent.prompt).toContain("Keep the established total domain term");
  expect(sent.prompt).toContain(h.source.resourceId);
  expect(sent.prompt).toContain("contentHash");
  expect(h.adapter.start).not.toHaveBeenCalled();
  await expect(
    h.operations.approvePlan({ ...h.request, expectedPlanDigest: `sha256:${"0".repeat(64)}` }),
  ).rejects.toThrow("exact current proposal");
  const result = await h.operations.approvePlan({
    ...h.request,
    expectedPlanDigest: reviewed.planDigest ?? "missing",
  });
  expect(result.inspection.status).toBe("current");
  expect((await h.operations.inspectPlan(h.request)).taskId).toBe(result.taskId);
  expect(
    (await new FileTaskContextRepository(h.data).getInitialRegistration(result.taskId))?.owner
      .contextSelection?.sourceResourceIds,
  ).toEqual([h.source.resourceId]);
  const restarted = new LocalTaskPlanningOperations(h.data, [h.adapter], h.environment);
  expect(
    (
      await restarted.approvePlan({
        ...h.request,
        expectedPlanDigest: reviewed.planDigest ?? "missing",
      })
    ).taskId,
  ).toBe(result.taskId);
  expect(h.plan).toHaveBeenCalledTimes(1);
});

it("reserves concurrent retries once and reports lost process ownership as unverifiable without starting again", async () => {
  const h = await fixture();
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const original = h.plan.getMockImplementation();
  if (!original) throw new Error("Missing fixture planner");
  h.plan.mockImplementation(async (input) => {
    await gate;
    return original(input);
  });
  try {
    const requests = await Promise.all(
      Array.from({ length: 8 }, () => h.operations.startPlan(h.request)),
    );
    expect(requests.filter((result) => result.created)).toHaveLength(1);
    await vi.waitFor(() => expect(h.plan).toHaveBeenCalledTimes(1));
    const restarted = new LocalTaskPlanningOperations(h.data, [h.adapter], h.environment);
    expect((await restarted.startPlan(h.request)).plan.status).toBe("unverifiable");
    await expect(restarted.cancelPlan(h.request)).rejects.toThrow("unverifiable");
    await expect(restarted.startPlan({ ...h.request, goal: "different" })).rejects.toThrow(
      "different input",
    );
    expect(h.plan).toHaveBeenCalledTimes(1);
  } finally {
    release();
    await h.settled();
  }
});

it("refuses unshared sources and API authentication before provider dispatch", async () => {
  const h = await fixture();
  const source = await h.knowledge.inspectSource(h.source);
  await h.knowledge.setSourceSharing({
    resourceId: source.resourceId,
    expectedRevision: source.revision,
    expectedContentHash: source.contentHash,
    dataClassification: "internal",
    allowedRuntimeProviders: [],
  });
  await h.operations.startPlan(h.request);
  expect((await h.settled()).diagnostic).toContain("not shared");
  expect(h.plan).not.toHaveBeenCalled();
  const capability = await h.adapter.inspectCapabilities();
  vi.mocked(h.adapter.inspectCapabilities).mockResolvedValue({ ...capability, billingPath: "api" });
  const next = { ...h.request, requestId: randomUUID(), sourceResourceIds: [] };
  await h.operations.startPlan(next);
  await vi.waitFor(async () =>
    expect((await h.operations.inspectPlan(next)).status).toBe("failed"),
  );
  expect((await h.operations.inspectPlan(next)).diagnostic).toContain("API billing is not allowed");
  expect(h.plan).not.toHaveBeenCalled();
});

it("rejects source revocation and code changes after human review", async () => {
  const h = await fixture();
  await h.operations.startPlan(h.request);
  const reviewed = await h.settled();
  expect(reviewed.status).toBe("review");
  const source = await h.knowledge.inspectSource(h.source);
  await h.knowledge.setSourceSharing({
    resourceId: source.resourceId,
    expectedRevision: source.revision,
    expectedContentHash: source.contentHash,
    dataClassification: "internal",
    allowedRuntimeProviders: [],
  });
  await expect(
    h.operations.approvePlan({
      ...h.request,
      expectedPlanDigest: reviewed.planDigest ?? "missing",
    }),
  ).rejects.toThrow("planning context changed");
  expect((await h.operations.inspectPlan(h.request)).status).toBe("review");
  await writeFile(
    path.join(h.workspacePath, "index.ts"),
    "export function total() { return 2; }\n",
  );
  await expect(
    h.operations.approvePlan({
      ...h.request,
      expectedPlanDigest: reviewed.planDigest ?? "missing",
    }),
  ).rejects.toThrow("revision changed");
});

it("discards a proposal when source permissions change during generation", async () => {
  const h = await fixture();
  const original = h.plan.getMockImplementation();
  if (!original) throw new Error("Missing fixture planner");
  h.plan.mockImplementation(async (input) => {
    const source = await h.knowledge.inspectSource(h.source);
    await h.knowledge.setSourceSharing({
      resourceId: source.resourceId,
      expectedRevision: source.revision,
      expectedContentHash: source.contentHash,
      dataClassification: "internal",
      allowedRuntimeProviders: [],
    });
    return original(input);
  });
  await h.operations.startPlan(h.request);
  const result = await h.settled();
  expect(result.status).toBe("failed");
  expect(result.diagnostic).toContain("Planning context changed");
  expect(result.proposal).toBeUndefined();
});

it("propagates cancellation without registering a completed late proposal", async () => {
  const h = await fixture();
  const original = h.plan.getMockImplementation();
  if (!original) throw new Error("Missing fixture planner");
  h.plan.mockImplementation(async (input) => {
    await new Promise<void>((resolve) =>
      input.signal?.addEventListener("abort", () => resolve(), { once: true }),
    );
    return original(input);
  });
  await h.operations.startPlan(h.request);
  await vi.waitFor(() => expect(h.plan).toHaveBeenCalledTimes(1));
  await h.operations.cancelPlan(h.request);
  const result = await h.settled();
  expect(result.status).toBe("failed");
  expect(result.diagnostic).toContain("cancelled");
  expect(result.proposal).toBeUndefined();
  expect(h.adapter.start).not.toHaveBeenCalled();
});

it("preserves failed or ambiguous provider output without approving or leaking its body in diagnostics", async () => {
  const h = await fixture();
  h.plan.mockRejectedValue(new Error("PRIVATE SOURCE TEXT: uncertain transport"));
  await h.operations.startPlan(h.request);
  const result = await h.settled();
  expect(result.status).toBe("unverifiable");
  expect(result.diagnostic).not.toContain("PRIVATE SOURCE TEXT");
  expect(result.proposal).toBeUndefined();
  expect((await h.operations.startPlan(h.request)).created).toBe(false);
  expect(h.plan).toHaveBeenCalledTimes(1);
  const plans = await readdir(path.join(h.data, "task-plans"));
  const fileName = plans.find((name) => name.endsWith(".json"));
  if (!fileName) throw new Error("Missing fixture plan file");
  const file = path.join(h.data, "task-plans", fileName);
  const encoded = await readFile(file, "utf8");
  await writeFile(file, encoded.replace('"unverifiable"', '"review"'));
  await expect(h.operations.inspectPlan(h.request)).rejects.toThrow("integrity");
});

it("validates behavior ownership, exact paths, graph dependencies, and host-minted authority", () => {
  expect(taskPlanProposalSchema.safeParse(proposal).success).toBe(true);
  for (const change of [
    { allowedPaths: ["../index.ts"] },
    { allowedPaths: ["**/*.ts"] },
    { dependsOn: ["logic:total"] },
    { dependsOn: ["missing"] },
    { plannedSymbols: undefined },
    { capabilityId: "invented" },
    { plannedSymbolIds: ["other"] },
  ])
    expect(
      taskPlanProposalSchema.safeParse({
        ...proposal,
        logicPlans: [{ ...proposal.logicPlans[0], ...change }],
      }).success,
    ).toBe(false);
});
