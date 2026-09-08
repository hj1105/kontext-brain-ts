import { createHash } from "node:crypto";
import { access, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { TaskContextWorkflow } from "@kontext-brain/context";
import { afterEach, expect, it } from "vitest";
import {
  FileTaskContextRepository,
  type InitializeTaskRequest,
} from "../src/file-task-context-repository.js";

const dirs: string[] = [];
afterEach(async () => {
  await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true })));
});
const request: InitializeTaskRequest = {
  owner: {
    organizationId: "org",
    subjectId: "user",
    workspacePath: "/workspace",
    requestId: "01767302-bcdd-4fb4-b496-de660b10a853",
    requestDigest: `sha256:${"a".repeat(64)}`,
  },
  contract: {
    taskId: "task:initial",
    intent: "Create an evidence-backed task",
    acceptance: [
      { criterionId: "test", statement: "Pass", verifier: { kind: "test", ref: "test" } },
    ],
    nonGoals: [],
    targets: ["symbol:one"],
    risk: "low",
  },
  state: {
    codeRevision: "commit:one",
    sourceFreshnessDigest: "sources:one",
    effectiveScopes: [],
    normativeRecords: [],
    normativeRevisionCatalog: [],
    conflicts: [],
    evidence: [
      {
        evidenceId: "evidence:one",
        text: "Source text",
        availability: "current",
        allowedRuntimeProviders: ["codex"],
      },
    ],
    logicPlans: [
      { workItemId: "logic:one", plannedSymbolIds: ["symbol:one"], allowedPaths: ["src/one.ts"] },
    ],
  },
  additionalRequiredEvidenceIds: ["evidence:one"],
  createdAt: "2026-09-06T00:00:00.000Z",
};
async function fixture() {
  const dir = await mkdtemp(path.join(tmpdir(), "kontext-task-init-"));
  dirs.push(dir);
  return { dir, repo: new FileTaskContextRepository(dir) };
}

it("publishes the owner, current context and prepared Task together and reuses the existing compiler", async () => {
  const h = await fixture();
  const result = await h.repo.initializeTask(request);
  expect(result.created).toBe(true);
  const reopened = new FileTaskContextRepository(h.dir);
  expect(await reopened.getCurrent(request.contract.taskId)).toEqual(request.state);
  expect(await reopened.get(request.contract.taskId)).toEqual(result.registration.prepared);
  await expect(access(h.repo.currentStateFilePath(request.contract.taskId))).rejects.toThrow();
  await expect(access(h.repo.preparedTaskFilePath(request.contract.taskId))).rejects.toThrow();
  const context = await new TaskContextWorkflow(reopened, reopened).beginLogic({
    taskId: request.contract.taskId,
    logic: { workItemId: "logic:one", plannedSymbolIds: ["symbol:one"] },
    runtimeProvider: "codex",
    issuedAt: request.createdAt,
    expiresAt: "2026-09-06T00:10:00.000Z",
    totalTokenBudget: 4096,
    optionalEvidenceTokenBudget: 0,
  });
  expect(context.editingAllowed).toBe(true);
  expect(context.mandatory.evidence[0]?.text).toBe("Source text");
});

it("accepts one initial publication under contention and returns identical duplicates without replacing it", async () => {
  const h = await fixture();
  const results = await Promise.all(
    Array.from({ length: 12 }, () => new FileTaskContextRepository(h.dir).initializeTask(request)),
  );
  expect(results.filter((item) => item.created)).toHaveLength(1);
  expect(new Set(results.map((item) => JSON.stringify(item.registration))).size).toBe(1);
  for (const owner of [
    { ...request.owner, subjectId: "other" },
    { ...request.owner, requestDigest: `sha256:${"b".repeat(64)}` },
  ]) {
    await expect(h.repo.initializeTask({ ...request, owner })).rejects.toThrow("conflicts");
  }
  expect((await h.repo.getInitialRegistration(request.contract.taskId))?.owner).toEqual(
    request.owner,
  );
});

it("does not overwrite existing legacy context or treat seeded state as absent during CAS", async () => {
  const h = await fixture();
  await h.repo.publishCurrent(request.contract.taskId, request.state);
  await expect(h.repo.initializeTask(request)).rejects.toThrow("already has context");
  const fresh = await fixture();
  await fresh.repo.initializeTask(request);
  const observed = await fresh.repo.getCurrentVersion(request.contract.taskId);
  await expect(
    fresh.repo.publishCurrent(request.contract.taskId, request.state, { expectedDigest: null }),
  ).rejects.toThrow("changed");
  const updated = { ...request.state, codeRevision: "commit:two" };
  expect(
    await fresh.repo.publishCurrent(request.contract.taskId, updated, {
      expectedDigest: observed.digest,
    }),
  ).toMatchObject({ created: false });
  expect((await fresh.repo.getCurrent(request.contract.taskId)).codeRevision).toBe("commit:two");
  expect((await fresh.repo.get(request.contract.taskId))?.snapshot.baseCodeRevision).toBe(
    "commit:one",
  );
  await fresh.repo.initializeTask(request);
  expect((await fresh.repo.getCurrent(request.contract.taskId)).codeRevision).toBe("commit:two");
});

it("never masks corrupted ownership behind later current/prepared overrides", async () => {
  const h = await fixture();
  const initial = await h.repo.initializeTask(request);
  await h.repo.publishCurrent(request.contract.taskId, request.state);
  await h.repo.put(initial.registration.prepared);
  const file = path.join(
    h.dir,
    "task-context",
    "initial",
    `${createHash("sha256").update(request.contract.taskId).digest("hex")}.json`,
  );
  await writeFile(file, "corrupt initial registration");
  await expect(h.repo.getCurrent(request.contract.taskId)).rejects.toThrow();
  await expect(h.repo.get(request.contract.taskId)).rejects.toThrow();
  await expect(h.repo.initializeTask(request)).rejects.toThrow();
  expect(await readFile(file, "utf8")).toBe("corrupt initial registration");
});

it("rejects concurrent legacy publishing versus initialization without exposing a half-created Task", async () => {
  const h = await fixture();
  const result = await Promise.allSettled([
    h.repo.initializeTask(request),
    h.repo.publishCurrent(request.contract.taskId, request.state, { expectedDigest: null }),
  ]);
  expect(result.filter((item) => item.status === "fulfilled")).toHaveLength(1);
  const initial = await h.repo.getInitialRegistration(request.contract.taskId);
  expect(await h.repo.get(request.contract.taskId)).toEqual(initial?.prepared);
});
