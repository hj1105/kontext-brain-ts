import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { CurrentTaskContextState } from "@kontext-brain/context";
import { TaskContextWorkflow } from "@kontext-brain/context";
import { FileTaskContextRepository } from "@kontext-brain/local";
import { afterEach, expect, it } from "vitest";
import { inspectRuntimeTask } from "../src/runtime-task-inspection.js";

const directories: string[] = [];
afterEach(async () => {
  await Promise.all(directories.splice(0).map((dir) => rm(dir, { recursive: true })));
});
async function fixture() {
  const dir = await mkdtemp(path.join(tmpdir(), "kontext-task-inspection-"));
  directories.push(dir);
  const repository = new FileTaskContextRepository(dir);
  const current: CurrentTaskContextState = {
    codeRevision: "revision:one",
    sourceFreshnessDigest: "fresh:one",
    effectiveScopes: [],
    normativeRecords: [],
    normativeRevisionCatalog: [],
    conflicts: [],
    evidence: [
      {
        evidenceId: "evidence:one",
        text: "Private Evidence text must not be returned",
        availability: "current",
        allowedRuntimeProviders: ["codex"],
      },
    ],
    logicPlans: [
      { workItemId: "logic:one", plannedSymbolIds: ["planned:one"], allowedPaths: ["src/one.ts"] },
    ],
  };
  await repository.publishCurrent("task:one", current);
  const workflow = new TaskContextWorkflow(repository, repository);
  return {
    repository,
    current,
    prepare: () =>
      workflow.prepareTask({
        contract: {
          taskId: "task:one",
          intent: "Implement one behavior",
          acceptance: [
            {
              criterionId: "criterion:one",
              statement: "Behavior passes",
              verifier: { kind: "test", ref: "workspace:test" },
            },
          ],
          nonGoals: [],
          targets: ["planned:one"],
          risk: "low",
        },
        additionalRequiredEvidenceIds: ["evidence:one"],
        createdAt: new Date().toISOString(),
      }),
  };
}
it("reads current planned metadata without disclosing Evidence text or changing prepared state", async () => {
  const h = await fixture();
  const before = await h.prepare();
  const result = await inspectRuntimeTask("task:one", h.repository, h.repository);
  expect(result).toMatchObject({
    taskId: "task:one",
    status: "current",
    requiredEvidenceIds: ["evidence:one"],
    logic: [{ workItemId: "logic:one", allowedPaths: ["src/one.ts"] }],
  });
  expect(JSON.stringify(result)).not.toContain("Private Evidence text");
  expect(await h.repository.get("task:one")).toEqual(before);
});
it("does not invent a Task Contract for published but unprepared state", async () => {
  const h = await fixture();
  expect(await inspectRuntimeTask("task:one", h.repository, h.repository)).toMatchObject({
    status: "unprepared",
    contract: null,
    contextDigest: null,
  });
  expect(await h.repository.get("task:one")).toBeUndefined();
});
it("reports a changed code revision as stale", async () => {
  const h = await fixture();
  await h.prepare();
  await h.repository.publishCurrent("task:one", { ...h.current, codeRevision: "revision:two" });
  expect((await inspectRuntimeTask("task:one", h.repository, h.repository)).status).toBe("stale");
});
it.each(["contract", "snapshot"] as const)(
  "rejects a mismatched prepared %s identity",
  async (field) => {
    const h = await fixture();
    const prepared = await h.prepare();
    await expect(
      inspectRuntimeTask("task:one", h.repository, {
        get: async () => ({ ...prepared, [field]: { ...prepared[field], taskId: "task:other" } }),
        put: async () => {
          throw new Error("Inspection must not write");
        },
      }),
    ).rejects.toThrow("different task");
  },
);
it("reports inaccessible required Evidence instead of allowing execution", async () => {
  const h = await fixture();
  await h.prepare();
  await h.repository.publishCurrent("task:one", {
    ...h.current,
    evidence: h.current.evidence.map((evidence) => ({ ...evidence, availability: "inaccessible" })),
  });
  expect((await inspectRuntimeTask("task:one", h.repository, h.repository)).status).toBe(
    "inaccessible",
  );
});
