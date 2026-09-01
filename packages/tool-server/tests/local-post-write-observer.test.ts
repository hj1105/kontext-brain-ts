import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  InMemoryPreparedTaskContextStore,
  InMemoryTaskContextStateProvider,
  createContextReceipt,
} from "@kontext-brain/context";
import { FileQuarantineStore } from "@kontext-brain/local";
import {
  type TaskContextSnapshot,
  type TaskContract,
  computeTaskContextDigest,
} from "@kontext-brain/spec";
import { afterEach, describe, expect, it } from "vitest";
import {
  FileWriteAuthorizationEventStore,
  InMemoryWriteAuthorizationBindingStore,
  LocalPostWriteObserver,
  captureWorkspaceSnapshot,
} from "../src/index.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("LocalPostWriteObserver", () => {
  it("accepts an exact authorized patch and quarantines a later unobserved out-of-scope write", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "kontext-post-write-"));
    temporaryDirectories.push(root);
    const workspacePath = path.join(root, "workspace");
    await mkdir(path.join(workspacePath, "src"), { recursive: true });
    await writeFile(
      path.join(workspacePath, "src", "handler.ts"),
      "export function handler() { return 1; }\n",
    );
    await writeFile(path.join(workspacePath, "src", "outside.ts"), "export const outside = 1;\n");
    initializeGit(workspacePath);

    const taskId = "task:post-write";
    const workItemId = "work-item:handler";
    const contract: TaskContract = {
      taskId,
      intent: "Observe every write after it happens.",
      acceptance: [
        {
          criterionId: "acceptance:observer",
          statement: "Observed writes remain in scope.",
          verifier: { kind: "test", ref: "observer:test" },
        },
      ],
      nonGoals: [],
      targets: ["symbol:handler"],
      risk: "low",
    };
    const snapshotInput = {
      taskId,
      baseCodeRevision: "commit:base",
      effectiveScopes: [{ kind: "personal" as const, subjectId: "user:local" }],
      normativeRevisions: [],
      requiredEvidenceIds: [],
      sourceFreshnessDigest: "freshness:current",
    };
    const snapshot: TaskContextSnapshot = {
      ...snapshotInput,
      contextDigest: computeTaskContextDigest(snapshotInput),
      createdAt: "2026-08-29T00:00:00.000Z",
    };
    const receipt = createContextReceipt({
      snapshot,
      logic: { workItemId, plannedSymbolIds: ["symbol:handler"] },
      allowedPaths: ["src/handler.ts"],
      evidenceIds: [],
      issuedAt: "2026-08-29T00:00:00.000Z",
      expiresAt: "2026-08-29T02:00:00.000Z",
    });
    const current = new InMemoryTaskContextStateProvider();
    current.set(taskId, {
      codeRevision: "commit:base",
      sourceFreshnessDigest: snapshot.sourceFreshnessDigest,
      effectiveScopes: snapshot.effectiveScopes,
      normativeRecords: [],
      normativeRevisionCatalog: [],
      conflicts: [],
      evidence: [],
      logicPlans: [
        {
          workItemId,
          plannedSymbolIds: ["symbol:handler"],
          allowedPaths: ["src/handler.ts"],
        },
      ],
    });
    const prepared = new InMemoryPreparedTaskContextStore();
    await prepared.put({ contract, snapshot, additionalRequiredEvidenceIds: [] });
    const baseline = await captureWorkspaceSnapshot(workspacePath, ["src/handler.ts"]);
    const bindings = new InMemoryWriteAuthorizationBindingStore();
    await bindings.put(workspacePath, {
      request: {
        taskId,
        logic: { workItemId, plannedSymbolIds: ["symbol:handler"] },
        runtimeProvider: "codex",
        issuedAt: receipt.issuedAt,
        expiresAt: receipt.expiresAt,
        totalTokenBudget: 10_000,
        optionalEvidenceTokenBudget: 1_000,
      },
      allowedPaths: [path.join(workspacePath, "src", "handler.ts")],
      receipt,
      baseline,
    });
    const events = new FileWriteAuthorizationEventStore(root);
    await events.put({
      toolUseId: "tool:allowed",
      workspacePath,
      taskId,
      workItemId,
      receiptId: receipt.receiptId,
      contextDigest: receipt.contextDigest,
      baselineRevision: baseline.revision,
      authorizedPaths: ["src/handler.ts"],
      authorizedAt: "2026-08-29T00:30:00.000Z",
    });
    const quarantine = new FileQuarantineStore(root);
    const observer = new LocalPostWriteObserver(current, prepared, bindings, events, quarantine);

    await writeFile(
      path.join(workspacePath, "src", "handler.ts"),
      "export function handler() { return 2; }\n",
    );
    const allowed = await observer.observe({
      cwd: workspacePath,
      toolName: "workspace_poll",
      observedAt: "2026-08-29T00:31:00.000Z",
    });
    expect(allowed.assessment).toEqual({ quarantined: false });
    expect(allowed.changedPaths).toEqual(["src/handler.ts"]);
    expect(allowed.preWriteAuthorizationObserved).toBe(true);

    await writeFile(path.join(workspacePath, "src", "outside.ts"), "export const outside = 2;\n");
    const unobserved = await observer.observe({
      cwd: workspacePath,
      toolName: "Bash",
      observedAt: "2026-08-29T00:32:00.000Z",
    });
    expect(unobserved.assessment.record?.reasons).toEqual(
      expect.arrayContaining(["path_out_of_scope", "unobserved_write"]),
    );
    expect(await quarantine.list("active")).toEqual([
      expect.objectContaining({ quarantineId: unobserved.assessment.record?.quarantineId }),
    ]);
  });
});

function initializeGit(workspacePath: string): void {
  execFileSync("git", ["init", "-q"], { cwd: workspacePath });
  execFileSync("git", ["add", "."], { cwd: workspacePath });
  execFileSync(
    "git",
    [
      "-c",
      "user.name=Kontext Test",
      "-c",
      "user.email=kontext@example.invalid",
      "commit",
      "-qm",
      "baseline",
    ],
    { cwd: workspacePath },
  );
}
