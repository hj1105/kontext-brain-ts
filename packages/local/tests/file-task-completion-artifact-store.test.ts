import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { createAccuracyManifest, createChangeBundle } from "@kontext-brain/spec";
import { afterEach, describe, expect, it } from "vitest";
import { FileTaskCompletionArtifactStore } from "../src/index.js";

const temporaryDirectories: string[] = [];
const taskId = "task:completion-store";
const run = {
  verificationRunId: "verification:1",
  tier: "targeted" as const,
  verifierKind: "test" as const,
  verifierRef: "package:test",
  codeRevision: "commit:result",
  contextDigest: "context:current",
  subjectIds: [taskId],
  result: "passed" as const,
  observedAt: "2026-08-28T12:00:00.000Z",
};
const bundle = createChangeBundle({
  taskId,
  workItemId: "work-item:store",
  baseRevision: "commit:base",
  resultRevision: run.codeRevision,
  taskContextDigest: run.contextDigest,
  patchDigest: "sha256:patch",
  changedSymbolIds: ["symbol:store"],
  changedPaths: ["src/store.ts"],
  contextReceiptIds: ["receipt:store"],
  evidenceIds: ["evidence:store"],
  normativeRevisions: [],
  verificationRunIds: [run.verificationRunId],
  proposals: [],
  unresolved: [],
  submittedAt: "2026-08-28T12:01:00.000Z",
});
const finding = {
  findingId: "review-finding:1",
  status: "open" as const,
  codeRevision: run.codeRevision,
  contextDigest: run.contextDigest,
  message: "The changed symbol lacks a boundary test.",
  reviewerProvider: "claude" as const,
  authorProviders: ["codex" as const],
  reviewedAt: "2026-08-28T12:01:30.000Z",
  symbolId: "symbol:store",
  ruleRef: "acceptance:store",
  evidenceIds: ["evidence:store"],
};
const manifest = createAccuracyManifest({
  taskId,
  taskContractDigest: "sha256:contract",
  contextDigest: run.contextDigest,
  baseCodeRevision: bundle.baseRevision,
  resultCodeRevision: run.codeRevision,
  normativeRevisions: [],
  evidenceIds: bundle.evidenceIds,
  workItemIds: [bundle.workItemId],
  changeBundleIds: [bundle.bundleId],
  changedSymbolIds: bundle.changedSymbolIds,
  verificationRunIds: [run.verificationRunId],
  reviewFindingIds: [],
  emergencyBypassIds: [],
  createdAt: "2026-08-28T12:02:00.000Z",
});

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("FileTaskCompletionArtifactStore", () => {
  it("persists Verification Runs, Change Bundles, Review Findings, and Accuracy Manifest across restarts", async () => {
    const directory = await temporaryDirectory();
    const first = new FileTaskCompletionArtifactStore(directory);
    await first.putVerificationRuns(taskId, [run]);
    await first.putChangeBundle(bundle);
    await first.putReviewFindings(taskId, [finding]);
    await first.putAccuracyManifest(manifest);
    const reopened = new FileTaskCompletionArtifactStore(directory);

    expect(await reopened.listVerificationRuns(taskId)).toEqual([run]);
    expect(await reopened.listChangeBundles(taskId)).toEqual([bundle]);
    expect(await reopened.listReviewFindings(taskId)).toEqual([finding]);
    expect(await reopened.getAccuracyManifest(taskId)).toEqual(manifest);
    expect((await stat(reopened.filePath(taskId))).mode & 0o777).toBe(0o600);
  });

  it("detects persisted artifact tampering", async () => {
    const store = new FileTaskCompletionArtifactStore(await temporaryDirectory());
    await store.putVerificationRuns(taskId, [run]);
    const filePath = store.filePath(taskId);
    const envelope = JSON.parse(await readFile(filePath, "utf8"));
    envelope.payload.verificationRuns[0].codeRevision = "commit:tampered";
    await writeFile(filePath, JSON.stringify(envelope), "utf8");

    await expect(store.listVerificationRuns(taskId)).rejects.toThrow("digest mismatch");
  });

  it("allows an open Review Finding to resolve once and keeps terminal evidence immutable", async () => {
    const store = new FileTaskCompletionArtifactStore(await temporaryDirectory());
    await store.putReviewFindings(taskId, [finding]);
    const resolved = {
      ...finding,
      status: "resolved" as const,
      resolutionMessage: "A boundary test now covers the behavior.",
      resolvedByProvider: "claude" as const,
      resolvedAt: "2026-08-28T12:03:00.000Z",
    };

    await expect(store.putReviewFindings(taskId, [resolved])).resolves.toEqual([resolved]);
    await expect(
      store.putReviewFindings(taskId, [{ ...resolved, resolutionMessage: "changed" }]),
    ).rejects.toThrow("immutable");
  });
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-completion-store-"));
  temporaryDirectories.push(directory);
  return directory;
}
