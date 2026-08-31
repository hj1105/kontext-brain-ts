import { describe, expect, it } from "vitest";
import {
  type CheckChangeRequest,
  type KontextCompletionOperations,
  KontextCompletionToolRouter,
  type ProposeTransitionRequest,
  type SubmitChangeBundleRequest,
} from "../src/index.js";

class RecordingCompletionOperations implements KontextCompletionOperations {
  checked?: CheckChangeRequest;
  submitted?: SubmitChangeBundleRequest;
  proposed?: ProposeTransitionRequest;

  async checkChange(request: CheckChangeRequest): Promise<unknown> {
    this.checked = request;
    return { runs: [] };
  }

  async submitChangeBundle(request: SubmitChangeBundleRequest): Promise<unknown> {
    this.submitted = request;
    return { accepted: true, issues: [] };
  }

  async proposeTransition(request: ProposeTransitionRequest): Promise<unknown> {
    this.proposed = request;
    return { state: "awaiting_evidence", issues: [] };
  }
}

const taskId = "task:completion-tools";
const contextDigest = "context:current";

describe("KontextCompletionToolRouter", () => {
  it("validates and forwards check and immutable bundle handoff inputs", async () => {
    const operations = new RecordingCompletionOperations();
    const router = new KontextCompletionToolRouter(operations);
    await router.checkChange({
      taskId,
      workItemId: "work-item:handler",
      workspacePath: "/workspace",
      tier: "targeted",
      affectedSymbolIds: ["symbol:handler"],
      codeRevision: "commit:result",
      contextDigest,
      observedAt: "2026-08-28T11:00:00.000Z",
      nextAttemptAt: "2026-08-28T11:01:00.000Z",
    });
    await router.submitChangeBundle({
      bundle: {
        taskId,
        workItemId: "work-item:handler",
        baseRevision: "commit:base",
        resultRevision: "commit:result",
        taskContextDigest: contextDigest,
        patchDigest: "sha256:patch",
        changedSymbolIds: ["symbol:handler"],
        changedPaths: ["src/handler.ts"],
        contextReceiptIds: ["context-receipt:handler"],
        evidenceIds: ["evidence:decision"],
        normativeRevisions: [],
        verificationRunIds: ["verification:handler"],
        proposals: [],
        unresolved: [],
        submittedAt: "2026-08-28T11:02:00.000Z",
      },
      observedPatch: {
        patchDigest: "sha256:patch",
        changedPaths: ["src/handler.ts"],
        changedSymbolIds: ["symbol:handler"],
      },
      receipts: [
        {
          receiptId: "context-receipt:handler",
          taskId,
          workItemId: "work-item:handler",
          plannedSymbolIds: ["symbol:handler"],
          allowedPaths: ["src/handler.ts"],
          contextDigest,
          normativeRevisions: [],
          evidenceIds: ["evidence:decision"],
          issuedAt: "2026-08-28T11:00:00.000Z",
          expiresAt: "2026-08-28T12:00:00.000Z",
        },
      ],
    });

    expect(operations.checked?.tier).toBe("targeted");
    expect(operations.submitted?.bundle.workItemId).toBe("work-item:handler");

    await expect(
      router.submitChangeBundle({
        bundle: {
          ...operations.submitted?.bundle,
          bundleId: "change-bundle:caller-forged",
        },
        observedPatch: operations.submitted?.observedPatch,
        receipts: operations.submitted?.receipts,
      }),
    ).rejects.toThrow();
  });

  it("accepts Evidence for transition but rejects direct state or Accuracy Manifest injection", async () => {
    const operations = new RecordingCompletionOperations();
    const router = new KontextCompletionToolRouter(operations);
    const input = {
      taskId,
      currentState: "in_progress",
      workStarted: true,
      completionRequested: true,
      currentCodeRevision: "commit:result",
      context: { status: "current", contextDigest },
      evidence: [],
      invariantEvaluations: [],
      reviewFindings: [],
      requestedAt: "2026-08-28T11:03:00.000Z",
    };
    await router.proposeTransition(input);
    expect(operations.proposed?.completionRequested).toBe(true);

    await expect(router.proposeTransition({ ...input, state: "done" })).rejects.toThrow();
    await expect(
      router.proposeTransition({ ...input, accuracyManifest: { manifestId: "forged" } }),
    ).rejects.toThrow();
  });
});
