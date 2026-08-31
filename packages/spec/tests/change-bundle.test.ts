import { describe, expect, it } from "vitest";
import { createChangeBundle, isChangeBundleValid } from "../src/index.js";

const input = {
  taskId: "task:bundle",
  workItemId: "work-item:bundle",
  baseRevision: "commit:base",
  resultRevision: "commit:result",
  taskContextDigest: "context:current",
  patchDigest: "sha256:patch",
  changedSymbolIds: ["symbol:b", "symbol:a"],
  changedPaths: ["src/b.ts", "src/a.ts"],
  contextReceiptIds: ["receipt:1"],
  evidenceIds: ["evidence:1"],
  normativeRevisions: [
    { kind: "decision" as const, recordId: "decision:1", revisionId: "decision:1@1" },
  ],
  verificationRunIds: ["verification:1"],
  proposals: [],
  unresolved: [],
  submittedAt: "2026-08-28T08:00:00.000Z",
};

describe("Change Bundle", () => {
  it("canonicalizes set-like fields into one immutable content ID", () => {
    const first = createChangeBundle(input);
    const second = createChangeBundle({
      ...input,
      changedSymbolIds: [...input.changedSymbolIds].reverse(),
      changedPaths: [...input.changedPaths].reverse(),
    });

    expect(first).toEqual(second);
    expect(isChangeBundleValid(first)).toBe(true);
  });

  it("detects bundle content changed without issuing a new ID", () => {
    const bundle = createChangeBundle(input);

    expect(isChangeBundleValid({ ...bundle, resultRevision: "commit:tampered" })).toBe(false);
  });
});
