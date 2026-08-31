import { describe, expect, it } from "vitest";
import {
  type TaskContextDigestInput,
  computeTaskContextDigest,
  createTaskContextSnapshot,
  isTaskContextSnapshotValid,
} from "../src/index.js";

const input: TaskContextDigestInput = {
  taskId: "task:1",
  baseCodeRevision: "base-commit",
  effectiveScopes: [
    { kind: "workspace", workspaceId: "workspace:kontext" },
    { kind: "organization", organizationId: "org:acme" },
  ],
  normativeRevisions: [
    { kind: "decision", recordId: "decision:runtime", revisionId: "decision:runtime@1" },
    { kind: "invariant", recordId: "invariant:no-io", revisionId: "invariant:no-io@3" },
  ],
  requiredEvidenceIds: ["evidence:2", "evidence:1"],
  sourceFreshnessDigest: "freshness:1",
};

describe("Task Context digest", () => {
  it("is stable when set-like inputs arrive in a different order", () => {
    const reordered: TaskContextDigestInput = {
      ...input,
      effectiveScopes: [...input.effectiveScopes].reverse(),
      normativeRevisions: [...input.normativeRevisions].reverse(),
      requiredEvidenceIds: [...input.requiredEvidenceIds].reverse(),
    };

    expect(computeTaskContextDigest(reordered)).toBe(computeTaskContextDigest(input));
  });

  it("changes when a normative revision changes", () => {
    const changed: TaskContextDigestInput = {
      ...input,
      normativeRevisions: input.normativeRevisions.map((revision) =>
        revision.kind === "invariant" ? { ...revision, revisionId: "invariant:no-io@4" } : revision,
      ),
    };

    expect(computeTaskContextDigest(changed)).not.toBe(computeTaskContextDigest(input));
  });

  it("does not collapse IDs that contain the key separator", () => {
    const left: TaskContextDigestInput = {
      ...input,
      normativeRevisions: [{ kind: "decision", recordId: "a:b", revisionId: "c" }],
    };
    const right: TaskContextDigestInput = {
      ...input,
      normativeRevisions: [{ kind: "decision", recordId: "a", revisionId: "b:c" }],
    };

    expect(computeTaskContextDigest(left)).not.toBe(computeTaskContextDigest(right));
  });

  it("detects snapshot content changed without a new digest", () => {
    const snapshot = createTaskContextSnapshot(input, "2026-08-28T01:00:00.000Z");
    const tampered = {
      ...snapshot,
      normativeRevisions: [
        {
          kind: "decision" as const,
          recordId: "decision:runtime",
          revisionId: "decision:runtime@2",
        },
      ],
    };

    expect(isTaskContextSnapshotValid(snapshot)).toBe(true);
    expect(isTaskContextSnapshotValid(tampered)).toBe(false);
  });
});
