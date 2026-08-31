import { describe, expect, it } from "vitest";
import type { DecisionRevision, NormativeActivation, NormativeManifest } from "../src/index.js";
import { resolveNormativeLayers } from "../src/index.js";

function revision(
  revisionId: string,
  statement: string,
  scope: DecisionRevision["scope"],
): DecisionRevision {
  return {
    kind: "decision",
    organizationId: "org:acme",
    recordId: "decision:runtime",
    revisionId,
    scope,
    evidence: [{ evidenceId: `evidence:${revisionId}` }],
    egress: {
      dataClassification: "internal",
      allowedRuntimeProviders: ["codex", "claude"],
    },
    authoredBy: "user:owner",
    authoredAt: "2026-08-28T00:00:00.000Z",
    statement,
  };
}

function activation(
  value: DecisionRevision,
  state: NormativeActivation["state"],
): NormativeActivation {
  return {
    organizationId: value.organizationId,
    kind: value.kind,
    recordId: value.recordId,
    revisionId: value.revisionId,
    scope: value.scope,
    state,
    acceptedBy: "user:owner",
    acceptedAt: "2026-08-28T00:01:00.000Z",
    mergeCommit: state === "accepted" ? "abc123" : undefined,
  };
}

function manifest(value: DecisionRevision, state: NormativeActivation["state"]): NormativeManifest {
  return {
    schemaVersion: 1,
    organizationId: value.organizationId,
    revisions: [value],
    activations: [activation(value, state)],
  };
}

describe("resolveNormativeLayers", () => {
  it("replaces an identical Local Acceptance with the managed canonical activation", () => {
    const local = revision("revision:local", "Use Codex CLI through public extension points.", {
      kind: "workspace",
      workspaceId: "workspace:local",
    });
    const managed = revision("revision:managed", local.statement, {
      kind: "codebase",
      codebaseId: "codebase:example",
    });
    const resolved = resolveNormativeLayers(
      manifest(local, "accepted_local"),
      manifest(managed, "accepted"),
    );

    expect(resolved.contextStale).toBe(false);
    expect(resolved.canonicalizedLocalRevisionIds).toEqual(["revision:local"]);
    expect(resolved.effective).toEqual([
      expect.objectContaining({
        origin: "managed",
        revision: expect.objectContaining({ revisionId: "revision:managed" }),
      }),
    ]);
  });

  it("keeps a changed merge visible and marks dependent context stale", () => {
    const local = revision("revision:local", "Use Codex CLI.", {
      kind: "workspace",
      workspaceId: "workspace:local",
    });
    const managed = revision("revision:managed", "Use Codex App Server.", {
      kind: "codebase",
      codebaseId: "codebase:example",
    });
    const resolved = resolveNormativeLayers(
      manifest(local, "accepted_local"),
      manifest(managed, "accepted"),
    );

    expect(resolved.contextStale).toBe(true);
    expect(resolved.effective.map((record) => record.origin)).toEqual(["local", "managed"]);
    expect(resolved.conflicts).toEqual([
      {
        kind: "decision",
        recordId: "decision:runtime",
        localRevisionId: "revision:local",
        managedRevisionIds: ["revision:managed"],
      },
    ]);
  });

  it("keeps unrelated local-only acceptance effective offline", () => {
    const local = revision("revision:local", "Use Codex CLI.", {
      kind: "workspace",
      workspaceId: "workspace:local",
    });
    const emptyManaged: NormativeManifest = {
      schemaVersion: 1,
      organizationId: "org:acme",
      revisions: [],
      activations: [],
    };
    const resolved = resolveNormativeLayers(manifest(local, "accepted_local"), emptyManaged);

    expect(resolved.contextStale).toBe(false);
    expect(resolved.localOnlyRevisionIds).toEqual(["revision:local"]);
    expect(resolved.effective[0]?.origin).toBe("local");
  });
});
