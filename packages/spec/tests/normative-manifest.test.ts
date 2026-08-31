import { describe, expect, it } from "vitest";
import type { DecisionRevision, NormativeActivation, NormativeManifest } from "../src/index.js";
import {
  decodeNormativeManifest,
  encodeNormativeManifest,
  normativeManifestDigest,
  updateNormativeManifest,
  validateNormativeManifest,
} from "../src/index.js";

const first: DecisionRevision = {
  kind: "decision",
  organizationId: "org:acme",
  recordId: "decision:storage",
  revisionId: "revision:1",
  scope: { kind: "codebase", codebaseId: "codebase:example" },
  evidence: [{ evidenceId: "evidence:adr", sourceSpan: "ADR 0001" }],
  egress: {
    dataClassification: "internal",
    allowedRuntimeProviders: ["codex", "claude"],
  },
  authoredBy: "user:owner",
  authoredAt: "2026-08-28T00:00:00.000Z",
  statement: "Use PostgreSQL for structured state.",
};
const firstActivation: NormativeActivation = {
  organizationId: "org:acme",
  kind: "decision",
  recordId: "decision:storage",
  revisionId: "revision:1",
  scope: { kind: "codebase", codebaseId: "codebase:example" },
  state: "accepted",
  acceptedBy: "user:owner",
  acceptedAt: "2026-08-28T00:01:00.000Z",
  mergeCommit: "abc123",
};
const emptyManifest: NormativeManifest = {
  schemaVersion: 1,
  organizationId: "org:acme",
  revisions: [],
  activations: [],
};

describe("normative manifest", () => {
  it("round-trips deterministic Git content and digest", () => {
    const manifest = updateNormativeManifest(emptyManifest, first, firstActivation);
    const encoded = encodeNormativeManifest(manifest);

    expect(decodeNormativeManifest(encoded)).toEqual(manifest);
    expect(
      encodeNormativeManifest({ ...manifest, revisions: [...manifest.revisions].reverse() }),
    ).toBe(encoded);
    expect(normativeManifestDigest(decodeNormativeManifest(encoded))).toBe(
      normativeManifestDigest(manifest),
    );
  });

  it("preserves immutable history while moving the activation pointer", () => {
    const initial = updateNormativeManifest(emptyManifest, first, firstActivation);
    const second: DecisionRevision = {
      ...first,
      revisionId: "revision:2",
      supersedesRevisionId: first.revisionId,
      statement: "Use PostgreSQL plus object storage according to data shape.",
    };
    const updated = updateNormativeManifest(initial, second, {
      ...firstActivation,
      revisionId: second.revisionId,
      acceptedAt: "2026-08-28T01:00:00.000Z",
      mergeCommit: "def456",
    });

    expect(updated.revisions.map((revision) => revision.revisionId)).toEqual([
      "revision:1",
      "revision:2",
    ]);
    expect(updated.activations).toHaveLength(1);
    expect(updated.activations[0]?.revisionId).toBe("revision:2");
  });

  it("rejects dangling activation pointers and revisions without Evidence", () => {
    expect(
      validateNormativeManifest({
        ...emptyManifest,
        revisions: [{ ...first, evidence: [] }],
        activations: [{ ...firstActivation, revisionId: "missing" }],
      }).map((issue) => issue.code),
    ).toEqual(expect.arrayContaining(["missing_evidence", "missing_revision"]));
    expect(() =>
      decodeNormativeManifest(
        JSON.stringify({
          ...emptyManifest,
          revisions: [first],
          activations: [{ ...firstActivation, revisionId: "missing" }],
        }),
      ),
    ).toThrow("unknown revision");
  });

  it("rejects mutation of an existing immutable revision id", () => {
    const manifest = updateNormativeManifest(emptyManifest, first, firstActivation);
    expect(() =>
      updateNormativeManifest(
        manifest,
        { ...first, statement: "Mutated in place." },
        firstActivation,
      ),
    ).toThrow("Immutable normative revision collision");
  });
});
