import { randomUUID } from "node:crypto";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { NormativeManifest } from "@kontext-brain/spec";
import { afterEach, describe, expect, it } from "vitest";
import {
  FileLocalNormativeOverlayStore,
  InMemoryLocalNormativeOverlayStore,
  type LocalOverlayKey,
} from "../src/index.js";

const temporaryDirectories: string[] = [];
const key: LocalOverlayKey = {
  organizationId: "org:acme",
  subjectId: "user:owner",
  workspaceId: "workspace:local",
  codebaseId: "codebase:example",
};
const manifest: NormativeManifest = {
  schemaVersion: 1,
  organizationId: key.organizationId,
  revisions: [
    {
      kind: "decision",
      organizationId: key.organizationId,
      recordId: "decision:runtime",
      revisionId: "revision:local:1",
      scope: { kind: "workspace", workspaceId: key.workspaceId },
      evidence: [{ evidenceId: "evidence:session:1", sourceSpan: "decision 4" }],
      egress: {
        dataClassification: "internal",
        allowedRuntimeProviders: ["codex", "claude"],
      },
      authoredBy: key.subjectId,
      authoredAt: "2026-08-28T00:00:00.000Z",
      statement: "Use Codex CLI through public extension points.",
    },
  ],
  activations: [
    {
      organizationId: key.organizationId,
      kind: "decision",
      recordId: "decision:runtime",
      revisionId: "revision:local:1",
      scope: { kind: "workspace", workspaceId: key.workspaceId },
      state: "accepted_local",
      acceptedBy: key.subjectId,
      acceptedAt: "2026-08-28T00:01:00.000Z",
    },
  ],
};

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("FileLocalNormativeOverlayStore", () => {
  it("atomically stores a private, keyed manifest envelope and reads it back", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "kontext-local-overlay-"));
    temporaryDirectories.push(directory);
    const store = new FileLocalNormativeOverlayStore(directory);
    const written = await store.save(key, manifest);

    expect(written.created).toBe(true);
    expect(await store.load(key)).toEqual(manifest);
    expect(store.filePath(key)).not.toContain(key.organizationId);
    expect(store.filePath(key)).not.toContain(key.subjectId);
    expect((await stat(store.filePath(key))).mode & 0o777).toBe(0o600);
    const envelope = JSON.parse(await readFile(store.filePath(key), "utf8"));
    expect(envelope.manifest.revisions[0].evidence).toEqual([
      { evidenceId: "evidence:session:1", sourceSpan: "decision 4" },
    ]);
  });

  it("uses an expected digest to reject stale local writers", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "kontext-local-overlay-"));
    temporaryDirectories.push(directory);
    const store = new FileLocalNormativeOverlayStore(directory);
    const written = await store.save(key, manifest);

    expect((await store.save(key, manifest, { expectedDigest: written.digest })).created).toBe(
      false,
    );
    await expect(
      store.save(key, manifest, { expectedDigest: `sha256:${randomUUID()}` }),
    ).rejects.toThrow("changed since it was read");
  });
});

describe("local overlay policy", () => {
  it("rejects a managed activation in both file and in-memory adapters", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "kontext-local-overlay-"));
    temporaryDirectories.push(directory);
    const stores = [
      new FileLocalNormativeOverlayStore(directory),
      new InMemoryLocalNormativeOverlayStore(),
    ];
    const localActivation = manifest.activations[0];
    if (!localActivation) throw new Error("Test fixture requires a local activation");
    const managed: NormativeManifest = {
      ...manifest,
      activations: [
        {
          ...localActivation,
          state: "accepted",
          mergeCommit: "abc123",
        },
      ],
    };

    for (const store of stores) {
      await expect(store.save(key, managed)).rejects.toThrow("cannot contain managed activations");
    }
  });
});
