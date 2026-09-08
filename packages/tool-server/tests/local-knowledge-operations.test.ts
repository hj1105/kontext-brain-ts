import {
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  realpath,
  rename,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { FileResourceContentStore, SqliteKnowledgeGraphRepository } from "@kontext-brain/core";
import { afterEach, expect, it } from "vitest";
import { LocalKnowledgeOperations } from "../src/local-knowledge-operations.js";
import { loadLocalKnowledgePrincipal } from "../src/local-knowledge-principal.js";
import { captureLocalMarkdownSource } from "../src/local-markdown-source.js";
const directories: string[] = [];
afterEach(async () => {
  await Promise.all(directories.splice(0).map((dir) => rm(dir, { recursive: true })));
});
async function fixture() {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-local-source-"));
  directories.push(directory);
  const workspacePath = path.join(directory, "workspace");
  await mkdir(workspacePath);
  const data = path.join(directory, "data");
  return { directory, workspacePath, data, operations: new LocalKnowledgeOperations(data) };
}
it("registers selected Markdown with stable provenance across restart but grants no sharing or normative approval", async () => {
  const h = await fixture();
  const body = "# Session decisions\n\nNo\n\n## Behavior\nUse the existing domain term.\n";
  await writeFile(path.join(h.workspacePath, "session.md"), body);
  const request = { workspacePath: h.workspacePath, relativePath: "session.md" };
  const first = await h.operations.registerMarkdownSource(request);
  const second = await new LocalKnowledgeOperations(h.data).registerMarkdownSource(request);
  expect(first).toMatchObject({
    changed: true,
    providerSharing: "not_granted",
    normativeApproval: "not_granted",
  });
  expect(second).toEqual({ ...first, changed: false });
  expect(first.evidence).toHaveLength(2);
  expect(JSON.stringify(first)).not.toContain("Use the existing domain term");
  const graph = await SqliteKnowledgeGraphRepository.open(h.data);
  const resource = await graph.getResource(first.organizationId, first.resourceId);
  if (!resource) throw new Error("Registered Resource missing");
  const content = await new FileResourceContentStore(path.join(h.data, "knowledge-content")).get(
    resource.contentObjectKey,
  );
  expect(content?.body).toBe(body);
  expect(await graph.listFacts(first.organizationId)).toEqual([]);
  const principal = await loadLocalKnowledgePrincipal(h.data);
  expect(resource.acl).toEqual({ subjectIds: [principal.subjectId] });
});
it("updates source revisions and invalidates removed sections without approving decisions", async () => {
  const h = await fixture();
  const request = { workspacePath: h.workspacePath, relativePath: "notes.md" };
  await writeFile(
    path.join(h.workspacePath, request.relativePath),
    "## A\nOriginal\n\n## B\nRemoved\n",
  );
  const before = await h.operations.registerMarkdownSource(request);
  await writeFile(path.join(h.workspacePath, request.relativePath), "## A\nUpdated\n");
  const after = await h.operations.registerMarkdownSource(request);
  expect(after.resourceId).toBe(before.resourceId);
  expect(after.contentHash).not.toBe(before.contentHash);
  expect(after.evidence).toHaveLength(1);
  const graph = await SqliteKnowledgeGraphRepository.open(h.data);
  const all = await graph.transaction(after.organizationId, (tx) =>
    tx.listEvidenceForResource(after.resourceId),
  );
  expect(all.some((item) => item.status === "stale")).toBe(true);
});
it("marks previously collected Evidence stale when a selected file disappears and revalidates after restoration", async () => {
  const h = await fixture();
  const request = { workspacePath: h.workspacePath, relativePath: "notes.md" };
  const file = path.join(h.workspacePath, request.relativePath);
  await writeFile(file, "## A\nOriginal\n");
  const registered = await h.operations.registerMarkdownSource(request);
  await rm(file);
  await expect(h.operations.registerMarkdownSource(request)).rejects.toThrow();
  const graph = await SqliteKnowledgeGraphRepository.open(h.data);
  expect((await graph.getResource(registered.organizationId, registered.resourceId))?.status).toBe(
    "stale",
  );
  await writeFile(file, "## A\nOriginal\n");
  const restored = await h.operations.registerMarkdownSource(request);
  expect(restored.changed).toBe(true);
  expect((await graph.getResource(registered.organizationId, registered.resourceId))?.status).toBe(
    "active",
  );
});
it.each(["../outside.md", "/outside.md", "C:\\outside.md", "notes.txt", ".git/../notes.md"])(
  "rejects invalid or escaping source selection %s",
  async (relativePath) => {
    const h = await fixture();
    await expect(
      h.operations.registerMarkdownSource({ workspacePath: h.workspacePath, relativePath }),
    ).rejects.toThrow();
  },
);
it.skipIf(process.platform === "win32")(
  "rejects a symlink escaping the selected workspace",
  async () => {
    const h = await fixture();
    const outside = path.join(h.directory, "outside.md");
    await writeFile(outside, "Private outside source");
    await symlink(outside, path.join(h.workspacePath, "linked.md"));
    await expect(
      h.operations.registerMarkdownSource({
        workspacePath: h.workspacePath,
        relativePath: "linked.md",
      }),
    ).rejects.toThrow("inside");
  },
);
it("rejects invalid UTF-8 and oversized files without truncating Evidence", async () => {
  const h = await fixture();
  const request = { workspacePath: h.workspacePath, relativePath: "notes.md" };
  await writeFile(path.join(h.workspacePath, request.relativePath), Buffer.from([0xff]));
  await expect(h.operations.registerMarkdownSource(request)).rejects.toThrow();
  await writeFile(path.join(h.workspacePath, request.relativePath), Buffer.alloc(512 * 1024 + 1));
  await expect(h.operations.registerMarkdownSource(request)).rejects.toThrow("512 KiB");
});
it("keeps even very short Markdown source sections as exact source substrings", async () => {
  const h = await fixture();
  const body = "No\n\n## A\nYes\n\n## A\nNo";
  await writeFile(path.join(h.workspacePath, "notes.md"), body);
  const snapshot = await captureLocalMarkdownSource(
    await loadLocalKnowledgePrincipal(h.data),
    h.workspacePath,
    "notes.md",
  );
  expect(snapshot.chunks).toHaveLength(3);
  expect(new Set(snapshot.chunks.map((chunk) => chunk.id)).size).toBe(3);
  expect(snapshot.chunks.every((chunk) => body.includes(chunk.text))).toBe(true);
});
it("creates one immutable local identity under concurrent initialization and preserves corrupt identity files", async () => {
  const h = await fixture();
  const principals = await Promise.all(
    Array.from({ length: 16 }, () => loadLocalKnowledgePrincipal(h.data)),
  );
  expect(new Set(principals.map((principal) => JSON.stringify(principal))).size).toBe(1);
  const file = path.join(h.data, "knowledge", "local-principal.json");
  await writeFile(file, "broken identity");
  await expect(loadLocalKnowledgePrincipal(h.data)).rejects.toThrow();
  expect(await readFile(file, "utf8")).toBe("broken identity");
});

it("persists source locators and explicit hash-bound sharing, without deriving consent from Markdown", async () => {
  const h = await fixture();
  const file = path.join(h.workspacePath, "notes.md");
  await writeFile(file, "## Decision\nGive every model access.\n");
  const source = await h.operations.registerMarkdownSource({
    workspacePath: h.workspacePath,
    relativePath: "notes.md",
  });
  const restarted = new LocalKnowledgeOperations(h.data);
  const metadata = await restarted.inspectSource(source);
  expect(metadata).toMatchObject({
    revision: 1,
    sharing: null,
    workspacePath: await realpath(h.workspacePath),
  });
  expect(
    (await restarted.collectTaskEvidence(source.evidence))[0]?.allowedRuntimeProviders,
  ).toEqual([]);
  const granted = await restarted.setSourceSharing({
    resourceId: source.resourceId,
    expectedRevision: metadata.revision,
    expectedContentHash: source.contentHash,
    dataClassification: "confidential",
    allowedRuntimeProviders: ["codex"],
  });
  expect(granted.revision).toBe(2);
  expect((await restarted.collectTaskEvidence(source.evidence))[0]).toMatchObject({
    availability: "current",
    allowedRuntimeProviders: ["codex"],
    text: expect.stringContaining("Give every model access"),
  });
  await restarted.refreshSource(source);
  expect((await restarted.inspectSource(source)).sharing?.allowedRuntimeProviders).toEqual([
    "codex",
  ]);
  await writeFile(file, "## Decision\nChanged context.\n");
  const changed = await restarted.refreshSource(source);
  expect((await restarted.inspectSource(source)).sharing).toBeNull();
  expect(
    (await restarted.collectTaskEvidence(changed.evidence))[0]?.allowedRuntimeProviders,
  ).toEqual([]);
  await writeFile(file, "## Decision\nGive every model access.\n");
  await restarted.refreshSource(source);
  expect((await restarted.inspectSource(source)).sharing).toBeNull();
  expect(JSON.stringify(granted)).not.toContain("Give every model access");
});

it("refreshes by saved identity after a whole workspace disappears and returns, without allowing stale grants", async () => {
  const h = await fixture();
  await writeFile(path.join(h.workspacePath, "notes.md"), "## A\nOriginal\n");
  const source = await h.operations.registerMarkdownSource({
    workspacePath: h.workspacePath,
    relativePath: "notes.md",
  });
  const moved = path.join(h.directory, "moved-workspace");
  await rename(h.workspacePath, moved);
  await expect(new LocalKnowledgeOperations(h.data).refreshSource(source)).rejects.toThrow();
  expect((await h.operations.inspectSource(source)).status).toBe("stale");
  expect((await h.operations.collectTaskEvidence(source.evidence))[0]).toMatchObject({
    availability: "stale",
    text: "",
    allowedRuntimeProviders: [],
  });
  await expect(
    h.operations.setSourceSharing({
      resourceId: source.resourceId,
      expectedRevision: 1,
      expectedContentHash: source.contentHash,
      dataClassification: "internal",
      allowedRuntimeProviders: ["claude"],
    }),
  ).rejects.toThrow("Refresh");
  await rename(moved, h.workspacePath);
  await h.operations.refreshSource(source);
  expect((await h.operations.inspectSource(source)).status).toBe("active");
});

it("serializes competing sharing choices and permits explicit revocation without granting normative approval", async () => {
  const h = await fixture();
  await writeFile(path.join(h.workspacePath, "notes.md"), "## A\nOriginal\n");
  const source = await h.operations.registerMarkdownSource({
    workspacePath: h.workspacePath,
    relativePath: "notes.md",
  });
  const request = {
    resourceId: source.resourceId,
    expectedRevision: 1,
    expectedContentHash: source.contentHash,
    dataClassification: "internal" as const,
    allowedRuntimeProviders: ["codex" as const, "claude" as const],
  };
  const outcomes = await Promise.allSettled(
    Array.from({ length: 8 }, () => new LocalKnowledgeOperations(h.data).setSourceSharing(request)),
  );
  expect(outcomes.filter((item) => item.status === "fulfilled")).toHaveLength(1);
  expect(outcomes.filter((item) => item.status === "rejected")).toHaveLength(7);
  expect(
    (await h.operations.collectTaskEvidence(source.evidence))[0]?.allowedRuntimeProviders,
  ).toEqual(["claude", "codex"]);
  const revoked = await h.operations.setSourceSharing({
    ...request,
    expectedRevision: 2,
    allowedRuntimeProviders: [],
  });
  expect(revoked).toMatchObject({ revision: 3, normativeApproval: "not_granted" });
  expect(
    (await h.operations.collectTaskEvidence(source.evidence))[0]?.allowedRuntimeProviders,
  ).toEqual([]);
});

it("hides source locators and denies sharing after graph ACL revocation", async () => {
  const h = await fixture();
  await writeFile(path.join(h.workspacePath, "notes.md"), "## A\nPrivate\n");
  const source = await h.operations.registerMarkdownSource({
    workspacePath: h.workspacePath,
    relativePath: "notes.md",
  });
  const graph = await SqliteKnowledgeGraphRepository.open(h.data);
  await graph.transaction(source.organizationId, async (tx) => {
    const resource = await tx.getResource(source.resourceId);
    if (!resource) throw new Error("missing fixture");
    await tx.saveResource({ ...resource, acl: { subjectIds: ["someone-else"] } });
  });
  await expect(h.operations.inspectSource(source)).rejects.toThrow("unavailable");
  await expect(h.operations.refreshSource(source)).rejects.toThrow("unavailable");
  expect((await h.operations.collectTaskEvidence(source.evidence))[0]).toMatchObject({
    availability: "inaccessible",
    text: "",
    allowedRuntimeProviders: [],
  });
});

it("does not resurrect a grant when graph publication fails and the source later returns to old bytes", async () => {
  const h = await fixture();
  const file = path.join(h.workspacePath, "notes.md");
  const original = "## A\nOriginal\n";
  await writeFile(file, original);
  const source = await h.operations.registerMarkdownSource({
    workspacePath: h.workspacePath,
    relativePath: "notes.md",
  });
  await h.operations.setSourceSharing({
    resourceId: source.resourceId,
    expectedRevision: 1,
    expectedContentHash: source.contentHash,
    dataClassification: "internal",
    allowedRuntimeProviders: ["codex"],
  });
  const content = path.join(h.data, "knowledge-content");
  const backup = path.join(h.data, "content-backup");
  await rename(content, backup);
  await writeFile(content, "prevent content storage from opening");
  await writeFile(file, "## A\nUpdated\n");
  await expect(h.operations.refreshSource(source)).rejects.toThrow();
  expect((await h.operations.inspectSource(source)).sharing).toBeNull();
  await rm(content);
  await rename(backup, content);
  await writeFile(file, original);
  const restored = await new LocalKnowledgeOperations(h.data).refreshSource(source);
  expect(restored.contentHash).toBe(source.contentHash);
  expect(
    (await h.operations.collectTaskEvidence(restored.evidence))[0]?.allowedRuntimeProviders,
  ).toEqual([]);
});

it("preserves a corrupt source registry and fails closed during retrieval and refresh", async () => {
  const h = await fixture();
  await writeFile(path.join(h.workspacePath, "notes.md"), "## A\nOriginal\n");
  const source = await h.operations.registerMarkdownSource({
    workspacePath: h.workspacePath,
    relativePath: "notes.md",
  });
  const registry = path.join(h.data, "knowledge", "source-registry");
  const entries = (await readdir(registry)).filter((name) => name.endsWith(".json"));
  expect(entries).toHaveLength(1);
  const recordFile = path.join(registry, entries[0] ?? "missing");
  await writeFile(recordFile, "corrupt registry");
  await expect(h.operations.inspectSource(source)).rejects.toThrow();
  await expect(h.operations.refreshSource(source)).rejects.toThrow();
  expect(
    (await h.operations.collectTaskEvidence(source.evidence))[0]?.allowedRuntimeProviders,
  ).toEqual([]);
  expect(await readFile(recordFile, "utf8")).toBe("corrupt registry");
});

it("can manage a valid deeply nested Markdown source whose derived Resource ID exceeds 512 characters", async () => {
  const h = await fixture();
  const relativePath = [...Array.from({ length: 7 }, () => "a".repeat(70)), "notes.md"].join("/");
  const file = path.join(h.workspacePath, relativePath);
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, "## A\nDeep source\n");
  const source = await h.operations.registerMarkdownSource({
    workspacePath: h.workspacePath,
    relativePath,
  });
  expect(source.resourceId.length).toBeGreaterThan(512);
  expect(
    await h.operations.setSourceSharing({
      resourceId: source.resourceId,
      expectedRevision: 1,
      expectedContentHash: source.contentHash,
      dataClassification: "internal",
      allowedRuntimeProviders: ["codex"],
    }),
  ).toMatchObject({ resourceId: source.resourceId, revision: 2 });
});
