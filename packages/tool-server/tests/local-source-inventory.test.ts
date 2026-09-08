import { copyFile, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { SqliteKnowledgeGraphRepository } from "@kontext-brain/core";
import { afterEach, expect, it } from "vitest";
import { LocalKnowledgeOperations } from "../src/local-knowledge-operations.js";
import { loadLocalKnowledgePrincipal } from "../src/local-knowledge-principal.js";
import { LocalSourceRegistry } from "../src/local-source-registry.js";

const directories: string[] = [];
afterEach(async () => {
  await Promise.all(directories.splice(0).map((dir) => rm(dir, { recursive: true })));
});
async function fixture() {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-source-inventory-"));
  directories.push(directory);
  const data = path.join(directory, "data");
  const workspacePath = path.join(directory, "workspace");
  await mkdir(workspacePath);
  const operations = new LocalKnowledgeOperations(data);
  const add = async (relativePath: string) => {
    await writeFile(path.join(workspacePath, relativePath), "## Note\nPrivate fixture body\n");
    return operations.registerMarkdownSource({ workspacePath, relativePath });
  };
  return { directory, data, workspacePath, operations, add };
}
it("lists only metadata across restart and exact-digest pages without recapturing files or granting permission", async () => {
  const h = await fixture();
  expect(await h.operations.listSources({})).toMatchObject({
    sources: [],
    nextCursor: null,
    observation: "saved_metadata_only",
  });
  const a = await h.add("a.md");
  const b = await h.add("b.md");
  const c = await h.add("c.md");
  const first = await h.operations.listSources({ limit: 2 });
  expect(first.sources).toHaveLength(2);
  expect(first.nextCursor).not.toBeNull();
  const restarted = new LocalKnowledgeOperations(h.data);
  const second = await restarted.listSources({ limit: 2, cursor: first.nextCursor ?? undefined });
  expect(second.sources).toHaveLength(1);
  expect(second.nextCursor).toBeNull();
  expect(second.inventoryDigest).toBe(first.inventoryDigest);
  expect([...first.sources, ...second.sources].map((source) => source.resourceId).sort()).toEqual(
    [a.resourceId, b.resourceId, c.resourceId].sort(),
  );
  expect(
    first.sources.every(
      (source) => source.sharing === null && source.normativeApproval === "not_granted",
    ),
  ).toBe(true);
  expect(JSON.stringify(first)).not.toContain("Private fixture body");
  await rm(path.join(h.workspacePath, "a.md"));
  expect(await restarted.listSources({ limit: 2 })).toEqual(first);
});
it("rejects an old cursor when sharing or captured content changes, without changing the caller's selection", async () => {
  const h = await fixture();
  const source = await h.add("a.md");
  await h.add("b.md");
  const before = await h.operations.listSources({ limit: 1 });
  await h.operations.setSourceSharing({
    resourceId: source.resourceId,
    expectedRevision: 1,
    expectedContentHash: source.contentHash,
    dataClassification: "internal",
    allowedRuntimeProviders: ["codex"],
  });
  await expect(
    h.operations.listSources({ cursor: before.nextCursor ?? undefined }),
  ).rejects.toThrow("changed");
  const granted = await h.operations.listSources({ limit: 1 });
  expect(granted.inventoryDigest).not.toBe(before.inventoryDigest);
  await writeFile(path.join(h.workspacePath, "a.md"), "## Note\nUpdated\n");
  await h.operations.refreshSource(source);
  await expect(
    h.operations.listSources({ cursor: granted.nextCursor ?? undefined }),
  ).rejects.toThrow("changed");
});
it("omits revoked, purged and other-principal registrations without exposing their titles or paths", async () => {
  const h = await fixture();
  const visible = await h.add("visible.md");
  const revoked = await h.add("secret.md");
  const purged = await h.add("purged.md");
  const principal = await loadLocalKnowledgePrincipal(h.data);
  const registry = new LocalSourceRegistry(h.data);
  const record = await registry.get(principal, visible.resourceId);
  if (!record) throw new Error("Missing fixture source");
  await registry.save({
    ...record,
    subjectId: "other-subject",
    resourceId: "other-resource",
    relativePath: "foreign.md",
    audit: record.audit.map((event) => ({ ...event, actor: "other-subject" })),
  });
  const graph = await SqliteKnowledgeGraphRepository.open(h.data);
  await graph.transaction(principal.organizationId, async (tx) => {
    for (const [source, status] of [
      [revoked, "active"],
      [purged, "purged"],
    ] as const) {
      const resource = await tx.getResource(source.resourceId);
      if (!resource) throw new Error("Missing fixture graph resource");
      await tx.saveResource({ ...resource, status, acl: { subjectIds: ["other-subject"] } });
    }
  });
  const result = await h.operations.listSources({});
  expect(result.sources.map((source) => source.resourceId)).toEqual([visible.resourceId]);
  expect(JSON.stringify(result)).not.toMatch(/secret\.md|purged\.md|foreign\.md|other-subject/);
});
it("refuses corrupt registry files instead of reporting an empty inventory, and leaves the file intact", async () => {
  const h = await fixture();
  await h.add("a.md");
  const directory = path.join(h.data, "knowledge", "source-registry");
  const [entry] = await readdir(directory);
  if (!entry) throw new Error("Missing fixture registration");
  const file = path.join(directory, entry);
  await writeFile(file, "corrupt fixture");
  await expect(h.operations.listSources({})).rejects.toThrow();
  expect(await readFile(file, "utf8")).toBe("corrupt fixture");
});
it("refuses a registration copied to a different identity slot", async () => {
  const h = await fixture();
  await h.add("a.md");
  const directory = path.join(h.data, "knowledge", "source-registry");
  const [entry] = await readdir(directory);
  if (!entry) throw new Error("Missing fixture registration");
  await copyFile(path.join(directory, entry), path.join(directory, `${"a".repeat(64)}.json`));
  await expect(h.operations.listSources({})).rejects.toThrow("location mismatch");
});
it("validates pagination without honoring caller-authored ownership or arbitrary cursors", async () => {
  const h = await fixture();
  for (const input of [
    { limit: 0 },
    { limit: 101 },
    { subjectId: "someone-else" },
    { cursor: { digest: "wrong", offset: 1 } },
  ]) {
    await expect(
      h.operations.listSources(input as Parameters<LocalKnowledgeOperations["listSources"]>[0]),
    ).rejects.toThrow();
  }
});
