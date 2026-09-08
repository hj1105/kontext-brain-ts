import { mkdir, mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { gzipSync } from "node:zlib";
import { afterEach, describe, expect, it } from "vitest";
import { FileResourceContentStore } from "../src/index.js";

const directories: string[] = [];

afterEach(async () => {
  await Promise.all(directories.splice(0).map((directory) => rm(directory, { recursive: true })));
});

describe("FileResourceContentStore", () => {
  it("stores one compressed object per Resource snapshot and restores its native chunks", async () => {
    const directory = await mkdtemp(join(tmpdir(), "kontext-content-"));
    directories.push(directory);
    const store = new FileResourceContentStore(directory);

    const key = await store.put({
      organizationId: "acme",
      resourceId: "notion:page-1",
      contentHash: "sha256",
      body: "Page body",
      chunks: { "block-1": "First block", "block-2": "Second block" },
    });

    expect(key.endsWith(".json.gz")).toBe(true);
    expect(await store.get(key)).toMatchObject({
      body: "Page body",
      chunks: { "block-1": "First block", "block-2": "Second block" },
    });
    expect(await countFiles(directory)).toBe(1);
  });

  it("rejects object keys that try to escape the configured root", async () => {
    const directory = await mkdtemp(join(tmpdir(), "kontext-content-"));
    directories.push(directory);
    const store = new FileResourceContentStore(directory);

    await expect(store.get("../../secret.json.gz")).rejects.toThrow("Invalid object key");
  });

  it("bounds every new path segment while preserving long native identities and organization separation", async () => {
    const directory = await mkdtemp(join(tmpdir(), "kontext-content-"));
    directories.push(directory);
    const store = new FileResourceContentStore(directory);
    const content = {
      organizationId: "org".repeat(300),
      resourceId: "근거/".repeat(500),
      contentHash: "hash".repeat(300),
      body: "Body",
      chunks: { section: "Body" },
    };
    const key = await store.put(content);
    expect(key.split("/")).toHaveLength(4);
    expect(key.split("/").every((segment) => segment.length < 100)).toBe(true);
    expect(await store.get(key)).toEqual(content);
    expect(await store.put(content)).toBe(key);
    const other = { ...content, organizationId: "other-org" };
    const otherKey = await store.put(other);
    expect(otherKey).not.toBe(key);
    expect(await store.get(otherKey)).toEqual(other);
  });

  it("continues reading and purging legacy three-segment keys without a migration", async () => {
    const directory = await mkdtemp(join(tmpdir(), "kontext-content-"));
    directories.push(directory);
    const content = {
      organizationId: "org",
      resourceId: "resource:one",
      contentHash: "sha256:old",
      body: "Legacy source",
      chunks: { section: "Legacy source" },
    };
    const key = [
      encodeURIComponent(content.organizationId),
      encodeURIComponent(content.resourceId),
      `${encodeURIComponent(content.contentHash)}.json.gz`,
    ].join("/");
    const file = join(directory, key);
    await mkdir(dirname(file), { recursive: true });
    await writeFile(file, gzipSync(JSON.stringify(content)));
    const store = new FileResourceContentStore(directory);
    expect(await store.get(key)).toEqual(content);
    await store.purge(key);
    expect(await store.get(key)).toBeNull();
  });
});

async function countFiles(directory: string): Promise<number> {
  let count = 0;
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (entry.isDirectory()) count += await countFiles(join(directory, entry.name));
    else count++;
  }
  return count;
}
