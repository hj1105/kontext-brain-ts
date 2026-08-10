import { mkdtemp, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
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
});

async function countFiles(directory: string): Promise<number> {
  let count = 0;
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (entry.isDirectory()) count += await countFiles(join(directory, entry.name));
    else count++;
  }
  return count;
}
