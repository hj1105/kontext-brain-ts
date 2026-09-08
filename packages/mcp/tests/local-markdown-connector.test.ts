import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { LocalMarkdownConnector } from "../src/local-markdown-connector.js";

const roots: string[] = [];

function makeRepo(): string {
  const root = mkdtempSync(join(tmpdir(), "kontext-local-md-"));
  roots.push(root);
  mkdirSync(join(root, "docs"));
  mkdirSync(join(root, "node_modules"));
  writeFileSync(
    join(root, "docs", "retry.md"),
    "# Retry policy\n\nRequests retry twice, then surface the failure.\n",
  );
  writeFileSync(join(root, "docs", "notes.txt"), "not markdown");
  writeFileSync(join(root, "README.md"), "no heading here, just prose\n");
  writeFileSync(join(root, "node_modules", "dep.md"), "# Dependency\n");
  return root;
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("LocalMarkdownConnector", () => {
  it("lists Markdown only, skipping excluded trees and other extensions", async () => {
    const resources = await new LocalMarkdownConnector("docs", makeRepo()).listResources();
    expect(resources.map((r) => r.id).sort()).toEqual(["README.md", "docs/retry.md"]);
  });

  it("names a document by its first heading and falls back to the path", async () => {
    const resources = await new LocalMarkdownConnector("docs", makeRepo()).listResources();
    const byId = new Map(resources.map((r) => [r.id, r]));
    expect(byId.get("docs/retry.md")?.name).toBe("Retry policy");
    expect(byId.get("README.md")?.name).toBe("README.md");
    expect(byId.get("docs/retry.md")?.description).toContain("retry twice");
  });

  it("restricts a walk to the requested subdirectories", async () => {
    const resources = await new LocalMarkdownConnector("docs", makeRepo(), {
      include: ["docs"],
    }).listResources();
    expect(resources.map((r) => r.id)).toEqual(["docs/retry.md"]);
  });

  it("fetches a document by its listed id", async () => {
    const root = makeRepo();
    const data = await new LocalMarkdownConnector("docs", root).fetchResource("docs/retry.md");
    expect(data.content).toContain("Retry policy");
    expect(data.metadata.path).toBe("docs/retry.md");
  });

  it("refuses an id that escapes the root", async () => {
    const connector = new LocalMarkdownConnector("docs", makeRepo());
    await expect(connector.fetchResource("../../etc/passwd")).rejects.toThrow(/outside root/);
  });

  it("searches document text and reports nothing for an empty query", async () => {
    const connector = new LocalMarkdownConnector("docs", makeRepo());
    expect((await connector.search("retry twice")).map((d) => d.resourceId)).toEqual([
      "docs/retry.md",
    ]);
    expect(await connector.search("   ")).toEqual([]);
  });
});
