import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { LocalCodeConnector, createSourceConnector } from "../src/index.js";

const temporaryDirectories: string[] = [];
afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

async function repository(): Promise<string> {
  const root = await mkdtemp(path.join(tmpdir(), "kontext-local-code-"));
  temporaryDirectories.push(root);
  await mkdir(path.join(root, "src", "billing"), { recursive: true });
  await mkdir(path.join(root, "docs"));
  await mkdir(path.join(root, "node_modules", "dep"), { recursive: true });
  await writeFile(
    path.join(root, "src", "billing", "invoice.ts"),
    [
      "export function computeTotal(items: readonly number[]): number {",
      "  return items.reduce((sum, item) => sum + item, 0);",
      "}",
      "export class InvoiceLedger { record(total: number): void { void total; } }",
      "function internalHelper(): void {}",
      "void internalHelper;",
      "",
    ].join("\n"),
  );
  await writeFile(path.join(root, "src", "billing", "invoice.test.ts"), "export const t = 1;\n");
  await writeFile(path.join(root, "src", "types.d.ts"), "export declare const x: number;\n");
  await writeFile(
    path.join(root, "src", "pricing.py"),
    "def apply_discount(price, rate):\n    return price * (1 - rate)\n\nclass PriceBook:\n    pass\n",
  );
  await writeFile(path.join(root, "node_modules", "dep", "index.js"), "module.exports = 1;\n");
  await writeFile(path.join(root, "docs", "decisions.md"), "# Billing\n\nRound half up.\n");
  return root;
}

describe("LocalCodeConnector", () => {
  it("describes each source file by its language and exported behaviour", async () => {
    const root = await repository();
    const resources = await new LocalCodeConnector("handbook", root).listResources();
    expect(resources.map((resource) => resource.id).sort()).toEqual([
      "src/billing/invoice.ts",
      "src/pricing.py",
    ]);
    const invoice = resources.find((resource) => resource.id === "src/billing/invoice.ts");
    expect(invoice?.description).toContain("typescript");
    expect(invoice?.description).toContain("computeTotal");
    expect(invoice?.description).toContain("InvoiceLedger");
    expect(invoice?.description).not.toContain("internalHelper");
    expect(invoice?.mimeType).toBe("text/x-typescript");
    const pricing = resources.find((resource) => resource.id === "src/pricing.py");
    expect(pricing?.description).toContain("python");
  });

  it("reads a file back by id and refuses paths outside the root", async () => {
    const root = await repository();
    const connector = new LocalCodeConnector("handbook", root);
    const fetched = await connector.fetchResource("src/pricing.py");
    expect(fetched.content).toContain("apply_discount");
    expect(fetched.metadata).toMatchObject({ source: "handbook", language: "python" });
    await expect(connector.fetchResource("../etc/passwd")).rejects.toThrow(/outside root/);
    const hits = await connector.search("PriceBook");
    expect(hits.map((hit) => hit.resourceId)).toEqual(["src/pricing.py"]);
  });

  it("keeps test files only when asked", async () => {
    const root = await repository();
    const withTests = await new LocalCodeConnector("handbook", root, {
      includeTests: true,
    }).listResources();
    expect(withTests.map((resource) => resource.id)).toContain("src/billing/invoice.test.ts");
  });
});

describe("createSourceConnector with code", () => {
  it("reads documents and code as one source, routing fetches by file type", async () => {
    const root = await repository();
    const connector = createSourceConnector({
      name: "handbook",
      transport: "local",
      path: root,
      code: true,
    });
    const resources = await connector.listResources();
    expect(resources.map((resource) => resource.id).sort()).toEqual([
      "docs/decisions.md",
      "src/billing/invoice.ts",
      "src/pricing.py",
    ]);
    expect((await connector.fetchResource("docs/decisions.md")).content).toContain("Round half up");
    expect((await connector.fetchResource("src/pricing.py")).content).toContain("apply_discount");
    expect((await connector.search("discount")).map((hit) => hit.resourceId)).toEqual([
      "src/pricing.py",
    ]);
  });

  it("stays Markdown-only without the flag", async () => {
    const root = await repository();
    const connector = createSourceConnector({ name: "handbook", transport: "local", path: root });
    expect((await connector.listResources()).map((resource) => resource.id)).toEqual([
      "docs/decisions.md",
    ]);
  });
});
