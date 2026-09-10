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
  it("exposes one module per directory, described by language and exported behaviour", async () => {
    const root = await repository();
    const resources = await new LocalCodeConnector("handbook", root).listResources();
    expect(resources.map((resource) => resource.id).sort()).toEqual(["src/", "src/billing/"]);
    const billing = resources.find((resource) => resource.id === "src/billing/");
    expect(billing?.name).toBe("src/billing");
    expect(billing?.description).toContain("typescript");
    expect(billing?.description).toContain("1 file");
    expect(billing?.description).toContain("computeTotal");
    expect(billing?.description).toContain("InvoiceLedger");
    expect(billing?.description).not.toContain("internalHelper");
    expect(billing?.mimeType).toBe("text/x-code-module");
    const src = resources.find((resource) => resource.id === "src/");
    expect(src?.description).toContain("python");
    expect(src?.description).toContain("apply_discount");
  });

  it("ranks modules by what they export and caps how many a source exposes", async () => {
    const root = await repository();
    const all = await new LocalCodeConnector("handbook", root).listResources();
    const exportsOf = (description: string): number =>
      (description.split("exports ")[1] ?? "").split(", ").filter(Boolean).length;
    const ranked = [...all].sort(
      (left, right) =>
        exportsOf(right.description) - exportsOf(left.description) ||
        left.id.length - right.id.length,
    );
    const [only] = await new LocalCodeConnector("handbook", root, {
      maxModules: 1,
    }).listResources();
    expect(only?.id).toBe(ranked[0]?.id);
    expect(all).toHaveLength(2);
  });

  it("reads a module back as its files and exports, and finds modules by content", async () => {
    const root = await repository();
    const connector = new LocalCodeConnector("handbook", root);
    const fetched = await connector.fetchResource("src/");
    expect(fetched.content).toContain("# src");
    expect(fetched.content).toContain("src/pricing.py: exports apply_discount, PriceBook");
    expect(fetched.metadata).toMatchObject({ source: "handbook", path: "src", files: "1" });
    await expect(connector.fetchResource("../etc/")).rejects.toThrow(/unknown module/);
    const hits = await connector.search("PriceBook");
    expect(hits.map((hit) => hit.resourceId)).toEqual(["src/"]);
  });

  it("keeps test files only when asked", async () => {
    const root = await repository();
    const withTests = await new LocalCodeConnector("handbook", root, {
      includeTests: true,
    }).fetchResource("src/billing/");
    expect(withTests.content).toContain("src/billing/invoice.test.ts");
  });
});

describe("LocalCodeConnector.syncCodeKnowledge", () => {
  it("projects each file into the graph at symbol level with its module's ontology nodes", async () => {
    const root = await repository();
    const snapshots: Array<{
      source: { connectorId: string; externalId: string; type: string };
      ontologyNodeIds?: readonly string[];
      entities?: readonly { name: string }[];
      chunks: readonly unknown[];
    }> = [];
    const report = await new LocalCodeConnector("handbook", root).syncCodeKnowledge({
      organizationId: "org",
      resourceSync: {
        async execute(snapshot) {
          snapshots.push(snapshot);
          return { resourceId: snapshot.source.externalId, changed: true } as never;
        },
        async remove() {
          return true;
        },
      },
      nodeIdsFor: (moduleId) => (moduleId === "src/billing/" ? ["Billing"] : ["Pricing"]),
    });
    expect(report).toEqual({ filesSynced: 2, filesFailed: 0 });
    const ids = snapshots
      .map((snapshot) => `${snapshot.source.connectorId}:${snapshot.source.externalId}`)
      .sort();
    expect(ids).toEqual(["code:handbook:src/billing/invoice.ts", "code:handbook:src/pricing.py"]);
    const invoice = snapshots.find((snapshot) => snapshot.source.externalId.endsWith("invoice.ts"));
    expect(invoice?.source.type).toBe("typescript-module");
    expect(invoice?.ontologyNodeIds).toEqual(["Billing"]);
    expect(invoice?.entities?.map((entity) => entity.name)).toEqual(
      expect.arrayContaining(["computeTotal", "InvoiceLedger"]),
    );
    expect(invoice?.chunks.length).toBeGreaterThan(1);
  });
});

describe("createSourceConnector with code", () => {
  it("reads documents and code modules as one source, routing fetches by id", async () => {
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
      "src/",
      "src/billing/",
    ]);
    expect((await connector.fetchResource("docs/decisions.md")).content).toContain("Round half up");
    expect((await connector.fetchResource("src/")).content).toContain("apply_discount");
    expect((await connector.search("discount")).map((hit) => hit.resourceId)).toEqual(["src/"]);
  });

  it("stays Markdown-only without the flag", async () => {
    const root = await repository();
    const connector = createSourceConnector({ name: "handbook", transport: "local", path: root });
    expect((await connector.listResources()).map((resource) => resource.id)).toEqual([
      "docs/decisions.md",
    ]);
  });
});
