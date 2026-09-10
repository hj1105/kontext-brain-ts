import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { LLMProviderRegistry } from "@kontext-brain/llm";
import { afterEach, describe, expect, it, vi } from "vitest";
import { loadLocalKnowledgePrincipal, resolveKontextDataDirectory } from "../src/index.js";
import { KontextLoader } from "../src/kontext-loader.js";
import { runOntologyCli } from "../src/ontology-cli.js";

const roots: string[] = [];
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
  vi.restoreAllMocks();
});

/** Only `invoke` is used by LangChainLLMAdapter; every document lands on one node. */
class OneNodeModel {
  async invoke(messages: ReadonlyArray<{ content: unknown }>): Promise<{ content: string }> {
    const text = messages.map((message) => String(message.content)).join("\n");
    if (text.includes("Extract topic categories")) return { content: '["Billing"]' };
    if (text.includes("Design ontology nodes")) {
      return {
        content: JSON.stringify({ nodes: [{ id: "Billing", description: "invoices", weight: 1 }] }),
      };
    }
    if (text.includes("Infer relationships")) return { content: JSON.stringify({ edges: [] }) };
    if (text.includes("Classify each document")) {
      const count = text.split("\n").filter((line) => /^\[\d+\]/.test(line)).length;
      return {
        content: JSON.stringify({
          mappings: { Billing: Array.from({ length: count }, (_, i) => i) },
          unmapped: [],
        }),
      };
    }
    throw new Error(`scripted model saw an unexpected prompt: ${text.slice(0, 120)}`);
  }
}

function registry(): LLMProviderRegistry {
  const llm = new LLMProviderRegistry();
  llm.register({
    providerName: "scripted",
    createChat: () => new OneNodeModel() as never,
    createEmbedding: () => {
      throw new Error("no embeddings");
    },
  });
  return llm;
}

describe("resolveKontextDataDirectory", () => {
  it("prefers the flag, then the sidecar's environment, else nothing", () => {
    expect(resolveKontextDataDirectory("/a", { KONTEXT_PLUGIN_DATA: "/b" })).toBe("/a");
    expect(resolveKontextDataDirectory(undefined, { KONTEXT_PLUGIN_DATA: "/b" })).toBe("/b");
    expect(resolveKontextDataDirectory(undefined, {})).toBeUndefined();
  });
});

describe("kontext-ontology setup with a data directory", () => {
  it("writes every document into the sidecar's knowledge graph under the same organization", async () => {
    const root = mkdtempSync(join(tmpdir(), "kontext-local-knowledge-"));
    roots.push(root);
    const data = join(root, "data");
    const docs = join(root, "docs");
    mkdirSync(docs, { recursive: true });
    writeFileSync(join(docs, "billing.md"), "# Billing\n\nRound half up. Retry payments twice.\n");
    writeFileSync(join(docs, "refunds.md"), "# Refunds\n\nRefund within 14 days.\n");
    mkdirSync(join(root, "src"), { recursive: true });
    writeFileSync(
      join(root, "src", "invoice.ts"),
      "export function computeInvoiceTotal(items: readonly number[]): number {\n  return items.reduce((sum, item) => sum + item, 0);\n}\n",
    );
    const config = join(root, "kontext.yaml");
    writeFileSync(
      config,
      [
        "llm:",
        "  traversal: {provider: scripted, model: x}",
        "  reasoning: {provider: scripted, model: x}",
        "mcp:",
        "  - name: handbook",
        "    transport: local",
        `    path: ${root}`,
        "    code: true",
        "",
      ].join("\n"),
    );
    const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
    const code = await runOntologyCli(
      ["setup", "--config", config, "--data-dir", data, "--write", "--json"],
      {
        // Why: the runtime is a loader option, not a fromYaml argument; the instance
        // method takes only the document.
        loadAgent: (_path, yaml, knowledge) =>
          new KontextLoader({
            llmRegistry: registry(),
            ...(knowledge ? { knowledgeRuntime: knowledge } : {}),
          }).fromYaml(yaml),
      },
    );
    const printed = out.mock.calls.map((call) => String(call[0])).join("");
    out.mockRestore();
    const result = JSON.parse(printed);
    expect(code, printed).toBe(0);
    expect(result.knowledgeStore).toBe(data);
    // Two Markdown documents and one code module document; one source file projected.
    expect(result.documentsClassified).toBe(3);
    expect(result.codeFilesSynced).toBe(1);

    // The Task sidecar opens the same graph with the same principal and finds the documents.
    const { SqliteKnowledgeGraphRepository } = await import("@kontext-brain/core");
    const principal = await loadLocalKnowledgePrincipal(data);
    const graph = await SqliteKnowledgeGraphRepository.open(data);
    const billing = await graph.listResourcesByOntologyNode(principal.organizationId, "Billing");
    expect(billing.map((resource) => resource.title).sort()).toEqual([
      "Billing",
      "Refunds",
      "src",
      "src/invoice.ts",
    ]);
    // The code file is its own Resource whose symbols are Entities, not just a module summary.
    const codeResource = await graph.getResourceBySource(principal.organizationId, {
      connectorId: "code",
      externalId: "handbook:src/invoice.ts",
      type: "typescript-module",
    });
    if (!codeResource) throw new Error("expected the code file to be its own Resource");
    // Why mentions rather than listEntitiesForResource: an exported symbol is a global
    // Entity (deterministic promotion), so it is reached through its mention in this file.
    const codeChunks = await graph.listChunks(principal.organizationId, codeResource.resourceId);
    const mentions = await graph.listEntityMentions(
      principal.organizationId,
      codeResource.resourceId,
    );
    expect(codeChunks.length).toBeGreaterThanOrEqual(2);
    expect(mentions.length).toBeGreaterThanOrEqual(1);
    expect(mentions.every((mention) => mention.entityId.includes("code-symbol:"))).toBe(true);
    const first = billing[0];
    if (!first) throw new Error("expected a Billing resource");
    const chunks = await graph.listChunks(principal.organizationId, first.resourceId);
    expect(chunks.length).toBeGreaterThan(0);
  });

  it("keeps working without a data directory, saving only the schema", async () => {
    const root = mkdtempSync(join(tmpdir(), "kontext-local-knowledge-none-"));
    roots.push(root);
    const docs = join(root, "docs");
    mkdirSync(docs, { recursive: true });
    writeFileSync(join(docs, "billing.md"), "# Billing\n\nRound half up.\n");
    const config = join(root, "kontext.yaml");
    writeFileSync(
      config,
      [
        "llm:",
        "  traversal: {provider: scripted, model: x}",
        "  reasoning: {provider: scripted, model: x}",
        "mcp:",
        "  - name: handbook",
        "    transport: local",
        `    path: ${docs}`,
        "",
      ].join("\n"),
    );
    const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
    const code = await runOntologyCli(["setup", "--config", config, "--json"], {
      loadAgent: (_path, yaml) => new KontextLoader({ llmRegistry: registry() }).fromYaml(yaml),
    });
    const printed = out.mock.calls.map((call) => String(call[0])).join("");
    out.mockRestore();
    expect(code, printed).toBe(0);
    expect(JSON.parse(printed).knowledgeStore).toBeNull();
  });
});
