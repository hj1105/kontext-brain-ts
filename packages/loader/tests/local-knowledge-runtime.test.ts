import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path, { join } from "node:path";
import { type TextEmbedder, normalizeVector } from "@kontext-brain/core";
import { LLMProviderRegistry } from "@kontext-brain/llm";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  loadLocalKnowledgePrincipal,
  ontologyProgressPath,
  resolveKontextDataDirectory,
} from "../src/index.js";
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

/** Concept axes stand in for a model: refund words share one axis whatever the wording. */
const toyEmbedder: TextEmbedder = {
  model: "toy:v1",
  async embed(texts) {
    return texts.map((text) => {
      const vector = new Float32Array(3);
      const lower = text.toLowerCase();
      if (/refund|money back/.test(lower)) vector[0] = 1;
      if (/invoice|billing/.test(lower)) vector[1] = 1;
      vector[2] = 0.1;
      return normalizeVector(vector);
    });
  },
};
const withToyEmbedder = { createEmbedder: () => toyEmbedder };

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
        loadAgent: (_path, yaml, knowledge, buildProgress) =>
          new KontextLoader({
            llmRegistry: registry(),
            ...(knowledge ? { knowledgeRuntime: knowledge } : {}),
            ...(buildProgress ? { buildProgress } : {}),
          }).fromYaml(yaml),
        ...withToyEmbedder,
      },
    );
    const printed = out.mock.calls.map((call) => String(call[0])).join("");
    out.mockRestore();
    const result = JSON.parse(printed);
    expect(code, printed).toBe(0);
    expect(result.knowledgeStore).toBe(data);
    // Every chunk the sync wrote got a vector in the configured (default, built-in) space.
    expect(result.embedding.provider).toBe("builtin");
    expect(result.embeddingError).toBeNull();
    expect(result.chunksEmbedded).toBeGreaterThan(0);
    expect(JSON.parse(readFileSync(path.join(data, "embedding.json"), "utf8")).provider).toBe(
      "builtin",
    );
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

    // The same graph answers a question with Evidence-cited chunks, filtered by node.
    const asked = vi.spyOn(process.stdout, "write").mockReturnValue(true);
    const queryCode = await runOntologyCli(
      [
        "query",
        "--data-dir",
        data,
        "--question",
        "money back within days",
        "--node",
        "Billing",
        "--json",
      ],
      withToyEmbedder,
    );
    const answer = JSON.parse(asked.mock.calls.map((call) => String(call[0])).join(""));
    asked.mockRestore();
    expect(queryCode).toBe(0);
    expect(answer.hits[0]?.source.externalId).toBe("docs/refunds.md");
    expect(answer.hits[0]?.evidenceId).toContain("|source|");
    // "money back" shares no word with the refund page; the vector side found it.
    expect(answer.mode).toBe("hybrid");
    expect(answer.hits[0]?.similarity).toBeGreaterThan(0.9);

    // The build left a finished progress record where a host polls for it.
    const progress = JSON.parse(readFileSync(ontologyProgressPath(data, config), "utf8"));
    expect(progress.finished).toBe(true);
    expect(progress.ok).toBe(true);
    // The last phase a build reports is embedding, after sync and code projection.
    expect(progress.phase).toBe("embed");
    expect(progress.total).toBeGreaterThan(0);

    // Nodes list what the graph filed under each of them.
    const listed = vi.spyOn(process.stdout, "write").mockReturnValue(true);
    const nodesCode = await runOntologyCli([
      "nodes",
      "--config",
      config,
      "--data-dir",
      data,
      "--json",
    ]);
    const nodesResult = JSON.parse(listed.mock.calls.map((call) => String(call[0])).join(""));
    listed.mockRestore();
    expect(nodesCode).toBe(0);
    expect(nodesResult.nodes.map((node: { id: string }) => node.id)).toEqual(["Billing"]);
    expect(nodesResult.nodes[0].resourceCount).toBe(4);
    expect(nodesResult.nodes[0].samples.map((sample: { title: string }) => sample.title)).toEqual(
      expect.arrayContaining(["Billing", "Refunds", "src", "src/invoice.ts"]),
    );
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
