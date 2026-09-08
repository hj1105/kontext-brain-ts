import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { LLMProviderRegistry } from "@kontext-brain/llm";
import { afterEach, describe, expect, it, vi } from "vitest";
import { parse } from "yaml";
import { KontextLoader } from "../src/kontext-loader.js";
import { runOntologyCli } from "../src/ontology-cli.js";

/**
 * Exercises `setup` end to end — real config file, real local-Markdown connector,
 * real autoSetup, real YAML write — with a scripted model standing in for a paid
 * one. The stub answers by prompt content rather than call order so a change in
 * the pipeline's call sequence surfaces as a failed assertion, not a silent pass.
 */

const roots: string[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
  vi.restoreAllMocks();
});

const NODES = JSON.stringify({
  nodes: [
    { id: "Reliability", description: "retries and failure handling", weight: 1 },
    { id: "Authentication", description: "sessions and expiry", weight: 1 },
  ],
});
const EDGES = JSON.stringify({
  edges: [{ from: "Reliability", to: "Authentication", weight: 0.4 }],
});
const CLASSIFICATION = JSON.stringify({
  mappings: { Reliability: [0], Authentication: [1] },
  unmapped: [],
});

/** Only `invoke` is used by LangChainLLMAdapter, so that is all this stands up. */
class ScriptedChatModel {
  readonly seen: string[] = [];

  async invoke(messages: ReadonlyArray<{ content: unknown }>): Promise<{ content: string }> {
    const text = messages.map((message) => String(message.content)).join("\n");
    this.seen.push(text);
    if (text.includes("Extract topic categories")) {
      return { content: '["Reliability", "Authentication"]' };
    }
    if (text.includes("Design ontology nodes")) return { content: NODES };
    if (text.includes("Infer relationships")) return { content: EDGES };
    if (text.includes("Classify each document")) return { content: CLASSIFICATION };
    if (text.includes("Create new ontology nodes"))
      return { content: JSON.stringify({ nodes: [] }) };
    throw new Error(`scripted model saw an unexpected prompt: ${text.slice(0, 120)}`);
  }
}

function scriptedRegistry(model: ScriptedChatModel): LLMProviderRegistry {
  const registry = new LLMProviderRegistry();
  registry.register({
    providerName: "scripted",
    createChat: () => model as never,
    createEmbedding: () => {
      // Why: without an embedding provider autoSetup takes the keyword mapping path,
      // which is what a first run against a fresh config actually does.
      throw new Error("no embeddings in this test");
    },
  });
  return registry;
}

function makeWorkspace(): { root: string; config: string } {
  const root = mkdtempSync(join(tmpdir(), "kontext-setup-"));
  roots.push(root);
  mkdirSync(join(root, "docs"));
  writeFileSync(
    join(root, "docs", "auth.md"),
    "# Session authentication\n\nSessions are validated on every request; expiry is not cached.\n",
  );
  writeFileSync(
    join(root, "docs", "retry.md"),
    "# Retry policy\n\nOutbound requests retry twice with backoff, then surface the failure.\n",
  );
  const config = join(root, "kontext.yaml");
  writeFileSync(
    config,
    [
      "llm:",
      "  traversal: {provider: scripted, model: scripted}",
      "  reasoning: {provider: scripted, model: scripted}",
      "language: ko",
      "mcp:",
      "  - name: repo-docs",
      "    transport: local",
      `    path: ${root}`,
      "    include: [docs]",
      "",
    ].join("\n"),
  );
  return { root, config };
}

async function runSetup(
  config: string,
  extra: readonly string[],
): Promise<{ code: number; printed: string; model: ScriptedChatModel }> {
  const model = new ScriptedChatModel();
  const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
  const code = await runOntologyCli(["setup", "--config", config, ...extra], {
    loadAgent: (path) => new KontextLoader({ llmRegistry: scriptedRegistry(model) }).fromFile(path),
  });
  const printed = out.mock.calls.map((call) => String(call[0])).join("");
  out.mockRestore();
  return { code, printed, model };
}

describe("kontext-ontology setup", () => {
  it("builds an ontology from local Markdown and saves it into the config", async () => {
    const { config } = makeWorkspace();
    const { code, printed, model } = await runSetup(config, ["--write"]);
    expect(code, printed).toBe(0);

    // The connector really walked the repository: the model was shown both titles.
    const prompts = model.seen.join("\n");
    expect(prompts).toContain("Retry policy");
    expect(prompts).toContain("Session authentication");

    const saved = parse(readFileSync(config, "utf8"));
    expect(saved.ontology.map((node: { id: string }) => node.id).sort()).toEqual([
      "Authentication",
      "Reliability",
    ]);
    expect(
      saved.ontology.find((n: { id: string }) => n.id === "Reliability").description,
    ).toContain("retries");
    // Everything the user already had survives the write.
    expect(saved.language).toBe("ko");
    expect(saved.mcp).toHaveLength(1);
    expect(saved.llm.traversal.provider).toBe("scripted");
    expect(printed).toContain("Saved 2 ontology node(s).");
  });

  it("reports the result but writes nothing without --write", async () => {
    const { config } = makeWorkspace();
    const before = readFileSync(config, "utf8");
    const { code, printed } = await runSetup(config, []);
    expect(code, printed).toBe(0);
    expect(readFileSync(config, "utf8")).toBe(before);
    expect(printed).toContain("Re-run with --write");
  });
});
