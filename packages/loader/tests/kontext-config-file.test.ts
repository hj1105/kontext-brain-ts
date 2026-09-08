import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { parse } from "yaml";
import {
  readConfigDocument,
  readMCPEntries,
  readOntologyNodeIds,
  toOntologyYamlNodes,
  withMCPEntries,
  withOntology,
  writeConfigDocument,
} from "../src/kontext-config-file.js";

const roots: string[] = [];

function makeFile(contents: string): string {
  const root = mkdtempSync(join(tmpdir(), "kontext-config-"));
  roots.push(root);
  const path = join(root, "kontext.yaml");
  writeFileSync(path, contents);
  return path;
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

const BASE = `llm:
  traversal:
    provider: openai
    model: gpt-4o-mini
  reasoning:
    provider: openai
    model: gpt-4o
language: ko
mcp:
  - name: notion
    command: notion-mcp
`;

describe("kontext.yaml editing", () => {
  it("keeps every key it does not manage when writing", () => {
    const path = makeFile(BASE);
    const document = readConfigDocument(path);
    writeConfigDocument(withOntology(document, [{ id: "Auth", description: "x", weight: 1 }]));
    const saved = parse(readFileSync(path, "utf8"));
    expect(saved.language).toBe("ko");
    expect(saved.llm.reasoning.model).toBe("gpt-4o");
    expect(saved.mcp).toHaveLength(1);
    expect(saved.ontology).toHaveLength(1);
  });

  it("treats a missing file as an empty document rather than failing", () => {
    const document = readConfigDocument(join(tmpdir(), "kontext-absent-file.yaml"));
    expect(readMCPEntries(document)).toEqual([]);
    expect(readOntologyNodeIds(document)).toEqual([]);
  });

  it("rejects a file whose top level is not a mapping", () => {
    expect(() => readConfigDocument(makeFile("- one\n- two\n"))).toThrow(/YAML mapping/);
  });

  it("round-trips a description containing YAML punctuation", () => {
    // A generated description routinely contains ':' and quotes. Concatenating it
    // into YAML produced a file that no longer parsed, which is why the document
    // is serialized rather than assembled from strings.
    const path = makeFile(BASE);
    const hostile = 'Retry: "twice", then #fail — see docs/a.md';
    writeConfigDocument(
      withOntology(readConfigDocument(path), [{ id: "Retry", description: hostile, weight: 1 }]),
    );
    const saved = parse(readFileSync(path, "utf8"));
    expect(saved.ontology[0].description).toBe(hostile);
  });

  it("reads back the MCP entries it wrote", () => {
    const path = makeFile(BASE);
    const document = readConfigDocument(path);
    writeConfigDocument(
      withMCPEntries(document, [
        ...readMCPEntries(document),
        { name: "repo-docs", transport: "local", path: "/repo" },
      ]),
    );
    expect(readMCPEntries(readConfigDocument(path)).map((e) => e.name)).toEqual([
      "notion",
      "repo-docs",
    ]);
  });
});

describe("comment preservation", () => {
  it("keeps the comments a hand-maintained config carries", () => {
    const path = makeFile(
      [
        "# Which model plans the ontology.",
        "llm:",
        "  traversal: {provider: openai, model: gpt-4o-mini}",
        "  reasoning: {provider: openai, model: gpt-4o}",
        "",
        "# Decisions live in Notion; docs/ is the fallback.",
        "mcp:",
        "  - name: notion # shared workspace",
        "    url: https://mcp.notion.invalid",
        "",
      ].join("\n"),
    );
    writeConfigDocument(
      withOntology(readConfigDocument(path), [{ id: "A", description: "x", weight: 1 }]),
    );
    const saved = readFileSync(path, "utf8");
    expect(saved).toContain("# Which model plans the ontology.");
    expect(saved).toContain("# Decisions live in Notion; docs/ is the fallback.");
    expect(saved).toContain("# shared workspace");
    expect(parse(saved).ontology).toHaveLength(1);
    expect(parse(saved).mcp[0].name).toBe("notion");
  });

  it("removes a key the caller dropped rather than leaving a stale one", () => {
    const path = makeFile("llm:\n  traversal: {provider: none, model: none}\nretired: true\n");
    const document = readConfigDocument(path);
    const { retired: _dropped, ...rest } = document.data;
    writeConfigDocument({ ...document, data: rest });
    expect(parse(readFileSync(path, "utf8")).retired).toBeUndefined();
  });
});

describe("toOntologyYamlNodes field preservation", () => {
  it("writes back the hand-authored fields the graph still carries", () => {
    // A rebuild replaces the whole ontology array. Emitting only id/description/weight
    // silently reset nodeType, webSearch, keywords and mcpSource on the next load.
    const [node] = toOntologyYamlNodes(
      [
        {
          id: "billing",
          description: "invoices",
          weight: 1,
          level: 0,
          nodeType: "ENTITY",
          webSearch: true,
          keywords: ["invoice", "dunning"],
          mcpSource: "notion",
        },
      ] as never,
      [] as never,
    );
    expect(node).toMatchObject({
      nodeType: "ENTITY",
      webSearch: true,
      keywords: ["invoice", "dunning"],
      mcpSource: "notion",
    });
  });

  it("omits the defaults so a plain node stays plain", () => {
    const [node] = toOntologyYamlNodes(
      [
        {
          id: "plain",
          description: "x",
          weight: 1,
          level: 0,
          nodeType: "DOMAIN",
          webSearch: false,
          keywords: [],
          mcpSource: null,
        },
      ] as never,
      [] as never,
    );
    expect(node).toEqual({ id: "plain", description: "x", weight: 1 });
  });
});

describe("toOntologyYamlNodes", () => {
  it("attaches each node's edges, heaviest first, and omits empty fields", () => {
    const nodes = toOntologyYamlNodes(
      [
        { id: "Auth", description: "sessions", weight: 1, parentId: null, level: 0 },
        { id: "Api", description: "endpoints", weight: 0.5, parentId: "Auth", level: 1 },
      ] as never,
      [
        { from: "Auth", to: "Api", weight: 0.2 },
        { from: "Auth", to: "Api", weight: 0.9, type: "uses" },
      ] as never,
    );
    expect(nodes[0]?.relates?.map((r) => r.weight)).toEqual([0.9, 0.2]);
    expect(nodes[0]?.relates?.[0]?.type).toBe("uses");
    expect(nodes[0]).not.toHaveProperty("parentId");
    expect(nodes[0]).not.toHaveProperty("level");
    expect(nodes[1]).toMatchObject({ parentId: "Auth", level: 1 });
    expect(nodes[1]).not.toHaveProperty("relates");
  });
});
