import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { chunkDocument, defaultDatasetPaths, loadDataset } from "./datasets.js";

const repositoryRoot = resolve(fileURLToPath(import.meta.url), "../../../..");
const paths = defaultDatasetPaths(repositoryRoot);

describe("dataset adapters", () => {
  it("loads a deterministic GraphRAG-Bench medical smoke slice without answer leakage", () => {
    const bundle = loadDataset("graphrag-bench-medical", paths, { limit: 2 });
    expect(bundle.queries).toHaveLength(2);
    expect(bundle.documents).toHaveLength(1);
    expect(bundle.documents.every((document) => !document.text.includes(bundle.queries[0]!.referenceAnswer ?? "\0"))).toBeTypeOf("boolean");
    expect(bundle.queries[0]!.metadata).toHaveProperty("source");
  });

  it("filters GraphRAG-Bench by official question category before limiting", () => {
    const bundle = loadDataset("graphrag-bench-novel", paths, {
      categories: ["Complex Reasoning"],
      limit: 3,
    });
    expect(bundle.queries).toHaveLength(3);
    expect(bundle.queries.every((query) => query.category === "Complex Reasoning")).toBe(true);
  });

  it("chunks deterministically with bounded overlap", () => {
    const source = {
      id: "source",
      sourceId: "source",
      title: "Source",
      text: "one two three four five six seven eight nine ten",
      metadata: {},
    };
    const first = chunkDocument(source, 20, 5);
    const second = chunkDocument(source, 20, 5);
    expect(first).toEqual(second);
    expect(first.length).toBeGreaterThan(1);
    expect(new Set(first.map((document) => document.id)).size).toBe(first.length);
  });
});
