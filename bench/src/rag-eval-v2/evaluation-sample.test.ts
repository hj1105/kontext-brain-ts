import { describe, expect, it } from "vitest";
import type { BenchmarkQuery, DatasetBundle } from "./contracts.js";
import { createEvaluationSample } from "./evaluation-sample.js";

function query(id: string, category: string): BenchmarkQuery {
  return {
    id,
    text: id,
    referenceAnswer: null,
    goldEvidenceIds: [],
    goldEvidenceText: [],
    answerable: true,
    category,
    metadata: {},
  };
}

const bundle: DatasetBundle = {
  id: "graphrag-bench-medical",
  track: "static-kb",
  documents: [],
  queries: [
    ...Array.from({ length: 18 }, (_, index) => query(`common-${index}`, "common")),
    query("rare-1", "rare"),
    query("rare-2", "rare"),
  ],
  provenance: { source: "test", version: "1", license: "test" },
};

describe("evaluation sample", () => {
  it("is deterministic, proportional, and keeps represented minority categories", () => {
    const first = createEvaluationSample(bundle, 10, 20260814);
    const second = createEvaluationSample(bundle, 10, 20260814);

    expect(first.manifest).toEqual(second.manifest);
    expect(first.queries).toHaveLength(10);
    expect(first.manifest.categories).toEqual([
      { category: "common", population: 18, selected: 9 },
      { category: "rare", population: 2, selected: 1 },
    ]);
    expect(first.manifest.sampleDigest).toMatch(/^[a-f0-9]{64}$/);
  });

  it("uses the full population when it is smaller than the requested sample", () => {
    const sample = createEvaluationSample(bundle, 200, 20260814);
    expect(sample.queries).toEqual(bundle.queries);
    expect(sample.manifest.selected).toBe(20);
  });
});
