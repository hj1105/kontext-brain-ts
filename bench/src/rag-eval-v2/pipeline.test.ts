import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { DEFAULT_RAG_EVAL_MANIFEST } from "./manifest.js";
import { freezeEvaluationSample, freezeRunManifest } from "./pipeline.js";

const directories: string[] = [];

afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

describe("run manifest freeze", () => {
  it("rejects checkpoint reuse after a benchmark configuration change", () => {
    const directory = mkdtempSync(join(tmpdir(), "rag-eval-manifest-"));
    directories.push(directory);
    freezeRunManifest(DEFAULT_RAG_EVAL_MANIFEST, directory);
    const changed = {
      ...DEFAULT_RAG_EVAL_MANIFEST,
      benchmarkPolicy: {
        ...DEFAULT_RAG_EVAL_MANIFEST.benchmarkPolicy,
        humanAuditPerDataset: 99,
      },
    };
    expect(() => freezeRunManifest(changed, directory)).toThrow(/manifest mismatch/);
  });

  it("rejects reuse when the frozen evaluation sample changes", () => {
    const directory = mkdtempSync(join(tmpdir(), "rag-eval-sample-"));
    directories.push(directory);
    const sample = {
      schemaVersion: "1.0.0" as const,
      datasetId: "graphrag-bench-medical" as const,
      method: "deterministic-proportional-category-stratified" as const,
      seed: 20260814,
      requested: 200,
      population: 10,
      selected: 1,
      categories: [{ category: "Fact", population: 10, selected: 1 }],
      queryIds: ["q1"],
      sampleDigest: "first",
    };
    freezeEvaluationSample(sample, directory);
    expect(() => freezeEvaluationSample({ ...sample, sampleDigest: "second" }, directory))
      .toThrow(/sample mismatch/i);
  });
});
