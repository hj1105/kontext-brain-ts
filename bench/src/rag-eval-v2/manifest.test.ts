import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_RAG_EVAL_MANIFEST,
  assertValidManifest,
  loadFrozenRunManifest,
  manifestDigest,
  manifestForRunDirectory,
} from "./manifest.js";
import { freezeRunManifest } from "./pipeline.js";

const directories: string[] = [];

afterEach(() => {
  for (const directory of directories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("rag eval manifest", () => {
  it("locks the agreed common models and no aggregate score policy", () => {
    expect(() => assertValidManifest(DEFAULT_RAG_EVAL_MANIFEST)).not.toThrow();
    expect(DEFAULT_RAG_EVAL_MANIFEST.models.embedding).toMatchObject({
      provider: "openai",
      model: "text-embedding-3-small",
      dimensions: 1536,
    });
    expect(DEFAULT_RAG_EVAL_MANIFEST.models.answer).toMatchObject({
      model: "gpt-5.6-terra",
      reasoningEffort: "medium",
    });
    expect(DEFAULT_RAG_EVAL_MANIFEST.models.judge).toMatchObject({
      model: "gpt-5.6-sol",
      reasoningEffort: "xhigh",
    });
    expect(DEFAULT_RAG_EVAL_MANIFEST.benchmarkPolicy.aggregateAcrossDatasets).toBe(false);
    expect(DEFAULT_RAG_EVAL_MANIFEST.benchmarkPolicy).toMatchObject({
      retrievalQueryScope: "all",
      answerJudgeSamplePerDataset: 200,
      answerJudgeSampleSeed: 20260814,
      answerCodexBatchSize: 1,
      judgeCodexBatchSize: 1,
      codexConcurrency: 1,
    });
    expect(DEFAULT_RAG_EVAL_MANIFEST.benchmarkPolicy.humanAuditPerDataset).toBe(100);
    expect(
      DEFAULT_RAG_EVAL_MANIFEST.frameworks.filter(
        (framework) => framework.versionPolicy === "official-pinned",
      ),
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "microsoft-graphrag", pinnedVersion: "3.1.1" }),
        expect.objectContaining({ id: "lightrag", pinnedVersion: "1.5.6" }),
        expect.objectContaining({ id: "hipporag2", pinnedVersion: "2.0.0a4" }),
      ]),
    );
  });

  it("produces a stable digest", () => {
    expect(manifestDigest(DEFAULT_RAG_EVAL_MANIFEST)).toBe(
      manifestDigest(structuredClone(DEFAULT_RAG_EVAL_MANIFEST)),
    );
  });

  it("loads the canonical manifest from a frozen run envelope", () => {
    const directory = temporaryDirectory();
    freezeRunManifest(DEFAULT_RAG_EVAL_MANIFEST, directory);

    expect(loadFrozenRunManifest(join(directory, "run-manifest.json"))).toEqual(
      DEFAULT_RAG_EVAL_MANIFEST,
    );
  });

  it("resumes with the frozen manifest when later defaults change", () => {
    const directory = temporaryDirectory();
    freezeRunManifest(DEFAULT_RAG_EVAL_MANIFEST, directory);
    const laterDefault = {
      ...DEFAULT_RAG_EVAL_MANIFEST,
      benchmarkPolicy: {
        ...DEFAULT_RAG_EVAL_MANIFEST.benchmarkPolicy,
        humanAuditPerDataset: 101,
      },
    };

    expect(manifestForRunDirectory(laterDefault, directory)).toEqual(DEFAULT_RAG_EVAL_MANIFEST);
    expect(manifestForRunDirectory(laterDefault, temporaryDirectory())).toEqual(laterDefault);
  });

  it("rejects malformed frozen run manifest envelopes", () => {
    const directory = temporaryDirectory();
    const path = join(directory, "run-manifest.json");
    writeFileSync(path, JSON.stringify(DEFAULT_RAG_EVAL_MANIFEST));

    expect(() => loadFrozenRunManifest(path)).toThrow(/envelope/i);
  });

  it("rejects a frozen run manifest whose canonical digest does not match", () => {
    const directory = temporaryDirectory();
    const path = join(directory, "run-manifest.json");
    writeFileSync(
      path,
      JSON.stringify({
        manifestDigest: "0".repeat(64),
        manifest: DEFAULT_RAG_EVAL_MANIFEST,
      }),
    );

    expect(() => loadFrozenRunManifest(path)).toThrow(/digest mismatch/i);
  });
});

function temporaryDirectory(): string {
  const directory = mkdtempSync(join(tmpdir(), "rag-eval-manifest-loader-"));
  directories.push(directory);
  return directory;
}
