import { describe, expect, it } from "vitest";
import {
  assertValidManifest,
  DEFAULT_RAG_EVAL_MANIFEST,
  manifestDigest,
} from "./manifest.js";

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
    expect(DEFAULT_RAG_EVAL_MANIFEST.frameworks.filter((framework) => framework.versionPolicy === "official-pinned"))
      .toEqual(expect.arrayContaining([
        expect.objectContaining({ id: "microsoft-graphrag", pinnedVersion: "3.1.1" }),
        expect.objectContaining({ id: "lightrag", pinnedVersion: "1.5.6" }),
        expect.objectContaining({ id: "hipporag2", pinnedVersion: "2.0.0a4" }),
      ]));
  });

  it("produces a stable digest", () => {
    expect(manifestDigest(DEFAULT_RAG_EVAL_MANIFEST)).toBe(
      manifestDigest(structuredClone(DEFAULT_RAG_EVAL_MANIFEST)),
    );
  });
});
