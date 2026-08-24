import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { validateCleanLatencySuiteConfig } from "./clean-latency-suite.js";
import {
  CLEAN_LATENCY_ANTHROPIC_ANSWER_MODEL,
  CLEAN_LATENCY_TAIL_LIMIT_MS,
  assessCleanLatency,
  cleanLatencyManifest,
  frozenConditions,
  summarizeCleanLatency,
  validateIndexSourceProvenance,
} from "./clean-latency.js";
import type { AnswerResult, JudgeResult, RetrievalResult } from "./contracts.js";

const queryIds = ["q1", "q2"];

function retrieval(queryId: string, latencyMs: number, completedAt?: string): RetrievalResult {
  return {
    datasetId: "graphrag-bench-medical",
    frameworkId: "kontext-brain",
    queryId,
    status: "ok",
    evidence: [],
    latencyMs,
    inputTokens: null,
    error: null,
    frameworkVersion: "test",
    configDigest: "test",
    completedAt,
  };
}

function answer(queryId: string, latencyMs: number, completedAt?: string): AnswerResult {
  return {
    datasetId: "graphrag-bench-medical",
    frameworkId: "kontext-brain",
    queryId,
    status: "ok",
    output: { answer: "", citations: [], abstained: true, abstentionReason: "test" },
    latencyMs,
    inputTokens: null,
    outputTokens: null,
    error: null,
    inputDigest: "test",
    completedAt,
  };
}

function judgement(queryId: string, latencyMs: number, completedAt?: string): JudgeResult {
  return {
    datasetId: "graphrag-bench-medical",
    frameworkId: "kontext-brain",
    queryId,
    status: "ok",
    output: null,
    latencyMs,
    inputTokens: null,
    outputTokens: null,
    error: null,
    inputDigest: "test",
    completedAt,
  };
}

describe("clean latency protocol", () => {
  it("reports retrieval, query-to-answer, and judge-inclusive E2E separately", () => {
    const summary = summarizeCleanLatency(
      queryIds,
      [retrieval("q1", 10), retrieval("q2", 20)],
      [answer("q1", 30), answer("q2", 40)],
      [judgement("q1", 50), judgement("q2", 60)],
    );

    expect(summary.retrieval.p95Ms).toBe(20);
    expect(summary.queryToAnswer.p95Ms).toBe(60);
    expect(summary.judgeInclusiveEvaluationEndToEnd.p95Ms).toBe(120);
  });

  it("invalidates 600-second tails, throttle waves, and index mutations", () => {
    const assessment = assessCleanLatency(
      queryIds,
      [
        retrieval("q1", 10, "2026-08-24T00:00:00.000Z"),
        retrieval("q2", CLEAN_LATENCY_TAIL_LIMIT_MS + 1, "2026-08-24T00:15:00.000Z"),
      ],
      [answer("q1", 30), answer("q2", 40)],
      [judgement("q1", 50), judgement("q2", 60)],
      false,
    );

    expect(assessment.status).toBe("invalid");
    expect(assessment.reasons).toEqual(
      expect.arrayContaining([
        "retrieval has 1 latency values over 600 seconds",
        "read-only index source metadata changed during the run",
      ]),
    );
    expect(assessment.suspiciousCompletionWaveGaps).toHaveLength(1);
  });

  it("requires the exact eight-row Medical/Novel matrix", () => {
    const systems = [
      "kontext-v15",
      "kontext-v13",
      "lightrag-1.5.6",
      "microsoft-graphrag-3.1.1",
    ] as const;
    const rows = (["graphrag-bench-medical", "graphrag-bench-novel"] as const).flatMap(
      (datasetId) =>
        systems.map((system, index) => ({
          datasetId,
          system,
          indexSourceDirectory: `/index/${datasetId}/${system}`,
          outputDirectoryName: `${datasetId}-${index}`,
        })),
    );
    expect(() =>
      validateCleanLatencySuiteConfig({ schemaVersion: "1.0.0", outputRoot: "output", rows }),
    ).not.toThrow();
    expect(() =>
      validateCleanLatencySuiteConfig({
        schemaVersion: "1.0.0",
        outputRoot: "output",
        rows: rows.slice(1),
      }),
    ).toThrow("each Medical/Novel x four-system row exactly once");
  });

  it("swaps only the answer model for the anthropic-api backend", () => {
    const codexManifest = cleanLatencyManifest();
    const anthropicManifest = cleanLatencyManifest("anthropic-api");

    expect(codexManifest).toEqual(cleanLatencyManifest("codex-exec"));
    expect(anthropicManifest.models.answer).toEqual({
      provider: "anthropic",
      model: CLEAN_LATENCY_ANTHROPIC_ANSWER_MODEL,
      reasoningEffort: "medium",
      execution: "anthropic-api",
    });
    expect(anthropicManifest.models.judge).toEqual(codexManifest.models.judge);
    expect(anthropicManifest.models.embedding).toEqual(codexManifest.models.embedding);
    expect(anthropicManifest.benchmarkPolicy).toEqual(codexManifest.benchmarkPolicy);
  });

  it("records the completion backend and models in the frozen conditions", () => {
    const codexConditions = frozenConditions("codex-exec", cleanLatencyManifest("codex-exec"));
    expect(codexConditions.completionBackend).toBe("codex-exec");
    expect(codexConditions.answerModel).toBe("gpt-5.6-terra");
    expect(codexConditions.judgeModel).toBe("gpt-5.6-sol");
    expect(codexConditions.answerModelMatchesIndexBuild).toBe(true);

    const anthropicConditions = frozenConditions(
      "anthropic-api",
      cleanLatencyManifest("anthropic-api"),
    );
    expect(anthropicConditions.completionBackend).toBe("anthropic-api");
    expect(anthropicConditions.answerModel).toBe(CLEAN_LATENCY_ANTHROPIC_ANSWER_MODEL);
    expect(anthropicConditions.judgeModel).toBe("gpt-5.6-sol");
    expect(anthropicConditions.answerModelMatchesIndexBuild).toBe(false);
    expect(anthropicConditions.maxRetries).toBe(codexConditions.maxRetries);
  });

  it("rejects unknown completion backends in the suite config", () => {
    expect(() =>
      validateCleanLatencySuiteConfig({
        schemaVersion: "1.0.0",
        outputRoot: "output",
        completionBackend: "gemini" as never,
        rows: [],
      }),
    ).toThrow("Unsupported completionBackend gemini");
  });

  it("fails closed unless the warm source has matching model provenance", () => {
    const root = mkdtempSync(join(tmpdir(), "clean-latency-source-"));
    try {
      const source = join(root, "dataset", "framework", "index");
      mkdirSync(source, { recursive: true });
      const manifest = cleanLatencyManifest();
      writeFileSync(
        join(root, "run-manifest.json"),
        JSON.stringify({ manifest: { ...manifest, models: manifest.models } }),
      );
      expect(validateIndexSourceProvenance(source, manifest).path).toBe(
        realpathSync(join(root, "run-manifest.json")),
      );
      writeFileSync(
        join(root, "run-manifest.json"),
        JSON.stringify({
          manifest: {
            ...manifest,
            models: { ...manifest.models, answer: { ...manifest.models.answer, model: "wrong" } },
          },
        }),
      );
      expect(() => validateIndexSourceProvenance(source, manifest)).toThrow(
        "model provenance mismatch",
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("allows an intentional answer-model difference only when the caller opts out", () => {
    const root = mkdtempSync(join(tmpdir(), "clean-latency-anthropic-source-"));
    try {
      const source = join(root, "dataset", "framework", "index");
      mkdirSync(source, { recursive: true });
      const indexBuildManifest = cleanLatencyManifest();
      writeFileSync(
        join(root, "run-manifest.json"),
        JSON.stringify({ manifest: { models: indexBuildManifest.models } }),
      );
      const anthropicManifest = cleanLatencyManifest("anthropic-api");
      expect(() => validateIndexSourceProvenance(source, anthropicManifest)).toThrow(
        "model provenance mismatch",
      );
      const validated = validateIndexSourceProvenance(source, anthropicManifest, {
        requireAnswerModelMatch: false,
      });
      expect(validated.indexBuildModels.answer).toMatchObject({ model: "gpt-5.6-terra" });
      expect(validated.indexBuildModels.embedding).toMatchObject({
        model: "text-embedding-3-small",
      });
      writeFileSync(
        join(root, "run-manifest.json"),
        JSON.stringify({
          manifest: {
            models: {
              ...indexBuildManifest.models,
              embedding: { ...indexBuildManifest.models.embedding, model: "wrong" },
            },
          },
        }),
      );
      expect(() =>
        validateIndexSourceProvenance(source, anthropicManifest, {
          requireAnswerModelMatch: false,
        }),
      ).toThrow("model provenance mismatch");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
