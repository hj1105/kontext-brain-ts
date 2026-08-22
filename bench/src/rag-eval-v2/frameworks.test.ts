import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { CommandRunner } from "./codex-json.js";
import type { DatasetBundle } from "./contracts.js";
import {
  ExternalCommandFrameworkAdapter,
  VectorRagRerankerAdapter,
  createFrameworkAdapters,
} from "./frameworks.js";
import { writeJsonLines } from "./jsonl.js";
import { DEFAULT_RAG_EVAL_MANIFEST } from "./manifest.js";
import type { EmbeddingClient } from "./openai-embeddings.js";

const originalCommand = process.env.RAG_EVAL_GRAPHRAG_COMMAND;
const originalOpenAiApiKey = process.env.OPENAI_API_KEY;
const originalKontextMode = process.env.KONTEXT_RAG_EVAL_MODE;
const originalPrecomputedIndex = process.env.KONTEXT_RAG_EVAL_PRECOMPUTED_INDEX;
const temporaryDirectories: string[] = [];

afterEach(() => {
  if (originalCommand === undefined) delete process.env.RAG_EVAL_GRAPHRAG_COMMAND;
  else process.env.RAG_EVAL_GRAPHRAG_COMMAND = originalCommand;
  restoreEnvironment("OPENAI_API_KEY", originalOpenAiApiKey);
  restoreEnvironment("KONTEXT_RAG_EVAL_MODE", originalKontextMode);
  restoreEnvironment("KONTEXT_RAG_EVAL_PRECOMPUTED_INDEX", originalPrecomputedIndex);
  for (const directory of temporaryDirectories.splice(0))
    rmSync(directory, { recursive: true, force: true });
});

describe("kontext cache-only coverage modes", () => {
  it("selects the promoted v13 profile when no mode override is configured", async () => {
    restoreEnvironment("OPENAI_API_KEY", undefined);
    restoreEnvironment("KONTEXT_RAG_EVAL_MODE", undefined);

    const adapter = createFrameworkAdapters(DEFAULT_RAG_EVAL_MANIFEST).find(
      (candidate) => candidate.id === "kontext-brain",
    );

    await expect(adapter?.doctor()).resolves.toMatchObject({
      status: "blocked",
      version: "workspace-0.1.0+v13-anchored-evidence-answer-stack",
      detail: "OPENAI_API_KEY is required for bidirectional KG chunk seeds",
    });
  });

  it("rejects an unknown mode instead of silently falling back to legacy retrieval", () => {
    process.env.KONTEXT_RAG_EVAL_MODE = "typo-mode";

    expect(() => createFrameworkAdapters(DEFAULT_RAG_EVAL_MANIFEST)).toThrow(
      "Unknown KONTEXT_RAG_EVAL_MODE: typo-mode",
    );
  });

  it.each([
    [
      "v14a-anchored-deterministic-soft-coverage-stack",
      "workspace-0.1.0+v14a-anchored-deterministic-soft-coverage-stack",
    ],
    [
      "v14b-anchored-deterministic-quota-coverage-stack",
      "workspace-0.1.0+v14b-anchored-deterministic-quota-coverage-stack",
    ],
  ])("wires %s and its precomputed index through the framework factory", async (mode, version) => {
    const precomputedIndexDirectory = mkdtempSync(join(tmpdir(), "rag-eval-v14-cache-"));
    temporaryDirectories.push(precomputedIndexDirectory);
    restoreEnvironment("OPENAI_API_KEY", undefined);
    process.env.KONTEXT_RAG_EVAL_MODE = mode;
    process.env.KONTEXT_RAG_EVAL_PRECOMPUTED_INDEX = precomputedIndexDirectory;

    const adapter = createFrameworkAdapters(DEFAULT_RAG_EVAL_MANIFEST).find(
      (candidate) => candidate.id === "kontext-brain",
    );

    await expect(adapter?.doctor()).resolves.toMatchObject({ status: "ready", version });
  });
});

describe("vector embedding checkpoints", () => {
  it("resumes after a quota failure without embedding completed batches again", async () => {
    const workDirectory = mkdtempSync(join(tmpdir(), "rag-eval-v2-vector-checkpoint-"));
    temporaryDirectories.push(workDirectory);
    const bundle: DatasetBundle = {
      id: "graphrag-bench-medical",
      track: "static-kb",
      documents: [
        {
          id: "document-1",
          sourceId: "source-1",
          title: "Long document",
          text: "word ".repeat(23_000),
          metadata: {},
        },
      ],
      queries: [
        {
          id: "query-1",
          text: "word",
          referenceAnswer: "word",
          goldEvidenceIds: ["source-1"],
          goldEvidenceText: ["word"],
          answerable: true,
          category: "test",
          metadata: {},
        },
      ],
      provenance: { source: "test", version: "1", license: "test" },
    };
    const firstClient = new FakeEmbeddingClient(2);
    const firstAdapter = new VectorRagRerankerAdapter(
      firstClient as unknown as EmbeddingClient,
      DEFAULT_RAG_EVAL_MANIFEST,
    );

    await expect(
      firstAdapter.retrieve(bundle, { workDirectory, topK: 1, candidateK: 1 }),
    ).rejects.toThrow("simulated quota failure");
    expect(firstClient.documentBatchIds[0]).toHaveLength(100);

    const resumedClient = new FakeEmbeddingClient(null);
    const resumedAdapter = new VectorRagRerankerAdapter(
      resumedClient as unknown as EmbeddingClient,
      DEFAULT_RAG_EVAL_MANIFEST,
    );
    const results = await resumedAdapter.retrieve(bundle, {
      workDirectory,
      topK: 1,
      candidateK: 1,
    });

    expect(results[0]?.status).toBe("ok");
    expect(resumedClient.documentBatchIds[0]?.[0]).toContain("vector-chunk-000100");
    expect(resumedClient.documentBatchIds.flat()).not.toContain("document-1::vector-chunk-000000");
  });
});

class FakeEmbeddingClient {
  readonly model = "text-embedding-3-small";
  readonly dimensions = 1536;
  readonly documentBatchIds: string[][] = [];
  private calls = 0;
  private inputTokens = 0;

  constructor(private readonly failOnCall: number | null) {}

  async embed(
    inputs: readonly { readonly id: string }[],
    task: "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY",
  ): Promise<Array<{ id: string; values: number[] }>> {
    this.calls += 1;
    if (task === "RETRIEVAL_DOCUMENT") this.documentBatchIds.push(inputs.map((input) => input.id));
    if (this.failOnCall === this.calls) throw new Error("simulated quota failure");
    this.inputTokens += inputs.length;
    return inputs.map((input) => ({
      id: input.id,
      values: Array.from({ length: 1536 }, () => 0.01),
    }));
  }

  getUsage(): { requests: number; inputTokens: number; totalTokens: number } {
    return { requests: this.calls, inputTokens: this.inputTokens, totalTokens: this.inputTokens };
  }
}

function restoreEnvironment(name: string, value: string | undefined): void {
  // biome-ignore lint/performance/noDelete: tests must restore an originally absent variable.
  if (value === undefined) delete process.env[name];
  else process.env[name] = value;
}

describe("external framework doctor", () => {
  it("reports a doctor command failure as a blocked framework", async () => {
    process.env.RAG_EVAL_GRAPHRAG_COMMAND = JSON.stringify(["graphrag-adapter"]);
    const framework = DEFAULT_RAG_EVAL_MANIFEST.frameworks.find(
      (candidate) => candidate.id === "microsoft-graphrag",
    )!;
    const runner: CommandRunner = async () => {
      throw new Error("doctor timed out");
    };
    const doctor = await new ExternalCommandFrameworkAdapter(
      framework,
      DEFAULT_RAG_EVAL_MANIFEST,
      runner,
    ).doctor();
    expect(doctor).toMatchObject({
      status: "blocked",
      version: "unresolved",
      detail: "doctor failed: doctor timed out",
    });
  });

  it("requires the exact official pinned version", async () => {
    process.env.RAG_EVAL_GRAPHRAG_COMMAND = JSON.stringify(["graphrag-adapter"]);
    const framework = DEFAULT_RAG_EVAL_MANIFEST.frameworks.find(
      (candidate) => candidate.id === "microsoft-graphrag",
    )!;
    const runner: CommandRunner = async () => ({
      exitCode: 0,
      stdout: JSON.stringify({ status: "ready", version: "3.0.0", detail: "installed" }),
      stderr: "",
      durationMs: 1,
    });
    const doctor = await new ExternalCommandFrameworkAdapter(
      framework,
      DEFAULT_RAG_EVAL_MANIFEST,
      runner,
    ).doctor();
    expect(doctor).toMatchObject({
      status: "blocked",
      version: "3.0.0",
      detail: "Expected pinned version 3.1.1, found 3.0.0",
    });
  });
});

describe("external framework retrieval", () => {
  it("accepts one result for every identical duplicate query row", async () => {
    process.env.RAG_EVAL_GRAPHRAG_COMMAND = JSON.stringify(["graphrag-adapter"]);
    const framework = DEFAULT_RAG_EVAL_MANIFEST.frameworks.find(
      (candidate) => candidate.id === "microsoft-graphrag",
    );
    if (!framework) throw new Error("Missing Microsoft GraphRAG manifest entry");
    const workDirectory = mkdtempSync(join(tmpdir(), "rag-eval-external-duplicate-query-"));
    temporaryDirectories.push(workDirectory);
    const query = {
      id: "duplicate-query",
      text: "What happened?",
      referenceAnswer: "An event happened.",
      goldEvidenceIds: ["source-1"],
      goldEvidenceText: ["An event happened."],
      answerable: true,
      category: "test",
      metadata: {},
    } as const;
    const bundle: DatasetBundle = {
      id: "graphrag-bench-novel",
      track: "static-kb",
      documents: [
        {
          id: "document-1",
          sourceId: "source-1",
          title: "Event",
          text: "An event happened.",
          metadata: {},
        },
      ],
      queries: [query, { ...query }],
      provenance: { source: "test", version: "1", license: "test" },
    };
    const runner: CommandRunner = async (_command, args) => {
      const outputIndex = args.indexOf("--output");
      if (outputIndex >= 0) {
        const outputPath = args[outputIndex + 1];
        if (!outputPath) throw new Error("Missing external adapter output path");
        const record = {
          datasetId: bundle.id,
          frameworkId: framework.id,
          queryId: query.id,
          status: "ok" as const,
          evidence: [
            {
              id: "native-context",
              sourceId: "graphrag-local-search-context",
              text: "0|title: One\nsource_id: source-1\n\n1|title: Two\nsource_id: source-2",
              score: 1,
              rank: 1,
              metadata: { nativeContext: true },
            },
          ],
          latencyMs: 1,
          inputTokens: null,
          error: null,
          frameworkVersion: framework.pinnedVersion ?? "unresolved",
          configDigest: "adapter-digest",
        };
        writeJsonLines(outputPath, [record, { ...record }]);
      }
      return { exitCode: 0, stdout: "", stderr: "", durationMs: 1 };
    };

    const results = await new ExternalCommandFrameworkAdapter(
      framework,
      DEFAULT_RAG_EVAL_MANIFEST,
      runner,
    ).retrieve(bundle, { workDirectory, topK: 10, candidateK: 50 });

    expect(results).toHaveLength(2);
    expect(results.map((result) => result.queryId)).toEqual(["duplicate-query", "duplicate-query"]);
    expect(results[0]?.evidence[0]?.sourceIds).toEqual(["source-1", "source-2"]);
  });
});
