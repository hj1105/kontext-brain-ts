import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { CommandRunner } from "./codex-json.js";
import type { DatasetBundle } from "./contracts.js";
import type { EmbeddingClient } from "./openai-embeddings.js";
import { ExternalCommandFrameworkAdapter, VectorRagRerankerAdapter } from "./frameworks.js";
import { DEFAULT_RAG_EVAL_MANIFEST } from "./manifest.js";

const originalCommand = process.env.RAG_EVAL_GRAPHRAG_COMMAND;
const temporaryDirectories: string[] = [];

afterEach(() => {
  if (originalCommand === undefined) delete process.env.RAG_EVAL_GRAPHRAG_COMMAND;
  else process.env.RAG_EVAL_GRAPHRAG_COMMAND = originalCommand;
  for (const directory of temporaryDirectories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

describe("vector embedding checkpoints", () => {
  it("resumes after a quota failure without embedding completed batches again", async () => {
    const workDirectory = mkdtempSync(join(tmpdir(), "rag-eval-v2-vector-checkpoint-"));
    temporaryDirectories.push(workDirectory);
    const bundle: DatasetBundle = {
      id: "graphrag-bench-medical",
      track: "static-kb",
      documents: [{
        id: "document-1",
        sourceId: "source-1",
        title: "Long document",
        text: "word ".repeat(23_000),
        metadata: {},
      }],
      queries: [{
        id: "query-1",
        text: "word",
        referenceAnswer: "word",
        goldEvidenceIds: ["source-1"],
        goldEvidenceText: ["word"],
        answerable: true,
        category: "test",
        metadata: {},
      }],
      provenance: { source: "test", version: "1", license: "test" },
    };
    const firstClient = new FakeEmbeddingClient(2);
    const firstAdapter = new VectorRagRerankerAdapter(
      firstClient as unknown as EmbeddingClient,
      DEFAULT_RAG_EVAL_MANIFEST,
    );

    await expect(firstAdapter.retrieve(bundle, { workDirectory, topK: 1, candidateK: 1 }))
      .rejects.toThrow("simulated quota failure");
    expect(firstClient.documentBatchIds[0]).toHaveLength(100);

    const resumedClient = new FakeEmbeddingClient(null);
    const resumedAdapter = new VectorRagRerankerAdapter(
      resumedClient as unknown as EmbeddingClient,
      DEFAULT_RAG_EVAL_MANIFEST,
    );
    const results = await resumedAdapter.retrieve(bundle, { workDirectory, topK: 1, candidateK: 1 });

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
    return inputs.map((input) => ({ id: input.id, values: Array.from({ length: 1536 }, () => 0.01) }));
  }

  getUsage(): { requests: number; inputTokens: number; totalTokens: number } {
    return { requests: this.calls, inputTokens: this.inputTokens, totalTokens: this.inputTokens };
  }
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
