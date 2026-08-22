import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { CodexJsonClient, type CommandRunner } from "./codex-json.js";
import type { DatasetBundle } from "./contracts.js";
import { KontextBrainAdapter } from "./kontext-framework.js";
import { DEFAULT_RAG_EVAL_MANIFEST } from "./manifest.js";
import type { EmbeddingClient, EmbeddingInput, EmbeddingTask } from "./openai-embeddings.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("KontextBrainAdapter bidirectional KG mode", () => {
  it("publishes an isolated v13 version for anchored evidence answering", async () => {
    const adapter = new KontextBrainAdapter(DEFAULT_RAG_EVAL_MANIFEST, {
      embeddingClient: null,
      retrievalMode: "multi-query-anchored-evidence-answer-stack",
    });

    await expect(adapter.doctor()).resolves.toMatchObject({
      status: "blocked",
      version: "workspace-0.1.0+v13-anchored-evidence-answer-stack",
    });
  });

  it("freezes v13 candidate and perspective-fusion settings", async () => {
    const root = mkdtempSync(join(tmpdir(), "kontext-rag-eval-v13-"));
    temporaryDirectories.push(root);
    const dataDirectory = join(root, "data");
    const workDirectory = join(root, "run");
    mkdirSync(dataDirectory, { recursive: true });
    writeFileSync(
      join(dataDirectory, "gb-medical-chunks.jsonl"),
      `${JSON.stringify({ id: "med-0", body: "Alpha evidence establishes the requested fact." })}\n`,
    );
    writeFileSync(
      join(dataDirectory, "gb-medical-kg.json"),
      `${JSON.stringify({ entities: [], edges: [], chunkToEntities: [["med-0", []]] })}\n`,
    );
    const runner: CommandRunner = async (_command, args, stdin) => {
      const outputPath = args[args.indexOf("--output-last-message") + 1];
      if (!outputPath) throw new Error("Codex command omitted --output-last-message path");
      const text = stdin.includes("Generate up to three complementary")
        ? JSON.stringify({ queries: ["Which literal alpha passage supports the fact?"] })
        : JSON.stringify({ ranked_ids: ["med-0"] });
      writeFileSync(outputPath, JSON.stringify({ text }));
      return { exitCode: 0, stdout: "", stderr: "", durationMs: 1 };
    };
    const adapter = new KontextBrainAdapter(DEFAULT_RAG_EVAL_MANIFEST, {
      codexClient: new CodexJsonClient(runner),
      embeddingClient: new FakeEmbeddingClient(),
      retrievalMode: "multi-query-anchored-evidence-answer-stack",
      benchmarkDataDirectory: dataDirectory,
    });

    const results = await adapter.retrieve(testBundle(), {
      workDirectory,
      topK: 1,
      candidateK: 1,
    });

    expect(results[0]).toMatchObject({
      status: "ok",
      frameworkVersion: "workspace-0.1.0+v13-anchored-evidence-answer-stack",
      evidence: [{ metadata: { retrievalMode: "v13-anchored-evidence-answer-stack" } }],
    });
    const config = JSON.parse(
      readFileSync(
        join(
          workDirectory,
          "graphrag-bench-medical",
          "kontext-brain",
          "index",
          "v13-anchored-evidence-answer-stack",
          "kontext-kg-config.json",
        ),
        "utf8",
      ),
    ) as {
      answerPolicy: string;
      embedding: { vectorCandidateCount: number };
      multiQuery: {
        maximumExpandedQueries: number;
        perspectiveFusion: {
          reciprocalRankConstant: number;
          originalQueryWeight: number;
          expandedQueryWeight: number;
        };
      };
    };
    expect(config.answerPolicy).toBe("supported-evidence-needs");
    expect(config.embedding.vectorCandidateCount).toBe(50);
    expect(config.multiQuery).toMatchObject({
      maximumExpandedQueries: 3,
      perspectiveFusion: {
        reciprocalRankConstant: 10,
        originalQueryWeight: 2,
        expandedQueryWeight: 1,
      },
    });
  });

  it("runs v14a from read-only v13 caches without Codex or embedding calls", async () => {
    const root = mkdtempSync(join(tmpdir(), "kontext-rag-eval-v14a-"));
    temporaryDirectories.push(root);
    const dataDirectory = join(root, "data");
    const seedWorkDirectory = join(root, "seed-run");
    const workDirectory = join(root, "v14-run");
    mkdirSync(dataDirectory, { recursive: true });
    writeFileSync(
      join(dataDirectory, "gb-medical-chunks.jsonl"),
      `${JSON.stringify({ id: "med-0", body: "Alpha evidence establishes the requested fact." })}\n`,
    );
    writeFileSync(
      join(dataDirectory, "gb-medical-kg.json"),
      `${JSON.stringify({ entities: [], edges: [], chunkToEntities: [["med-0", []]] })}\n`,
    );
    const seedRunner: CommandRunner = async (_command, args, stdin) => {
      const outputPath = args[args.indexOf("--output-last-message") + 1];
      if (!outputPath) throw new Error("Codex command omitted --output-last-message path");
      const text = stdin.includes("Generate up to three complementary")
        ? JSON.stringify({ queries: ["Which literal alpha passage supports the fact?"] })
        : JSON.stringify({ ranked_ids: ["med-0"] });
      writeFileSync(outputPath, JSON.stringify({ text }));
      return { exitCode: 0, stdout: "", stderr: "", durationMs: 1 };
    };
    const seedAdapter = new KontextBrainAdapter(DEFAULT_RAG_EVAL_MANIFEST, {
      codexClient: new CodexJsonClient(seedRunner),
      embeddingClient: new FakeEmbeddingClient(),
      retrievalMode: "multi-query-anchored-evidence-answer-stack",
      benchmarkDataDirectory: dataDirectory,
    });
    await seedAdapter.retrieve(testBundle(), {
      workDirectory: seedWorkDirectory,
      topK: 1,
      candidateK: 1,
    });
    const precomputedIndexDirectory = join(
      seedWorkDirectory,
      "graphrag-bench-medical",
      "kontext-brain",
      "index",
      "v13-anchored-evidence-answer-stack",
    );
    const cacheDirectories = [
      "document-embedding-batches",
      "query-embedding-batches",
      "multi-query-expansions",
      "multi-query-embedding-batches",
    ].map((name) => join(precomputedIndexDirectory, name));
    const beforeMtimes = cacheDirectories.map((directory) => statSync(directory).mtimeMs);
    const embeddingClient = new ThrowingEmbeddingClient();
    let codexCalls = 0;
    const adapter = new KontextBrainAdapter(DEFAULT_RAG_EVAL_MANIFEST, {
      codexClient: new CodexJsonClient(async () => {
        codexCalls += 1;
        throw new Error("v14 cache-only retrieval must not call Codex");
      }),
      embeddingClient,
      retrievalMode: "v14a-anchored-deterministic-soft-coverage-stack",
      benchmarkDataDirectory: dataDirectory,
      precomputedIndexDirectory,
    });

    const results = await adapter.retrieve(testBundle(), {
      workDirectory,
      topK: 1,
      candidateK: 1,
    });

    expect(results[0]).toMatchObject({
      status: "ok",
      frameworkVersion: "workspace-0.1.0+v14a-anchored-deterministic-soft-coverage-stack",
      evidence: [
        { metadata: { retrievalMode: "v14a-anchored-deterministic-soft-coverage-stack" } },
      ],
    });
    expect(embeddingClient.embedCalls).toBe(0);
    expect(codexCalls).toBe(0);
    expect(cacheDirectories.map((directory) => statSync(directory).mtimeMs)).toEqual(beforeMtimes);
    const config = JSON.parse(
      readFileSync(
        join(
          workDirectory,
          "graphrag-bench-medical",
          "kontext-brain",
          "index",
          "v14a-anchored-deterministic-soft-coverage-stack",
          "kontext-kg-config.json",
        ),
        "utf8",
      ),
    ) as {
      cacheReuse: {
        readOnly: boolean;
        newCodexCalls: number;
        newEmbeddingCalls: number;
        newInputTokens: number;
      };
      embedding: { cacheOnly: boolean; vectorCandidateCount: number };
      llmReranker: null;
      multiQuery: {
        execution: string;
        perspectiveFusion: { originalQueryWeight: number; expandedQueryWeight: number };
      };
      deterministicCoverage: { policy: string };
    };
    expect(config).toMatchObject({
      cacheReuse: {
        readOnly: true,
        newCodexCalls: 0,
        newEmbeddingCalls: 0,
        newInputTokens: 0,
      },
      embedding: { cacheOnly: true, vectorCandidateCount: 50 },
      llmReranker: null,
      multiQuery: {
        execution: "precomputed-cache",
        perspectiveFusion: { originalQueryWeight: 2, expandedQueryWeight: 1 },
      },
      deterministicCoverage: { policy: "soft" },
    });
    expect(
      readdirSync(
        join(
          workDirectory,
          "graphrag-bench-medical",
          "kontext-brain",
          "index",
          "v14a-anchored-deterministic-soft-coverage-stack",
        ),
      ).sort(),
    ).toEqual(["kontext-kg-config.json", "retrieval-checkpoints"]);
  });

  it("applies v14b original and per-expansion quotas to the final evidence window", async () => {
    const root = mkdtempSync(join(tmpdir(), "kontext-rag-eval-v14b-"));
    temporaryDirectories.push(root);
    const seedWorkDirectory = join(root, "seed-run");
    const bundle = perspectiveBundle();
    const runner: CommandRunner = async (_command, args, stdin) => {
      const outputPath = args[args.indexOf("--output-last-message") + 1];
      if (!outputPath) throw new Error("Codex command omitted --output-last-message path");
      const text = stdin.includes("Generate up to three complementary")
        ? JSON.stringify({ queries: ["expansion one", "expansion two", "expansion three"] })
        : JSON.stringify({
            ranked_ids: bundle.documents.map((document) => `${document.id}-0`),
          });
      writeFileSync(outputPath, JSON.stringify({ text }));
      return { exitCode: 0, stdout: "", stderr: "", durationMs: 1 };
    };
    const seedAdapter = new KontextBrainAdapter(DEFAULT_RAG_EVAL_MANIFEST, {
      codexClient: new CodexJsonClient(runner),
      embeddingClient: new PerspectiveEmbeddingClient(),
      retrievalMode: "multi-query-anchored-evidence-answer-stack",
    });
    await seedAdapter.retrieve(bundle, {
      workDirectory: seedWorkDirectory,
      topK: 10,
      candidateK: 50,
    });
    const precomputedIndexDirectory = join(
      seedWorkDirectory,
      bundle.id,
      "kontext-brain",
      "index",
      "v13-anchored-evidence-answer-stack",
    );
    const adapter = new KontextBrainAdapter(DEFAULT_RAG_EVAL_MANIFEST, {
      codexClient: new CodexJsonClient(async () => {
        throw new Error("v14 cache-only retrieval must not call Codex");
      }),
      embeddingClient: new ThrowingPerspectiveEmbeddingClient(),
      retrievalMode: "v14b-anchored-deterministic-quota-coverage-stack",
      precomputedIndexDirectory,
    });

    const [result] = await adapter.retrieve(bundle, {
      workDirectory: join(root, "v14-run"),
      topK: 10,
      candidateK: 50,
    });

    expect(result?.status).toBe("ok");
    expect(result?.evidence.slice(0, 5).map((item) => item.sourceId)).toEqual([
      "original-1",
      "original-2",
      "original-3",
      "original-4",
      "original-5",
    ]);
    expect(result?.evidence.slice(5, 8).map((item) => item.sourceId)).toEqual([
      "expansion-1",
      "expansion-2",
      "expansion-3",
    ]);
    const softAdapter = new KontextBrainAdapter(DEFAULT_RAG_EVAL_MANIFEST, {
      codexClient: new CodexJsonClient(async () => {
        throw new Error("v14 cache-only retrieval must not call Codex");
      }),
      embeddingClient: new ThrowingPerspectiveEmbeddingClient(),
      retrievalMode: "v14a-anchored-deterministic-soft-coverage-stack",
      precomputedIndexDirectory,
    });
    const [softResult] = await softAdapter.retrieve(bundle, {
      workDirectory: join(root, "v14a-run"),
      topK: 10,
      candidateK: 50,
    });
    expect(result?.configDigest).not.toBe(softResult?.configDigest);
    const quotaConfig = JSON.parse(
      readFileSync(
        join(
          root,
          "v14-run",
          bundle.id,
          "kontext-brain",
          "index",
          "v14b-anchored-deterministic-quota-coverage-stack",
          "kontext-kg-config.json",
        ),
        "utf8",
      ),
    ) as { deterministicCoverage: Record<string, unknown>; llmReranker: unknown };
    expect(quotaConfig).toMatchObject({
      deterministicCoverage: {
        policy: "quota",
        topWindow: 10,
        originalQuota: 5,
        perExpansionQuota: 1,
        goldAccess: false,
      },
      llmReranker: null,
    });
  });

  it.each([
    "document-embedding-batches",
    "query-embedding-batches",
    "multi-query-expansions",
    "multi-query-embedding-batches",
  ])("fails closed without external calls when %s is missing or invalid", async (artifact) => {
    const root = mkdtempSync(join(tmpdir(), "kontext-rag-eval-v14-cache-miss-"));
    temporaryDirectories.push(root);
    const { dataDirectory, precomputedIndexDirectory } = await seedMedicalV13Cache(root);
    const artifactDirectory = join(precomputedIndexDirectory, artifact);
    if (artifact === "document-embedding-batches" || artifact === "multi-query-embedding-batches") {
      rmSync(artifactDirectory, { recursive: true, force: true });
    } else {
      const metadata = readdirSync(artifactDirectory).find((name) => name.endsWith(".json"));
      if (!metadata) throw new Error(`Missing test fixture metadata for ${artifact}`);
      writeFileSync(join(artifactDirectory, metadata), "{}\n", "utf8");
    }
    const embeddingClient = new ThrowingEmbeddingClient();
    let codexCalls = 0;
    const adapter = new KontextBrainAdapter(DEFAULT_RAG_EVAL_MANIFEST, {
      codexClient: new CodexJsonClient(async () => {
        codexCalls += 1;
        throw new Error("v14 cache-only retrieval must not call Codex");
      }),
      embeddingClient,
      retrievalMode: "v14a-anchored-deterministic-soft-coverage-stack",
      benchmarkDataDirectory: dataDirectory,
      precomputedIndexDirectory,
    });

    await expect(
      adapter.retrieve(testBundle(), {
        workDirectory: join(root, "v14-run"),
        topK: 1,
        candidateK: 1,
      }),
    ).rejects.toThrow(/Required cached .* missing or invalid/);
    expect(embeddingClient.embedCalls).toBe(0);
    expect(codexCalls).toBe(0);
  });

  it("routes retrieval through KontextAgent's evidence-backed branch", async () => {
    const root = mkdtempSync(join(tmpdir(), "kontext-rag-eval-kg-"));
    temporaryDirectories.push(root);
    const dataDirectory = join(root, "data");
    const workDirectory = join(root, "run");
    mkdirSync(dataDirectory, { recursive: true });
    writeFileSync(
      join(dataDirectory, "gb-medical-chunks.jsonl"),
      `${JSON.stringify({ id: "med-0", body: "Alpha evidence establishes the requested fact." })}\n`,
    );
    writeFileSync(
      join(dataDirectory, "gb-medical-kg.json"),
      `${JSON.stringify({ entities: [], edges: [], chunkToEntities: [["med-0", []]] })}\n`,
    );
    writeFileSync(join(dataDirectory, "gb-novel-chunks.jsonl"), "");
    writeFileSync(
      join(dataDirectory, "gb-novel-kg.json"),
      `${JSON.stringify({ entities: [], edges: [], chunkToEntities: [] })}\n`,
    );

    const adapter = new KontextBrainAdapter(DEFAULT_RAG_EVAL_MANIFEST, {
      embeddingClient: new FakeEmbeddingClient(),
      retrievalMode: "bidirectional-kg",
      benchmarkDataDirectory: dataDirectory,
    });
    const results = await adapter.retrieve(testBundle(), { workDirectory, topK: 1, candidateK: 1 });

    expect(results).toHaveLength(1);
    expect(results[0]).toMatchObject({
      status: "ok",
      frameworkVersion: "workspace-0.1.0+bidirectional-kg-v2",
      evidence: [
        {
          id: "chunk:med-0",
          text: "Alpha evidence establishes the requested fact.",
          metadata: { retrievalMode: "bidirectional", chunkId: "med-0" },
        },
      ],
    });
  });

  it("keeps v3 max-existing-stack artifacts and version separate from v2", async () => {
    const root = mkdtempSync(join(tmpdir(), "kontext-rag-eval-max-stack-"));
    temporaryDirectories.push(root);
    const dataDirectory = join(root, "data");
    const workDirectory = join(root, "run");
    mkdirSync(dataDirectory, { recursive: true });
    writeFileSync(
      join(dataDirectory, "gb-medical-chunks.jsonl"),
      `${JSON.stringify({ id: "med-0", body: "Alpha evidence establishes the requested fact." })}\n`,
    );
    writeFileSync(
      join(dataDirectory, "gb-medical-kg.json"),
      `${JSON.stringify({ entities: [], edges: [], chunkToEntities: [["med-0", []]] })}\n`,
    );
    writeFileSync(join(dataDirectory, "gb-novel-chunks.jsonl"), "");
    writeFileSync(
      join(dataDirectory, "gb-novel-kg.json"),
      `${JSON.stringify({ entities: [], edges: [], chunkToEntities: [] })}\n`,
    );

    const adapter = new KontextBrainAdapter(DEFAULT_RAG_EVAL_MANIFEST, {
      embeddingClient: new FakeEmbeddingClient(),
      retrievalMode: "max-existing-stack",
      benchmarkDataDirectory: dataDirectory,
    });
    const results = await adapter.retrieve(testBundle(), { workDirectory, topK: 1, candidateK: 1 });

    expect(results[0]).toMatchObject({
      status: "ok",
      frameworkVersion: "workspace-0.1.0+v3-max-existing-stack",
      evidence: [
        {
          id: "chunk:med-0",
          metadata: {
            retrievalMode: "v3-max-existing-stack",
            chunkId: "med-0",
            vectorRank: 1,
            graphRank: 1,
            bm25Rank: 1,
            contextRerank: 1,
          },
        },
      ],
    });
    const config = JSON.parse(
      readFileSync(
        join(
          workDirectory,
          "graphrag-bench-medical",
          "kontext-brain",
          "index",
          "v3-max-existing-stack",
          "kontext-kg-config.json",
        ),
        "utf8",
      ),
    ) as { retrievalMode: string; frameworkVersion: string };
    expect(config).toEqual(
      expect.objectContaining({
        retrievalMode: "v3-max-existing-stack",
        frameworkVersion: "workspace-0.1.0+v3-max-existing-stack",
      }),
    );
  });

  it("keeps source-hydrated retrieval in a separate v4 index and returns source windows", async () => {
    const root = mkdtempSync(join(tmpdir(), "kontext-rag-eval-source-hydrated-"));
    temporaryDirectories.push(root);
    const dataDirectory = join(root, "data");
    const workDirectory = join(root, "run");
    mkdirSync(dataDirectory, { recursive: true });
    writeFileSync(
      join(dataDirectory, "gb-medical-chunks.jsonl"),
      [
        { id: "med-0", body: "Alpha evidence establishes the requested fact." },
        {
          id: "med-1",
          body: "The following source paragraph supplies useful surrounding context.",
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );
    writeFileSync(
      join(dataDirectory, "gb-medical-kg.json"),
      `${JSON.stringify({
        entities: [],
        edges: [],
        chunkToEntities: [
          ["med-0", []],
          ["med-1", []],
        ],
      })}\n`,
    );
    writeFileSync(join(dataDirectory, "gb-novel-chunks.jsonl"), "");
    writeFileSync(
      join(dataDirectory, "gb-novel-kg.json"),
      `${JSON.stringify({ entities: [], edges: [], chunkToEntities: [] })}\n`,
    );

    const adapter = new KontextBrainAdapter(DEFAULT_RAG_EVAL_MANIFEST, {
      embeddingClient: new FakeEmbeddingClient(),
      retrievalMode: "source-hydrated-stack",
      benchmarkDataDirectory: dataDirectory,
    });
    const results = await adapter.retrieve(testBundle(), { workDirectory, topK: 1, candidateK: 1 });

    expect(results[0]).toMatchObject({
      status: "ok",
      frameworkVersion: "workspace-0.1.0+v4-source-hydrated-stack",
      evidence: [
        {
          id: "source-window:med:0-1",
          sourceId: "med",
          metadata: {
            retrievalMode: "v4-source-hydrated-stack",
            anchorChunkIds: "med-0",
            sourceChunkIds: "med-0,med-1",
          },
        },
      ],
    });
    expect(
      readFileSync(
        join(
          workDirectory,
          "graphrag-bench-medical",
          "kontext-brain",
          "index",
          "v4-source-hydrated-stack",
          "kontext-kg-config.json",
        ),
        "utf8",
      ),
    ).toContain('"windowCharacters": 5000');
  });

  it("runs the same source/chunk stack on a canonical static corpus without dataset branches", async () => {
    const root = mkdtempSync(join(tmpdir(), "kontext-rag-eval-canonical-"));
    temporaryDirectories.push(root);
    const adapter = new KontextBrainAdapter(DEFAULT_RAG_EVAL_MANIFEST, {
      embeddingClient: new FakeEmbeddingClient(),
      retrievalMode: "source-hydrated-stack",
      benchmarkDataDirectory: join(root, "unused-data"),
    });
    const bundle: DatasetBundle = {
      id: "beir-scifact",
      track: "static-kb",
      documents: [
        {
          id: "doc-1",
          sourceId: "doc-1",
          title: "Scientific claim",
          text: "Alpha evidence establishes the requested scientific fact.",
          metadata: {},
        },
      ],
      queries: [
        {
          id: "query-1",
          text: "What establishes the scientific fact?",
          referenceAnswer: null,
          goldEvidenceIds: ["doc-1"],
          goldEvidenceText: [],
          answerable: true,
          category: "retrieval",
          metadata: {},
        },
      ],
      provenance: { source: "test", version: "1", license: "test" },
    };

    const results = await adapter.retrieve(bundle, {
      workDirectory: join(root, "run"),
      topK: 1,
      candidateK: 1,
    });

    expect(results[0]).toMatchObject({
      status: "ok",
      evidence: [
        {
          sourceId: "doc-1",
          metadata: {
            retrievalMode: "v4-source-hydrated-stack",
            sourceChunkIds: "doc-1-0",
          },
        },
      ],
    });
  });
});

class FakeEmbeddingClient implements EmbeddingClient {
  readonly model = "text-embedding-3-small";
  readonly dimensions = 1536;
  private requests = 0;
  private inputTokens = 0;

  async embed(inputs: readonly EmbeddingInput[], _task: EmbeddingTask) {
    this.requests += 1;
    this.inputTokens += inputs.length;
    return inputs.map((input) => ({
      id: input.id,
      values: [1, ...Array.from({ length: this.dimensions - 1 }, () => 0)],
    }));
  }

  getUsage() {
    return {
      requests: this.requests,
      inputTokens: this.inputTokens,
      totalTokens: this.inputTokens,
    };
  }
}

class ThrowingEmbeddingClient implements EmbeddingClient {
  readonly model = "text-embedding-3-small";
  readonly dimensions = 1536;
  embedCalls = 0;

  async embed(_inputs: readonly EmbeddingInput[], _task: EmbeddingTask): Promise<never> {
    this.embedCalls += 1;
    throw new Error("v14 cache-only retrieval must not call the embedding API");
  }

  getUsage() {
    return { requests: this.embedCalls, inputTokens: 0, totalTokens: 0 };
  }
}

class PerspectiveEmbeddingClient implements EmbeddingClient {
  readonly model = "text-embedding-3-small";
  readonly dimensions = 4;
  private requests = 0;
  private inputTokens = 0;

  async embed(inputs: readonly EmbeddingInput[], _task: EmbeddingTask) {
    this.requests += 1;
    this.inputTokens += inputs.length;
    return inputs.map((input) => ({ id: input.id, values: perspectiveVector(input.text) }));
  }

  getUsage() {
    return {
      requests: this.requests,
      inputTokens: this.inputTokens,
      totalTokens: this.inputTokens,
    };
  }
}

class ThrowingPerspectiveEmbeddingClient extends PerspectiveEmbeddingClient {
  override async embed(_inputs: readonly EmbeddingInput[], _task: EmbeddingTask): Promise<never> {
    throw new Error("v14 cache-only retrieval must not call the embedding API");
  }
}

function perspectiveVector(text: string): number[] {
  const value = text.toLowerCase();
  if (value.includes("expansion one")) return [0, 1, 0, 0];
  if (value.includes("expansion two")) return [0, 0, 1, 0];
  if (value.includes("expansion three")) return [0, 0, 0, 1];
  if (value.includes("unrelated filler")) return [0.5, 0.5, 0.5, 0.5];
  return [1, 0, 0, 0];
}

function perspectiveBundle(): DatasetBundle {
  const documents = [
    ...Array.from({ length: 5 }, (_, index) => ({
      id: `original-${index + 1}`,
      sourceId: `original-${index + 1}`,
      title: `Original ${index + 1}`,
      text: `Original question evidence ${index + 1}`,
      metadata: {},
    })),
    ...Array.from({ length: 3 }, (_, index) => ({
      id: `expansion-${index + 1}`,
      sourceId: `expansion-${index + 1}`,
      title: `Expansion ${index + 1}`,
      text: `Expansion ${["one", "two", "three"][index]} evidence`,
      metadata: {},
    })),
    ...Array.from({ length: 20 }, (_, index) => ({
      id: `filler-${index + 1}`,
      sourceId: `filler-${index + 1}`,
      title: `Filler ${index + 1}`,
      text: `Unrelated filler ${index + 1}`,
      metadata: {},
    })),
  ];
  return {
    id: "beir-scifact",
    track: "static-kb",
    documents,
    queries: [
      {
        id: "perspective-q-1",
        text: "Original question",
        referenceAnswer: null,
        goldEvidenceIds: [],
        goldEvidenceText: [],
        answerable: true,
        category: "test",
        metadata: {},
      },
    ],
    provenance: { source: "test", version: "1", license: "test" },
  };
}

async function seedMedicalV13Cache(root: string): Promise<{
  dataDirectory: string;
  precomputedIndexDirectory: string;
}> {
  const dataDirectory = join(root, "data");
  const seedWorkDirectory = join(root, "seed-run");
  mkdirSync(dataDirectory, { recursive: true });
  writeFileSync(
    join(dataDirectory, "gb-medical-chunks.jsonl"),
    `${JSON.stringify({ id: "med-0", body: "Alpha evidence establishes the requested fact." })}\n`,
  );
  writeFileSync(
    join(dataDirectory, "gb-medical-kg.json"),
    `${JSON.stringify({ entities: [], edges: [], chunkToEntities: [["med-0", []]] })}\n`,
  );
  const runner: CommandRunner = async (_command, args, stdin) => {
    const outputPath = args[args.indexOf("--output-last-message") + 1];
    if (!outputPath) throw new Error("Codex command omitted --output-last-message path");
    const text = stdin.includes("Generate up to three complementary")
      ? JSON.stringify({ queries: ["Which literal alpha passage supports the fact?"] })
      : JSON.stringify({ ranked_ids: ["med-0"] });
    writeFileSync(outputPath, JSON.stringify({ text }));
    return { exitCode: 0, stdout: "", stderr: "", durationMs: 1 };
  };
  const seedAdapter = new KontextBrainAdapter(DEFAULT_RAG_EVAL_MANIFEST, {
    codexClient: new CodexJsonClient(runner),
    embeddingClient: new FakeEmbeddingClient(),
    retrievalMode: "multi-query-anchored-evidence-answer-stack",
    benchmarkDataDirectory: dataDirectory,
  });
  await seedAdapter.retrieve(testBundle(), {
    workDirectory: seedWorkDirectory,
    topK: 1,
    candidateK: 1,
  });
  return {
    dataDirectory,
    precomputedIndexDirectory: join(
      seedWorkDirectory,
      "graphrag-bench-medical",
      "kontext-brain",
      "index",
      "v13-anchored-evidence-answer-stack",
    ),
  };
}

function testBundle(): DatasetBundle {
  return {
    id: "graphrag-bench-medical",
    track: "static-kb",
    documents: [
      {
        id: "Medical",
        sourceId: "Medical",
        title: "Medical",
        text: "Alpha evidence establishes the requested fact.",
        metadata: {},
      },
    ],
    queries: [
      {
        id: "q-1",
        text: "What establishes the fact?",
        referenceAnswer: "Alpha evidence",
        goldEvidenceIds: ["Medical"],
        goldEvidenceText: ["Alpha evidence establishes the requested fact."],
        answerable: true,
        category: "Fact Retrieval",
        metadata: {},
      },
    ],
    provenance: { source: "test", version: "1", license: "test" },
  };
}
