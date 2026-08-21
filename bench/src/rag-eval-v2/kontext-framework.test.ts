import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
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
