import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { type CommandRunner, runCommand } from "./codex-json.js";
import type {
  CorpusDocument,
  DatasetBundle,
  FrameworkDoctorResult,
  FrameworkId,
  RetrievalResult,
  RetrievedEvidence,
} from "./contracts.js";
import { readJsonLines, writeJsonAtomic, writeJsonLines } from "./jsonl.js";
import { KontextBrainAdapter } from "./kontext-framework.js";
import type { FrameworkManifest, RagEvalManifest } from "./manifest.js";
import { manifestDigest } from "./manifest.js";
import {
  type EmbeddingClient,
  OpenAIEmbeddingClient,
  cosineSimilarity,
} from "./openai-embeddings.js";

export interface FrameworkRunOptions {
  readonly workDirectory: string;
  readonly topK: number;
  readonly candidateK: number;
}

export interface FrameworkAdapter {
  readonly id: FrameworkId;
  doctor(): Promise<FrameworkDoctorResult>;
  retrieve(bundle: DatasetBundle, options: FrameworkRunOptions): Promise<RetrievalResult[]>;
}

interface VectorIndexMetadata {
  readonly schemaVersion: 1;
  readonly model: string;
  readonly dimensions: number;
  readonly count: number;
  readonly datasetId: string;
  readonly documentDigest: string;
  readonly embeddingInputTokens: number;
}

interface VectorEmbeddingBatchMetadata {
  readonly schemaVersion: 1;
  readonly model: string;
  readonly dimensions: number;
  readonly documentDigest: string;
  readonly offset: number;
  readonly ids: readonly string[];
  readonly inputTokens: number;
}

interface StoredDocument {
  readonly id: string;
  readonly sourceId: string;
  readonly title: string;
  readonly text: string;
  readonly metadata: CorpusDocument["metadata"];
}

export class VectorRagRerankerAdapter implements FrameworkAdapter {
  readonly id = "vector-rag-reranker" as const;

  constructor(
    private readonly embeddingClient: EmbeddingClient | null,
    private readonly manifest: RagEvalManifest,
  ) {}

  async doctor(): Promise<FrameworkDoctorResult> {
    return this.embeddingClient
      ? {
          frameworkId: this.id,
          status: "ready",
          version: "builtin-cosine+bm25-rrf-v1",
          detail: "OpenAI vector retrieval with BM25 reciprocal-rank-fusion reranking",
        }
      : {
          frameworkId: this.id,
          status: "blocked",
          version: "builtin-cosine+bm25-rrf-v1",
          detail: "OPENAI_API_KEY is not set",
        };
  }

  async retrieve(bundle: DatasetBundle, options: FrameworkRunOptions): Promise<RetrievalResult[]> {
    if (!this.embeddingClient) throw new Error("Vector adapter requires OPENAI_API_KEY");
    const indexDirectory = join(options.workDirectory, bundle.id, this.id, "index");
    const index = await this.loadOrBuildIndex(bundle, indexDirectory);
    const queryUsageBefore = this.embeddingClient.getUsage();
    const queryEmbeddings = await this.embeddingClient.embed(
      bundle.queries.map((query) => ({ id: query.id, text: query.text })),
      "RETRIEVAL_QUERY",
    );
    const queryUsageAfter = this.embeddingClient.getUsage();
    const queryInputTokens = queryUsageAfter.inputTokens - queryUsageBefore.inputTokens;
    const totalEmbeddingInputTokens = index.embeddingInputTokens + queryInputTokens;
    writeJsonAtomic(join(indexDirectory, "embedding-usage.json"), {
      provider: "openai",
      model: this.embeddingClient.model,
      dimensions: this.embeddingClient.dimensions,
      indexInputTokens: index.embeddingInputTokens,
      queryInputTokens,
      totalInputTokens: totalEmbeddingInputTokens,
      inputPriceUsdPerMillionTokens: 0.02,
      estimatedCostUsd: (totalEmbeddingInputTokens * 0.02) / 1_000_000,
    });
    const queryVectors = new Map(
      queryEmbeddings.map((embedding) => [embedding.id, embedding.values]),
    );
    const results: RetrievalResult[] = [];
    for (const query of bundle.queries) {
      const startedAt = performance.now();
      const queryVector = queryVectors.get(query.id);
      if (!queryVector) throw new Error(`Missing query embedding ${query.id}`);
      const vectorRanking = rankVectors(index.vectors, queryVector, options.candidateK);
      const candidates = vectorRanking.map((item) => index.documents[item.index]!);
      const lexicalRanking = rankBm25(candidates, query.text);
      const lexicalRankById = new Map(
        lexicalRanking.map((document, rank) => [document.id, rank + 1]),
      );
      const evidence = vectorRanking
        .map((item, vectorRank) => {
          const document = index.documents[item.index]!;
          const lexicalRank = lexicalRankById.get(document.id) ?? candidates.length + 1;
          return {
            document,
            score: 1 / (60 + vectorRank + 1) + 1 / (60 + lexicalRank),
            vectorScore: item.score,
          };
        })
        .sort(
          (left, right) =>
            right.score - left.score || left.document.id.localeCompare(right.document.id),
        )
        .slice(0, options.topK)
        .map<RetrievedEvidence>((item, rank) => ({
          id: item.document.id,
          sourceId: item.document.sourceId,
          text: item.document.text,
          score: item.score,
          rank: rank + 1,
          metadata: {
            ...item.document.metadata,
            vectorScore: item.vectorScore,
            reranker: "bm25-rrf",
          },
        }));
      results.push({
        datasetId: bundle.id,
        frameworkId: this.id,
        queryId: query.id,
        status: "ok",
        evidence,
        latencyMs: performance.now() - startedAt,
        inputTokens: null,
        error: null,
        frameworkVersion: "builtin-cosine+bm25-rrf-v1",
        configDigest: manifestDigest(this.manifest),
      });
    }
    return results;
  }

  private async loadOrBuildIndex(
    bundle: DatasetBundle,
    directory: string,
  ): Promise<{ documents: StoredDocument[]; vectors: Float32Array; embeddingInputTokens: number }> {
    mkdirSync(directory, { recursive: true });
    const metadataPath = join(directory, "index.json");
    const documentsPath = join(directory, "documents.jsonl");
    const vectorsPath = join(directory, "embeddings.f32");
    const chunkedDocuments = bundle.documents.flatMap((document) =>
      chunkForVectorBaseline(document),
    );
    const expectedDigest = digestDocuments(chunkedDocuments);
    if (existsSync(metadataPath) && existsSync(documentsPath) && existsSync(vectorsPath)) {
      const metadata = JSON.parse(readFileSync(metadataPath, "utf8")) as VectorIndexMetadata;
      if (
        metadata.datasetId === bundle.id &&
        metadata.documentDigest === expectedDigest &&
        metadata.model === this.manifest.models.embedding.model &&
        metadata.dimensions === this.manifest.models.embedding.dimensions
      ) {
        const documents = readJsonLines<StoredDocument>(documentsPath);
        const buffer = readFileSync(vectorsPath);
        const view = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
        return {
          documents,
          vectors: new Float32Array(view),
          embeddingInputTokens: metadata.embeddingInputTokens,
        };
      }
    }

    const embedded = await this.embedDocumentsWithCheckpoints(
      chunkedDocuments,
      join(directory, "embedding-batches"),
      expectedDigest,
    );
    const documents = chunkedDocuments.map<StoredDocument>((document) => ({
      id: document.id,
      sourceId: document.sourceId,
      title: document.title,
      text: document.text,
      metadata: document.metadata,
    }));
    const metadata: VectorIndexMetadata = {
      schemaVersion: 1,
      model: this.manifest.models.embedding.model,
      dimensions: this.manifest.models.embedding.dimensions!,
      count: documents.length,
      datasetId: bundle.id,
      documentDigest: expectedDigest,
      embeddingInputTokens: embedded.inputTokens,
    };
    writeJsonLines(documentsPath, documents);
    writeFileSync(vectorsPath, Buffer.from(embedded.vectors.buffer));
    writeFileSync(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`, "utf8");
    return {
      documents,
      vectors: embedded.vectors,
      embeddingInputTokens: embedded.inputTokens,
    };
  }

  private async embedDocumentsWithCheckpoints(
    documents: readonly CorpusDocument[],
    batchDirectory: string,
    documentDigest: string,
  ): Promise<{ vectors: Float32Array; inputTokens: number }> {
    const batchSize = 100;
    const dimensions = this.manifest.models.embedding.dimensions!;
    const model = this.manifest.models.embedding.model;
    const vectors = new Float32Array(documents.length * dimensions);
    let inputTokens = 0;
    mkdirSync(batchDirectory, { recursive: true });

    for (let offset = 0; offset < documents.length; offset += batchSize) {
      const batch = documents.slice(offset, offset + batchSize);
      const stem = `batch-${String(offset).padStart(8, "0")}`;
      const metadataPath = join(batchDirectory, `${stem}.json`);
      const vectorsPath = join(batchDirectory, `${stem}.f32`);
      const expectedIds = batch.map((document) => document.id);
      const cached = loadVectorEmbeddingBatch(
        metadataPath,
        vectorsPath,
        documentDigest,
        offset,
        expectedIds,
        model,
        dimensions,
      );
      if (cached) {
        vectors.set(cached.vectors, offset * dimensions);
        inputTokens += cached.inputTokens;
        continue;
      }

      const usageBefore = this.embeddingClient!.getUsage();
      const embeddings = await this.embeddingClient!.embed(
        batch.map((document) => ({ id: document.id, text: document.text, title: document.title })),
        "RETRIEVAL_DOCUMENT",
      );
      const usageAfter = this.embeddingClient!.getUsage();
      const batchInputTokens = usageAfter.inputTokens - usageBefore.inputTokens;
      const batchVectors = new Float32Array(embeddings.length * dimensions);
      embeddings.forEach((embedding, index) =>
        batchVectors.set(embedding.values, index * dimensions),
      );
      writeFileSync(
        vectorsPath,
        Buffer.from(batchVectors.buffer, batchVectors.byteOffset, batchVectors.byteLength),
      );
      const metadata: VectorEmbeddingBatchMetadata = {
        schemaVersion: 1,
        model,
        dimensions,
        documentDigest,
        offset,
        ids: expectedIds,
        inputTokens: batchInputTokens,
      };
      writeFileSync(metadataPath, `${JSON.stringify(metadata)}\n`, "utf8");
      vectors.set(batchVectors, offset * dimensions);
      inputTokens += batchInputTokens;
    }
    return { vectors, inputTokens };
  }
}

function loadVectorEmbeddingBatch(
  metadataPath: string,
  vectorsPath: string,
  documentDigest: string,
  offset: number,
  expectedIds: readonly string[],
  model: string,
  dimensions: number,
): { vectors: Float32Array; inputTokens: number } | null {
  if (!existsSync(metadataPath) || !existsSync(vectorsPath)) return null;
  try {
    const metadata = JSON.parse(readFileSync(metadataPath, "utf8")) as VectorEmbeddingBatchMetadata;
    if (
      metadata.schemaVersion !== 1 ||
      metadata.model !== model ||
      metadata.dimensions !== dimensions ||
      metadata.documentDigest !== documentDigest ||
      metadata.offset !== offset ||
      metadata.ids.length !== expectedIds.length ||
      metadata.ids.some((id, index) => id !== expectedIds[index])
    ) {
      return null;
    }
    const buffer = readFileSync(vectorsPath);
    if (buffer.byteLength !== expectedIds.length * dimensions * Float32Array.BYTES_PER_ELEMENT)
      return null;
    const view = new Float32Array(
      buffer.buffer,
      buffer.byteOffset,
      buffer.byteLength / Float32Array.BYTES_PER_ELEMENT,
    );
    return { vectors: new Float32Array(view), inputTokens: metadata.inputTokens };
  } catch {
    return null;
  }
}

export class ExternalCommandFrameworkAdapter implements FrameworkAdapter {
  readonly id: FrameworkId;
  private readonly command: readonly string[] | null;

  constructor(
    private readonly framework: FrameworkManifest,
    private readonly manifest: RagEvalManifest,
    private readonly commandRunner: CommandRunner = runCommand,
  ) {
    this.id = framework.id;
    const configured = framework.commandEnv ? process.env[framework.commandEnv] : undefined;
    this.command =
      configured === undefined ? defaultFrameworkCommand(framework.id) : parseCommand(configured);
  }

  async doctor(): Promise<FrameworkDoctorResult> {
    if (!this.command) {
      return {
        frameworkId: this.id,
        status: "blocked",
        version: "unresolved",
        detail: `${this.framework.commandEnv} is not configured as a JSON command array`,
      };
    }
    let result;
    try {
      result = await this.commandRunner(
        this.command[0]!,
        [...this.command.slice(1), "doctor"],
        "",
        180_000,
      );
    } catch (error) {
      return {
        frameworkId: this.id,
        status: "blocked",
        version: "unresolved",
        detail: `doctor failed: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
    if (result.exitCode !== 0) {
      return {
        frameworkId: this.id,
        status: "blocked",
        version: "unresolved",
        detail: result.stderr.trim() || `doctor exited ${result.exitCode}`,
      };
    }
    try {
      const payload = JSON.parse(result.stdout) as {
        version?: string;
        status?: string;
        detail?: string;
      };
      const versionMatches =
        !this.framework.pinnedVersion || payload.version === this.framework.pinnedVersion;
      return {
        frameworkId: this.id,
        status: payload.status === "ready" && versionMatches ? "ready" : "blocked",
        version: payload.version ?? "unresolved",
        detail: versionMatches
          ? (payload.detail ?? result.stdout.trim())
          : `Expected pinned version ${this.framework.pinnedVersion}, found ${payload.version ?? "unresolved"}`,
      };
    } catch {
      return {
        frameworkId: this.id,
        status: "blocked",
        version: "unresolved",
        detail: "doctor did not return JSON",
      };
    }
  }

  async retrieve(bundle: DatasetBundle, options: FrameworkRunOptions): Promise<RetrievalResult[]> {
    if (!this.command) throw new Error(`${this.framework.commandEnv} is not configured`);
    const frameworkDirectory = join(options.workDirectory, bundle.id, this.id);
    const datasetDirectory = join(frameworkDirectory, "dataset");
    const indexDirectory = join(frameworkDirectory, "index");
    const outputPath = join(frameworkDirectory, "retrieval.jsonl");
    mkdirSync(datasetDirectory, { recursive: true });
    mkdirSync(indexDirectory, { recursive: true });
    writeJsonLines(join(datasetDirectory, "corpus.jsonl"), bundle.documents);
    writeJsonLines(join(datasetDirectory, "queries.jsonl"), bundle.queries);
    const commonArgs = [
      "--dataset-dir",
      datasetDirectory,
      "--index-dir",
      indexDirectory,
      "--embedding-model",
      this.manifest.models.embedding.model,
      "--embedding-dimensions",
      String(this.manifest.models.embedding.dimensions),
      "--completion-model",
      this.manifest.models.answer.model,
      "--completion-reasoning-effort",
      this.manifest.models.answer.reasoningEffort!,
      "--completion-execution",
      this.manifest.models.answer.execution!,
      "--top-k",
      String(options.topK),
    ];
    const build = await this.commandRunner(
      this.command[0]!,
      [...this.command.slice(1), "build", ...commonArgs],
      "",
      24 * 60 * 60 * 1000,
    );
    if (build.exitCode !== 0) throw new Error(`${this.id} build failed: ${build.stderr}`);
    const retrieve = await this.commandRunner(
      this.command[0]!,
      [...this.command.slice(1), "retrieve", ...commonArgs, "--output", outputPath],
      "",
      24 * 60 * 60 * 1000,
    );
    if (retrieve.exitCode !== 0) throw new Error(`${this.id} retrieval failed: ${retrieve.stderr}`);
    const records = readJsonLines<RetrievalResult>(outputPath);
    validateExternalResults(bundle, this.id, records);
    const digest = manifestDigest(this.manifest);
    return records.map((record) => ({ ...record, configDigest: digest }));
  }
}

function defaultFrameworkCommand(id: FrameworkId): readonly string[] | null {
  const directoryName =
    id === "microsoft-graphrag"
      ? "graphrag"
      : id === "lightrag"
        ? "lightrag"
        : id === "hipporag2"
          ? "hipporag2"
          : null;
  if (!directoryName) return null;
  const projectDirectory = resolve(
    dirname(fileURLToPath(import.meta.url)),
    `../../framework-adapters/${directoryName}`,
  );
  return [
    "uv",
    "run",
    "--project",
    projectDirectory,
    "python",
    join(projectDirectory, "adapter.py"),
  ];
}

export function createFrameworkAdapters(manifest: RagEvalManifest): FrameworkAdapter[] {
  const apiKey = process.env.OPENAI_API_KEY ?? "";
  const embeddingClient = apiKey
    ? new OpenAIEmbeddingClient({
        apiKey,
        model: manifest.models.embedding.model,
        dimensions: manifest.models.embedding.dimensions,
        maxRetries: manifest.benchmarkPolicy.maxRetries,
      })
    : null;
  return manifest.frameworks.map((framework) => {
    if (framework.id === "kontext-brain") {
      return new KontextBrainAdapter(manifest, {
        embeddingClient,
        retrievalMode: kontextRetrievalMode(process.env.KONTEXT_RAG_EVAL_MODE),
        benchmarkDataDirectory: process.env.KONTEXT_RAG_EVAL_BENCH_DATA_DIR,
      });
    }
    if (framework.id === "vector-rag-reranker") {
      return new VectorRagRerankerAdapter(embeddingClient, manifest);
    }
    return new ExternalCommandFrameworkAdapter(framework, manifest);
  });
}

function kontextRetrievalMode(
  value: string | undefined,
):
  | "legacy"
  | "bidirectional-kg"
  | "max-existing-stack"
  | "source-hydrated-stack"
  | "source-hydrated-llm-stack"
  | "source-hydrated-llm-recall-safe-stack"
  | "source-hydrated-llm-candidate-safe-stack"
  | "source-hydrated-llm-coverage-aware-stack"
  | "multi-query-standard-rerank-stack"
  | "multi-query-coverage-aware-stack"
  | "adaptive-eece-stack" {
  if (
    value === "bidirectional-kg" ||
    value === "max-existing-stack" ||
    value === "source-hydrated-stack" ||
    value === "source-hydrated-llm-stack" ||
    value === "source-hydrated-llm-recall-safe-stack" ||
    value === "source-hydrated-llm-candidate-safe-stack" ||
    value === "source-hydrated-llm-coverage-aware-stack" ||
    value === "multi-query-standard-rerank-stack" ||
    value === "multi-query-coverage-aware-stack" ||
    value === "adaptive-eece-stack"
  ) {
    return value;
  }
  return "legacy";
}

function parseCommand(raw: string | undefined): readonly string[] | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (
      !Array.isArray(parsed) ||
      parsed.length === 0 ||
      parsed.some((item) => typeof item !== "string")
    ) {
      return null;
    }
    return parsed as string[];
  } catch {
    return null;
  }
}

function digestDocuments(documents: readonly CorpusDocument[]): string {
  const hash = createHash("sha256");
  for (const document of documents) {
    hash.update(document.id).update("\0").update(document.text).update("\0");
  }
  return hash.digest("hex");
}

function chunkForVectorBaseline(document: CorpusDocument): CorpusDocument[] {
  const chunkCharacters = 1024;
  const overlap = 128;
  const chunks: CorpusDocument[] = [];
  let start = 0;
  let index = 0;
  while (start < document.text.length) {
    const hardEnd = Math.min(document.text.length, start + chunkCharacters);
    let end = hardEnd;
    if (hardEnd < document.text.length) {
      const boundary = document.text.lastIndexOf(" ", hardEnd);
      if (boundary > start + 716) end = boundary;
    }
    const text = document.text.slice(start, end).trim();
    if (text) {
      chunks.push({
        id: `${document.id}::vector-chunk-${String(index).padStart(6, "0")}`,
        sourceId: document.sourceId,
        title: document.title,
        text,
        metadata: { ...document.metadata, chunkIndex: index, start, end },
      });
      index += 1;
    }
    if (end >= document.text.length) break;
    start = Math.max(start + 1, end - overlap);
  }
  return chunks;
}

function rankVectors(
  flatVectors: Float32Array,
  queryVector: readonly number[],
  candidateK: number,
): Array<{ index: number; score: number }> {
  const dimensions = queryVector.length;
  if (flatVectors.length % dimensions !== 0) throw new Error("Corrupt vector index length");
  const ranked: Array<{ index: number; score: number }> = [];
  for (let index = 0; index < flatVectors.length / dimensions; index += 1) {
    const vector = flatVectors.subarray(index * dimensions, (index + 1) * dimensions);
    ranked.push({ index, score: cosineSimilarity(vector, queryVector) });
  }
  return ranked
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .slice(0, candidateK);
}

function tokenize(value: string): string[] {
  return value
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter((token) => token.length >= 2);
}

export function rankBm25(documents: readonly StoredDocument[], query: string): StoredDocument[] {
  if (documents.length === 0) return [];
  const queryTokens = [...new Set(tokenize(query))];
  const tokenized = documents.map((document) => tokenize(document.text));
  const averageLength =
    tokenized.reduce((total, tokens) => total + tokens.length, 0) / tokenized.length;
  const documentFrequency = new Map<string, number>();
  for (const tokens of tokenized) {
    for (const token of new Set(tokens)) {
      if (queryTokens.includes(token))
        documentFrequency.set(token, (documentFrequency.get(token) ?? 0) + 1);
    }
  }
  const scores = documents.map((document, index) => {
    const tokens = tokenized[index]!;
    const frequencies = new Map<string, number>();
    for (const token of tokens) frequencies.set(token, (frequencies.get(token) ?? 0) + 1);
    let score = 0;
    for (const token of queryTokens) {
      const frequency = frequencies.get(token) ?? 0;
      if (frequency === 0) continue;
      const containingDocuments = documentFrequency.get(token) ?? 0;
      const inverseDocumentFrequency = Math.log(
        1 + (documents.length - containingDocuments + 0.5) / (containingDocuments + 0.5),
      );
      const denominator =
        frequency + 1.2 * (1 - 0.75 + 0.75 * (tokens.length / Math.max(1, averageLength)));
      score += inverseDocumentFrequency * ((frequency * 2.2) / denominator);
    }
    return { document, score };
  });
  return scores
    .sort(
      (left, right) =>
        right.score - left.score || left.document.id.localeCompare(right.document.id),
    )
    .map((item) => item.document);
}

function validateExternalResults(
  bundle: DatasetBundle,
  frameworkId: FrameworkId,
  records: readonly RetrievalResult[],
): void {
  const expectedQueryIds = new Set(bundle.queries.map((query) => query.id));
  const seen = new Set<string>();
  for (const record of records) {
    if (record.datasetId !== bundle.id || record.frameworkId !== frameworkId) {
      throw new Error(
        `External adapter returned mismatched dataset/framework for ${record.queryId}`,
      );
    }
    if (!expectedQueryIds.has(record.queryId))
      throw new Error(`Unexpected query ${record.queryId}`);
    if (seen.has(record.queryId)) throw new Error(`Duplicate result ${record.queryId}`);
    seen.add(record.queryId);
    for (const [index, evidence] of record.evidence.entries()) {
      if (evidence.rank !== index + 1)
        throw new Error(`Non-contiguous evidence rank for ${record.queryId}`);
    }
  }
  if (seen.size !== expectedQueryIds.size) {
    throw new Error(
      `External adapter returned ${seen.size}/${expectedQueryIds.size} query results`,
    );
  }
}
