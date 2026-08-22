import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  BidirectionalNLayerRetriever,
  type BidirectionalRetrievalInput,
  type BidirectionalRetrievalResult,
  ContentFetcherRegistry,
  DEFAULT_PIPELINE,
  DataSource,
  FileOntologyStore,
  InMemoryMetaIndexStore,
  InMemoryVectorStore,
  IngestPipeline,
  KeywordMappingStrategy,
  type LLMAdapter,
  OntologyGraph,
  type Principal,
  RouterLLMAdapter,
  ScoreBasedSelector,
  type SearchBudget,
  type SearchGraphPort,
  type SourceChunk,
  SourceContextHydrator,
  type SourceContextPolicy,
  TraversalStrategy,
  toEdges,
  toGraphConfig,
  toOntologyNodes,
} from "@kontext-brain/core";
import { KontextAgent } from "@kontext-brain/loader";
import {
  GenericMCPLayerAdapter,
  type MCPConnector,
  MCPContentFetcherBridge,
  type MCPData,
  type MCPResource,
} from "@kontext-brain/mcp";
import { BidirectionalBenchmarkSearchGraph } from "../bidirectional-benchmark-search-graph.js";
import type { BenchDoc } from "../corpus.js";
import type { KGSerialized, KGStore } from "../kg-builder.js";
import { CodexJsonClient, runCommand } from "./codex-json.js";
import type {
  AnswerPolicy,
  CorpusDocument,
  DatasetBundle,
  FrameworkDoctorResult,
  RetrievalResult,
} from "./contracts.js";
import type { FrameworkAdapter, FrameworkRunOptions } from "./frameworks.js";
import { readJsonLines, writeJsonAtomic } from "./jsonl.js";
import { LlmEvidenceReranker } from "./llm-evidence-reranker.js";
import { type RagEvalManifest, manifestDigest } from "./manifest.js";
import {
  CorpusBm25Ranker,
  applyOriginalAndExpansionQuota,
  fuseQueryPerspectives,
  fuseRankings,
} from "./max-existing-stack.js";
import { MultiQueryExpander, type MultiQueryExpansion } from "./multi-query-expander.js";
import {
  type EmbeddingClient,
  type EmbeddingInput,
  cosineSimilarity,
} from "./openai-embeddings.js";

export type KontextRetrievalMode =
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
  | "multi-query-plan-aware-coverage-stack"
  | "multi-query-anchored-evidence-answer-stack"
  | "corpus-complete-anchored-evidence-answer-stack"
  | "v14a-anchored-deterministic-soft-coverage-stack"
  | "v14b-anchored-deterministic-quota-coverage-stack"
  | "adaptive-eece-stack";

export interface KontextBrainAdapterOptions {
  readonly codexClient?: CodexJsonClient;
  readonly embeddingClient?: EmbeddingClient | null;
  readonly retrievalMode?: KontextRetrievalMode;
  readonly benchmarkDataDirectory?: string;
  readonly precomputedIndexDirectory?: string;
}

const BIDIRECTIONAL_FRAMEWORK_VERSION = "workspace-0.1.0+bidirectional-kg-v2";
const MAX_EXISTING_STACK_FRAMEWORK_VERSION = "workspace-0.1.0+v3-max-existing-stack";
const SOURCE_HYDRATED_STACK_FRAMEWORK_VERSION = "workspace-0.1.0+v4-source-hydrated-stack";
const SOURCE_HYDRATED_LLM_STACK_FRAMEWORK_VERSION = "workspace-0.1.0+v5-source-hydrated-llm-stack";
const SOURCE_HYDRATED_LLM_RECALL_SAFE_STACK_FRAMEWORK_VERSION =
  "workspace-0.1.0+v6-source-hydrated-llm-recall-safe-stack";
const SOURCE_HYDRATED_LLM_CANDIDATE_SAFE_STACK_FRAMEWORK_VERSION =
  "workspace-0.1.0+v7-source-hydrated-llm-candidate-safe-stack";
const SOURCE_HYDRATED_LLM_COVERAGE_AWARE_STACK_FRAMEWORK_VERSION =
  "workspace-0.1.0+v10-source-hydrated-llm-coverage-aware-stack";
const MULTI_QUERY_COVERAGE_AWARE_STACK_FRAMEWORK_VERSION =
  "workspace-0.1.0+v11b-multi-query-coverage-aware-stack";
const MULTI_QUERY_STANDARD_RERANK_STACK_FRAMEWORK_VERSION =
  "workspace-0.1.0+v11a-multi-query-standard-rerank-stack";
const MULTI_QUERY_PLAN_AWARE_COVERAGE_STACK_FRAMEWORK_VERSION =
  "workspace-0.1.0+v12-multi-query-plan-aware-coverage-stack";
const V13_STACK_DESCRIPTOR = {
  retrievalMode: "v13-anchored-evidence-answer-stack",
  frameworkVersion: "workspace-0.1.0+v13-anchored-evidence-answer-stack",
  answerPolicy: "supported-evidence-needs",
} as const satisfies {
  readonly retrievalMode: string;
  readonly frameworkVersion: string;
  readonly answerPolicy: AnswerPolicy;
};
const V15_STACK_DESCRIPTOR = {
  retrievalMode: "v15-corpus-complete-anchored-evidence-answer-stack",
  frameworkVersion: "workspace-0.1.0+v15-corpus-complete-anchored-evidence-answer-stack",
  answerPolicy: "supported-evidence-needs",
  corpusCoveragePolicy: "artifact-plus-canonical-gap-fill",
  corpusCoveragePolicyVersion: 1,
} as const satisfies {
  readonly retrievalMode: string;
  readonly frameworkVersion: string;
  readonly answerPolicy: AnswerPolicy;
  readonly corpusCoveragePolicy: string;
  readonly corpusCoveragePolicyVersion: number;
};
type DeterministicCoveragePolicy = "soft" | "quota";
interface CacheOnlyCoverageDescriptor {
  readonly retrievalMode: string;
  readonly frameworkVersion: string;
  readonly answerPolicy: AnswerPolicy;
  readonly selectionPolicy: DeterministicCoveragePolicy;
  readonly cacheOnly: true;
  readonly candidateCount: 50;
  readonly originalQueryWeight: 2;
  readonly expandedQueryWeight: 1;
  readonly reciprocalRankConstant: 10;
  readonly topWindow: 10;
  readonly originalQuota: number;
  readonly perExpansionQuota: number;
}
const V14A_STACK_DESCRIPTOR = {
  retrievalMode: "v14a-anchored-deterministic-soft-coverage-stack",
  frameworkVersion: "workspace-0.1.0+v14a-anchored-deterministic-soft-coverage-stack",
  answerPolicy: "supported-evidence-needs",
  selectionPolicy: "soft",
  cacheOnly: true,
  candidateCount: 50,
  originalQueryWeight: 2,
  expandedQueryWeight: 1,
  reciprocalRankConstant: 10,
  topWindow: 10,
  originalQuota: 0,
  perExpansionQuota: 0,
} as const satisfies CacheOnlyCoverageDescriptor;
const V14B_STACK_DESCRIPTOR = {
  ...V14A_STACK_DESCRIPTOR,
  retrievalMode: "v14b-anchored-deterministic-quota-coverage-stack",
  frameworkVersion: "workspace-0.1.0+v14b-anchored-deterministic-quota-coverage-stack",
  selectionPolicy: "quota",
  originalQuota: 5,
  perExpansionQuota: 1,
} as const satisfies CacheOnlyCoverageDescriptor;
const ADAPTIVE_EECE_STACK_FRAMEWORK_VERSION = "workspace-0.1.0+adaptive-eece-stack-v9";
const MAX_EXISTING_STACK_CANDIDATES = 20;
const MAX_EXISTING_STACK_FUSION = {
  vector: 2,
  graph: 2,
  bm25: 0.5,
  contextRerank: 0.5,
  reciprocalRankConstant: 10,
} as const;
const SOURCE_CONTEXT_POLICY: SourceContextPolicy = {
  windowCharacters: 5_000,
  maxContextCharacters: 36_000,
};
const RECALL_SAFE_SOURCE_CONTEXT_POLICY: SourceContextPolicy = {
  windowCharacters: 5_000,
  maxContextCharacters: 50_000,
};
const LLM_RERANK_CONCURRENCY = 20;
const MULTI_QUERY_POLICY_VERSION = "v11-search-perspectives-1";
const V13_CANDIDATE_COUNT = 50;
const V13_ORIGINAL_QUERY_WEIGHT = 2;
const EXPANDED_QUERY_WEIGHT = 1;
const BIDIRECTIONAL_ORGANIZATION_ID = "rag-eval";
const BIDIRECTIONAL_PRINCIPAL: Principal = {
  organizationId: BIDIRECTIONAL_ORGANIZATION_ID,
  subjectId: "benchmark-runner",
  groupIds: [],
};

class CorpusConnector implements MCPConnector {
  readonly name = "rag-eval-corpus";
  private readonly byId: ReadonlyMap<string, CorpusDocument>;
  private readonly searchable: readonly {
    readonly document: CorpusDocument;
    readonly lowerText: string;
  }[];

  constructor(private readonly documents: readonly CorpusDocument[]) {
    this.byId = new Map(documents.map((document) => [document.id, document]));
    this.searchable = documents.map((document) => ({
      document,
      lowerText: document.text.toLowerCase(),
    }));
  }

  async listResources(): Promise<MCPResource[]> {
    return this.documents.map((document) => ({
      id: document.id,
      name: document.title,
      description: summarizeForClassification(document.text),
      mimeType: "text/plain",
    }));
  }

  async fetchResource(resourceId: string): Promise<MCPData> {
    const document = this.byId.get(resourceId);
    if (!document) throw new Error(`Unknown corpus resource ${resourceId}`);
    return {
      resourceId,
      content: document.text,
      metadata: { sourceId: document.sourceId },
      fetchedAt: new Date(),
    };
  }

  async search(query: string): Promise<MCPData[]> {
    const tokens = query
      .toLowerCase()
      .split(/[^\p{L}\p{N}]+/u)
      .filter((token) => token.length >= 3);
    return this.searchable
      .map(({ document, lowerText }) => ({
        document,
        score: tokens.filter((token) => lowerText.includes(token)).length,
      }))
      .filter((item) => item.score > 0)
      .sort(
        (left, right) =>
          right.score - left.score || left.document.id.localeCompare(right.document.id),
      )
      .slice(0, 10)
      .map(({ document }) => ({
        resourceId: document.id,
        content: document.text,
        metadata: { sourceId: document.sourceId },
        fetchedAt: new Date(),
      }));
  }
}

class CodexLlmAdapter implements LLMAdapter {
  constructor(
    private readonly client: CodexJsonClient,
    private readonly manifest: RagEvalManifest,
  ) {}

  async complete(systemPrompt: string, context: string, query: string): Promise<string> {
    const response = await this.client.completeText(
      {
        model: this.manifest.models.answer.model,
        reasoningEffort: this.manifest.models.answer.reasoningEffort ?? "medium",
      },
      systemPrompt,
      context,
      query,
    );
    return response.value;
  }
}

export class KontextBrainAdapter implements FrameworkAdapter {
  readonly id = "kontext-brain" as const;
  private readonly codexClient: CodexJsonClient;
  private readonly embeddingClient: EmbeddingClient | null;
  private readonly retrievalMode: KontextRetrievalMode;
  private readonly benchmarkDataDirectory: string;
  private readonly precomputedIndexDirectory: string | null;

  constructor(
    private readonly manifest: RagEvalManifest,
    options: KontextBrainAdapterOptions = {},
  ) {
    this.codexClient = options.codexClient ?? new CodexJsonClient();
    this.embeddingClient = options.embeddingClient ?? null;
    this.retrievalMode = options.retrievalMode ?? "legacy";
    this.benchmarkDataDirectory =
      options.benchmarkDataDirectory ??
      resolve(dirname(fileURLToPath(import.meta.url)), "../../data");
    this.precomputedIndexDirectory = options.precomputedIndexDirectory ?? null;
  }

  async doctor(): Promise<FrameworkDoctorResult> {
    if (isCacheOnlyCoverageMode(this.retrievalMode)) {
      if (!validEmbeddingDimensions(this.manifest.models.embedding.dimensions)) {
        return {
          frameworkId: this.id,
          status: "blocked",
          version: frameworkVersion(this.retrievalMode),
          detail: "A positive embedding dimension is required to validate precomputed vectors",
        };
      }
      if (!this.precomputedIndexDirectory || !existsSync(this.precomputedIndexDirectory)) {
        return {
          frameworkId: this.id,
          status: "blocked",
          version: frameworkVersion(this.retrievalMode),
          detail: "KONTEXT_RAG_EVAL_PRECOMPUTED_INDEX must reference a readable v13 index",
        };
      }
      return {
        frameworkId: this.id,
        status: "ready",
        version: frameworkVersion(this.retrievalMode),
        detail: "read-only precomputed expansions and embeddings; deterministic coverage selection",
      };
    }
    if (this.retrievalMode !== "legacy") {
      if (!this.embeddingClient) {
        return {
          frameworkId: this.id,
          status: "blocked",
          version: frameworkVersion(this.retrievalMode),
          detail: "OPENAI_API_KEY is required for bidirectional KG chunk seeds",
        };
      }
    }
    const result = await runCommand("codex", ["--version"], "", 10_000).catch((error: Error) => ({
      exitCode: 1,
      stdout: "",
      stderr: error.message,
      durationMs: 0,
    }));
    return result.exitCode === 0
      ? {
          frameworkId: this.id,
          status: "ready",
          version: frameworkVersion(this.retrievalMode),
          detail:
            this.retrievalMode === "legacy"
              ? `autoSetup + DEFAULT_PIPELINE; ${result.stdout.trim()}`
              : this.retrievalMode === "max-existing-stack"
                ? `production bidirectional KG + OpenAI vector + BM25 + context rerank + RRF; ${result.stdout.trim()}`
                : this.retrievalMode === "source-hydrated-stack"
                  ? `v3 retrieval + source-native provenance hydration; ${result.stdout.trim()}`
                  : this.retrievalMode === "source-hydrated-llm-stack"
                    ? `over-retrieval + local GPT rerank + source-native provenance hydration; ${result.stdout.trim()}`
                    : this.retrievalMode === "source-hydrated-llm-recall-safe-stack"
                      ? `recall-safe local GPT rerank + source-native provenance hydration; ${result.stdout.trim()}`
                      : this.retrievalMode === "source-hydrated-llm-candidate-safe-stack"
                        ? `declared candidate-k + recall-safe local GPT rerank + source-native provenance hydration; ${result.stdout.trim()}`
                        : this.retrievalMode === "source-hydrated-llm-coverage-aware-stack"
                          ? `coverage-aware local GPT rerank + declared candidate-k + source-native provenance hydration; ${result.stdout.trim()}`
                          : this.retrievalMode === "multi-query-standard-rerank-stack"
                            ? `local GPT multi-query retrieval + standard rerank + source-native provenance hydration; ${result.stdout.trim()}`
                            : this.retrievalMode === "multi-query-coverage-aware-stack"
                              ? `local GPT multi-query retrieval + coverage-aware rerank + source-native provenance hydration; ${result.stdout.trim()}`
                              : this.retrievalMode === "multi-query-plan-aware-coverage-stack"
                                ? `local GPT multi-query retrieval + plan-aware coverage rerank + source-native provenance hydration; ${result.stdout.trim()}`
                                : this.retrievalMode ===
                                    "multi-query-anchored-evidence-answer-stack"
                                  ? `original-query-anchored multi-query retrieval + plan-aware coverage rerank + supported-needs answering; ${result.stdout.trim()}`
                                  : this.retrievalMode ===
                                      "corpus-complete-anchored-evidence-answer-stack"
                                    ? `v13 retrieval with artifact-independent canonical corpus gap filling; ${result.stdout.trim()}`
                                    : this.retrievalMode === "adaptive-eece-stack"
                                      ? `adaptive Entity–Event–Claim–Evidence KG + v7 retrieval stack; ${result.stdout.trim()}`
                                      : `production BidirectionalNLayerRetriever + evidence items + OpenAI vector seeds; ${result.stdout.trim()}`,
        }
      : {
          frameworkId: this.id,
          status: "blocked",
          version: "workspace-0.1.0",
          detail: result.stderr.trim(),
        };
  }

  async retrieve(bundle: DatasetBundle, options: FrameworkRunOptions): Promise<RetrievalResult[]> {
    if (bundle.track !== "static-kb") {
      return bundle.queries.map((query) => ({
        datasetId: bundle.id,
        frameworkId: this.id,
        queryId: query.id,
        status: "unsupported",
        evidence: [],
        latencyMs: 0,
        inputTokens: null,
        error: `The current kontext corpus adapter does not implement ${bundle.track}`,
        frameworkVersion: "workspace-0.1.0",
        configDigest: manifestDigest(this.manifest),
      }));
    }
    if (this.retrievalMode === "bidirectional-kg") {
      return this.retrieveBidirectionalKg(bundle, options);
    }
    if (this.retrievalMode === "max-existing-stack") {
      return this.retrieveMaxExistingStack(bundle, options);
    }
    if (this.retrievalMode === "source-hydrated-stack") {
      return this.retrieveMaxExistingStack(bundle, options, true);
    }
    if (this.retrievalMode === "source-hydrated-llm-stack") {
      return this.retrieveMaxExistingStack(bundle, options, true, true);
    }
    if (this.retrievalMode === "source-hydrated-llm-recall-safe-stack") {
      return this.retrieveMaxExistingStack(bundle, options, true, true, true);
    }
    if (this.retrievalMode === "source-hydrated-llm-candidate-safe-stack") {
      return this.retrieveMaxExistingStack(bundle, options, true, true, true, true);
    }
    if (this.retrievalMode === "source-hydrated-llm-coverage-aware-stack") {
      return this.retrieveMaxExistingStack(bundle, options, true, true, true, true, false, true);
    }
    if (this.retrievalMode === "multi-query-coverage-aware-stack") {
      return this.retrieveMaxExistingStack(
        bundle,
        options,
        true,
        true,
        true,
        true,
        false,
        true,
        true,
      );
    }
    if (this.retrievalMode === "multi-query-standard-rerank-stack") {
      return this.retrieveMaxExistingStack(
        bundle,
        options,
        true,
        true,
        true,
        true,
        false,
        false,
        true,
      );
    }
    if (this.retrievalMode === "multi-query-plan-aware-coverage-stack") {
      return this.retrieveMaxExistingStack(
        bundle,
        options,
        true,
        true,
        true,
        true,
        false,
        true,
        true,
        true,
      );
    }
    if (this.retrievalMode === "multi-query-anchored-evidence-answer-stack") {
      return this.retrieveMaxExistingStack(
        bundle,
        options,
        true,
        true,
        true,
        true,
        false,
        true,
        true,
        true,
        true,
      );
    }
    if (this.retrievalMode === "corpus-complete-anchored-evidence-answer-stack") {
      return this.retrieveMaxExistingStack(
        bundle,
        options,
        true,
        true,
        true,
        true,
        false,
        true,
        true,
        true,
        true,
        null,
        true,
      );
    }
    if (this.retrievalMode === "v14a-anchored-deterministic-soft-coverage-stack") {
      return this.retrieveMaxExistingStack(
        bundle,
        options,
        true,
        false,
        true,
        true,
        false,
        true,
        true,
        true,
        false,
        V14A_STACK_DESCRIPTOR,
      );
    }
    if (this.retrievalMode === "v14b-anchored-deterministic-quota-coverage-stack") {
      return this.retrieveMaxExistingStack(
        bundle,
        options,
        true,
        false,
        true,
        true,
        false,
        true,
        true,
        true,
        false,
        V14B_STACK_DESCRIPTOR,
      );
    }
    if (this.retrievalMode === "adaptive-eece-stack") {
      return this.retrieveMaxExistingStack(bundle, options, true, true, true, true, true);
    }
    const indexDirectory = join(options.workDirectory, bundle.id, this.id, "index");
    const stateId = `rag-eval-${documentDigest(bundle.documents).slice(0, 24)}`;
    const store = new FileOntologyStore(join(indexDirectory, "ontology"));
    const persisted = await store.load(stateId);
    const graph = new OntologyGraph(
      toOntologyNodes(persisted),
      toEdges(persisted),
      persisted.graphConfig
        ? toGraphConfig(persisted)
        : { maxDepth: 2, maxTokens: 2_000, strategy: TraversalStrategy.WEIGHTED_DFS },
    );
    const connector = new CorpusConnector(bundle.documents);
    const layerAdapter = new GenericMCPLayerAdapter(DataSource.CUSTOM, connector.name, connector);
    const fetcherRegistry = new ContentFetcherRegistry();
    fetcherRegistry.register(new MCPContentFetcherBridge(layerAdapter));
    const llm = new CodexLlmAdapter(this.codexClient, this.manifest);
    const vectorStore = new InMemoryVectorStore(async () => new Float32Array(0));
    const agent = new KontextAgent({
      ontologySchemaGraph: graph,
      router: new RouterLLMAdapter(llm, llm),
      mcpConnectors: [connector],
      mcpLayerAdapters: [layerAdapter],
      metaIndexStore: new InMemoryMetaIndexStore(),
      fetcherRegistry,
      vectorStore,
      mappingStrategy: new KeywordMappingStrategy(),
      metaSelector: new ScoreBasedSelector(),
      ingestPipeline: new IngestPipeline(llm, store, vectorStore),
      pipeline: DEFAULT_PIPELINE,
      legacySnapshotStore: store,
      stateId,
      mcpResourceCacheEntries: persisted.resources ?? [],
    });
    await agent.initialize();
    if (agent.ontologyGraph.nodes.size === 0) await agent.autoSetup();

    const digest = manifestDigest(this.manifest);
    const checkpointDirectory = join(indexDirectory, "retrieval-checkpoints", stateId);
    return mapWithConcurrency(bundle.queries, 1, async (query) => {
      const checkpointPath = join(
        checkpointDirectory,
        `${createHash("sha256").update(query.id).update("\0").update(query.text).digest("hex")}.json`,
      );
      if (existsSync(checkpointPath)) {
        try {
          const checkpoint = JSON.parse(readFileSync(checkpointPath, "utf8")) as RetrievalResult;
          if (
            checkpoint.datasetId === bundle.id &&
            checkpoint.frameworkId === this.id &&
            checkpoint.queryId === query.id &&
            checkpoint.configDigest === digest
          ) {
            return checkpoint;
          }
        } catch {
          // An invalid or interrupted checkpoint is recomputed below.
        }
      }
      const startedAt = performance.now();
      try {
        const retrieval = await agent.retrieve(query.text);
        const selectedSourceIds = [
          ...new Set(retrieval.selectedMetaDocs.map((document) => document.id)),
        ];
        const evidence = retrieval.context.trim()
          ? [
              {
                id: `${this.id}:${query.id}:context`,
                sourceId: selectedSourceIds.join(",") || "kontext-context",
                text: retrieval.context,
                score: 1,
                rank: 1,
                metadata: {
                  ontologyNodes: retrieval.usedOntologyNodes.map((node) => node.id).join(","),
                  selectedResources: selectedSourceIds.join(","),
                  contextTokens: retrieval.contextTokensUsed,
                },
              },
            ]
          : [];
        const result: RetrievalResult = {
          datasetId: bundle.id,
          frameworkId: this.id,
          queryId: query.id,
          status: "ok",
          evidence,
          latencyMs: performance.now() - startedAt,
          inputTokens: retrieval.contextTokensUsed,
          error: null,
          frameworkVersion: "workspace-0.1.0",
          configDigest: digest,
        };
        writeJsonAtomic(checkpointPath, result);
        return result;
      } catch (error) {
        const result: RetrievalResult = {
          datasetId: bundle.id,
          frameworkId: this.id,
          queryId: query.id,
          status: "error",
          evidence: [],
          latencyMs: performance.now() - startedAt,
          inputTokens: null,
          error: (error as Error).message,
          frameworkVersion: "workspace-0.1.0",
          configDigest: digest,
        };
        writeJsonAtomic(checkpointPath, result);
        return result;
      }
    });
  }

  private async retrieveBidirectionalKg(
    bundle: DatasetBundle,
    options: FrameworkRunOptions,
  ): Promise<RetrievalResult[]> {
    const embeddingClient = this.embeddingClient;
    if (!embeddingClient) throw new Error("Bidirectional KG retrieval requires OPENAI_API_KEY");
    const domain = graphRagDomain(bundle.id);
    if (!domain) {
      return bundle.queries.map((query) => ({
        datasetId: bundle.id,
        frameworkId: this.id,
        queryId: query.id,
        status: "unsupported",
        evidence: [],
        latencyMs: 0,
        inputTokens: null,
        error: `No evidence-backed KG artifact is registered for ${bundle.id}`,
        frameworkVersion: BIDIRECTIONAL_FRAMEWORK_VERSION,
        configDigest: manifestDigest(this.manifest),
      }));
    }

    const artifactPaths = {
      chunks: join(this.benchmarkDataDirectory, `gb-${domain}-chunks.jsonl`),
      graph: join(this.benchmarkDataDirectory, `gb-${domain}-kg.json`),
    };
    const docs = readJsonLines<{ readonly id: string; readonly body: string }>(
      artifactPaths.chunks,
    ).map<BenchDoc>((doc) => ({ ...doc, title: chunkTitle(doc.id) }));
    const graph = readKnowledgeGraph(artifactPaths.graph);
    const indexDirectory = join(
      options.workDirectory,
      bundle.id,
      this.id,
      "index",
      "bidirectional-kg",
    );
    const indexDigest = kgDocumentDigest(docs);
    const documentEmbeddings = await embedWithCheckpoints(
      embeddingClient,
      docs.map((doc) => ({ id: doc.id, title: doc.title, text: doc.body })),
      "RETRIEVAL_DOCUMENT",
      join(indexDirectory, "document-embedding-batches"),
      indexDigest,
    );
    const queryDigest = kgQueryDigest(bundle, indexDigest);
    const queryEmbeddings = await embedWithCheckpoints(
      embeddingClient,
      bundle.queries.map((query) => ({ id: query.id, text: query.text })),
      "RETRIEVAL_QUERY",
      join(indexDirectory, "query-embedding-batches"),
      queryDigest,
    );
    const queryVectors = splitVectors(
      queryEmbeddings.vectors,
      bundle.queries.map((query) => query.id),
      embeddingClient.dimensions,
    );
    const vectorSeedCount = Math.min(10, docs.length);
    const lexicalSeedCount = Math.min(5, docs.length);
    const seeds = bundle.queries.map((query) => {
      const queryVector = queryVectors.get(query.id);
      if (!queryVector) throw new Error(`Missing KG query embedding ${query.id}`);
      return {
        question: query.text,
        chunkIds: rankKgVectors(
          documentEmbeddings.vectors,
          queryVector,
          embeddingClient.dimensions,
          vectorSeedCount,
        ).flatMap((ranked) => {
          const doc = docs[ranked.index];
          return doc ? [doc.id] : [];
        }),
      };
    });
    const graphFanout = {
      seedChunks: vectorSeedCount,
      lexicalSeedChunks: lexicalSeedCount,
      queryAware: true,
      resourceChunks: 10,
      entityChunks: 10,
      entityFacts: 10,
      chunkEntities: 10,
      chunkFacts: 10,
    } as const;
    const searchGraph = new BidirectionalBenchmarkSearchGraph(graph, docs, seeds, graphFanout);
    const knowledgeRetriever = new BidirectionalNLayerRetriever(searchGraph);
    const agent = await this.createBidirectionalAgent(bundle, indexDirectory, knowledgeRetriever);
    const digest = manifestDigest(this.manifest);
    const checkpointDirectory = join(indexDirectory, "retrieval-checkpoints", queryDigest);
    writeJsonAtomic(join(indexDirectory, "kontext-kg-config.json"), {
      retrievalMode: "bidirectional",
      frameworkVersion: BIDIRECTIONAL_FRAMEWORK_VERSION,
      graphProjection: "GraphRAG-Bench KG -> production SearchGraphPort",
      agentEntryPoint: "KontextAgent.retrieve(question, principal)",
      embedding: {
        provider: "openai",
        model: embeddingClient.model,
        dimensions: embeddingClient.dimensions,
        vectorSeedCount,
        lexicalSeedCount,
      },
      graphFanout,
      searchBudget: {
        topK: options.topK,
        maxHops: 8,
        maxKgHops: 3,
        maxVisited: 200,
        maxCandidates: 500,
        timeBudgetMs: 1200,
        minScore: 0.02,
      },
      graph: { chunks: docs.length, entities: graph.entities.size, facts: graph.edges.length },
    });
    const totalEmbeddingInputTokens = documentEmbeddings.inputTokens + queryEmbeddings.inputTokens;
    writeJsonAtomic(join(indexDirectory, "embedding-usage.json"), {
      provider: "openai",
      model: embeddingClient.model,
      dimensions: embeddingClient.dimensions,
      indexInputTokens: documentEmbeddings.inputTokens,
      queryInputTokens: queryEmbeddings.inputTokens,
      totalInputTokens: totalEmbeddingInputTokens,
      inputPriceUsdPerMillionTokens: 0.02,
      estimatedCostUsd: (totalEmbeddingInputTokens * 0.02) / 1_000_000,
    });

    return mapWithConcurrency(bundle.queries, 1, async (query) => {
      const checkpointPath = join(
        checkpointDirectory,
        `${createHash("sha256").update(query.id).update("\0").update(query.text).digest("hex")}.json`,
      );
      if (existsSync(checkpointPath)) {
        try {
          const checkpoint = JSON.parse(readFileSync(checkpointPath, "utf8")) as RetrievalResult;
          if (
            checkpoint.datasetId === bundle.id &&
            checkpoint.frameworkId === this.id &&
            checkpoint.queryId === query.id &&
            checkpoint.configDigest === digest &&
            checkpoint.frameworkVersion === BIDIRECTIONAL_FRAMEWORK_VERSION
          ) {
            return checkpoint;
          }
        } catch {
          // An invalid or interrupted checkpoint is recomputed below.
        }
      }
      const startedAt = performance.now();
      try {
        const retrieval = await agent.retrieve(query.text, BIDIRECTIONAL_PRINCIPAL);
        if (retrieval.retrievalMode !== "bidirectional") {
          throw new Error(`Expected bidirectional retrieval, received ${retrieval.retrievalMode}`);
        }
        const trace = retrieval.searchTrace;
        const evidence = (retrieval.evidence ?? []).slice(0, options.topK).map((hit, rank) => ({
          id: hit.evidenceId,
          sourceId: hit.resourceId,
          text: hit.text,
          score: hit.score,
          rank: rank + 1,
          metadata: {
            retrievalMode: "bidirectional",
            chunkId: hit.chunkId,
            factKey: hit.factKey ?? null,
            factStatus: hit.factStatus ?? null,
            path: hit.path
              .map((edge) => `${edge.from.kind}:${edge.from.id}->${edge.to.kind}:${edge.to.id}`)
              .join(" | "),
            visited: trace?.visited ?? null,
            candidates: trace?.candidates ?? null,
            stoppedBy: trace?.stoppedBy ?? null,
          },
        }));
        const result: RetrievalResult = {
          datasetId: bundle.id,
          frameworkId: this.id,
          queryId: query.id,
          status: "ok",
          evidence,
          latencyMs: performance.now() - startedAt,
          inputTokens: retrieval.contextTokensUsed,
          error: null,
          frameworkVersion: BIDIRECTIONAL_FRAMEWORK_VERSION,
          configDigest: digest,
        };
        writeJsonAtomic(checkpointPath, result);
        return result;
      } catch (error) {
        const result: RetrievalResult = {
          datasetId: bundle.id,
          frameworkId: this.id,
          queryId: query.id,
          status: "error",
          evidence: [],
          latencyMs: performance.now() - startedAt,
          inputTokens: null,
          error: (error as Error).message,
          frameworkVersion: BIDIRECTIONAL_FRAMEWORK_VERSION,
          configDigest: digest,
        };
        writeJsonAtomic(checkpointPath, result);
        return result;
      }
    });
  }

  private async retrieveMaxExistingStack(
    bundle: DatasetBundle,
    options: FrameworkRunOptions,
    hydrateSourceContext = false,
    rerankWithLlm = false,
    recallSafeLlmRerank = false,
    honorDeclaredCandidateK = false,
    adaptiveEece = false,
    coverageAwareRerank = false,
    multiQuery = false,
    planAwareCoverage = false,
    anchoredEvidenceAnswer = false,
    deterministicCoverage: CacheOnlyCoverageDescriptor | null = null,
    completeCorpusCoverage = false,
  ): Promise<RetrievalResult[]> {
    const embeddingClient =
      this.embeddingClient ??
      (deterministicCoverage ? cacheOnlyEmbeddingClient(this.manifest) : null);
    if (!embeddingClient) throw new Error("Max existing stack retrieval requires OPENAI_API_KEY");
    const candidateCount = deterministicCoverage
      ? deterministicCoverage.candidateCount
      : anchoredEvidenceAnswer
        ? V13_CANDIDATE_COUNT
        : honorDeclaredCandidateK
          ? options.candidateK
          : MAX_EXISTING_STACK_CANDIDATES;
    const inheritedRetrievalMode = planAwareCoverage
      ? "v12-multi-query-plan-aware-coverage-stack"
      : multiQuery
        ? coverageAwareRerank
          ? "v11b-multi-query-coverage-aware-stack"
          : "v11a-multi-query-standard-rerank-stack"
        : coverageAwareRerank
          ? "v10-source-hydrated-llm-coverage-aware-stack"
          : adaptiveEece
            ? "adaptive-eece-stack-v9"
            : honorDeclaredCandidateK
              ? "v7-source-hydrated-llm-candidate-safe-stack"
              : recallSafeLlmRerank
                ? "v6-source-hydrated-llm-recall-safe-stack"
                : rerankWithLlm
                  ? "v5-source-hydrated-llm-stack"
                  : hydrateSourceContext
                    ? "v4-source-hydrated-stack"
                    : "v3-max-existing-stack";
    const retrievalMode =
      deterministicCoverage?.retrievalMode ??
      (completeCorpusCoverage
        ? V15_STACK_DESCRIPTOR.retrievalMode
        : anchoredEvidenceAnswer
          ? V13_STACK_DESCRIPTOR.retrievalMode
          : inheritedRetrievalMode);
    const inheritedFrameworkVersion = planAwareCoverage
      ? MULTI_QUERY_PLAN_AWARE_COVERAGE_STACK_FRAMEWORK_VERSION
      : multiQuery
        ? coverageAwareRerank
          ? MULTI_QUERY_COVERAGE_AWARE_STACK_FRAMEWORK_VERSION
          : MULTI_QUERY_STANDARD_RERANK_STACK_FRAMEWORK_VERSION
        : coverageAwareRerank
          ? SOURCE_HYDRATED_LLM_COVERAGE_AWARE_STACK_FRAMEWORK_VERSION
          : adaptiveEece
            ? ADAPTIVE_EECE_STACK_FRAMEWORK_VERSION
            : honorDeclaredCandidateK
              ? SOURCE_HYDRATED_LLM_CANDIDATE_SAFE_STACK_FRAMEWORK_VERSION
              : recallSafeLlmRerank
                ? SOURCE_HYDRATED_LLM_RECALL_SAFE_STACK_FRAMEWORK_VERSION
                : rerankWithLlm
                  ? SOURCE_HYDRATED_LLM_STACK_FRAMEWORK_VERSION
                  : hydrateSourceContext
                    ? SOURCE_HYDRATED_STACK_FRAMEWORK_VERSION
                    : MAX_EXISTING_STACK_FRAMEWORK_VERSION;
    const frameworkVersion =
      deterministicCoverage?.frameworkVersion ??
      (completeCorpusCoverage
        ? V15_STACK_DESCRIPTOR.frameworkVersion
        : anchoredEvidenceAnswer
          ? V13_STACK_DESCRIPTOR.frameworkVersion
          : inheritedFrameworkVersion);
    const answerPolicy =
      deterministicCoverage?.answerPolicy ??
      (completeCorpusCoverage
        ? V15_STACK_DESCRIPTOR.answerPolicy
        : anchoredEvidenceAnswer
          ? V13_STACK_DESCRIPTOR.answerPolicy
          : undefined);
    const originalQueryWeight =
      deterministicCoverage?.originalQueryWeight ??
      (anchoredEvidenceAnswer ? V13_ORIGINAL_QUERY_WEIGHT : 1);
    const expandedQueryWeight = deterministicCoverage?.expandedQueryWeight ?? EXPANDED_QUERY_WEIGHT;
    const perspectiveReciprocalRankConstant =
      deterministicCoverage?.reciprocalRankConstant ??
      MAX_EXISTING_STACK_FUSION.reciprocalRankConstant;
    const domain = graphRagDomain(bundle.id);
    if (!domain && bundle.track !== "static-kb") {
      return bundle.queries.map((query) => ({
        datasetId: bundle.id,
        frameworkId: this.id,
        queryId: query.id,
        status: "unsupported",
        evidence: [],
        latencyMs: 0,
        inputTokens: null,
        error: `Max-stack retrieval requires a static KB or registered KG artifact for ${bundle.id}`,
        frameworkVersion,
        answerPolicy,
        configDigest: maxExistingStackDigest(
          this.manifest,
          hydrateSourceContext,
          rerankWithLlm,
          recallSafeLlmRerank,
          honorDeclaredCandidateK,
          adaptiveEece,
          coverageAwareRerank,
          multiQuery,
          planAwareCoverage,
          anchoredEvidenceAnswer,
          candidateCount,
          deterministicCoverage,
          completeCorpusCoverage,
        ),
      }));
    }

    const artifactDocs = domain
      ? readJsonLines<{ readonly id: string; readonly body: string }>(
          join(this.benchmarkDataDirectory, `gb-${domain}-chunks.jsonl`),
        ).map<BenchDoc>((doc) => ({ ...doc, title: chunkTitle(doc.id) }))
      : chunkCanonicalDocuments(bundle.documents);
    const corpusCoverage = completeCorpusCoverage
      ? completeCorpusFromArtifact(artifactDocs, bundle.documents)
      : null;
    const docs = corpusCoverage?.docs ?? artifactDocs;
    const docsById = new Map(docs.map((doc) => [doc.id, doc]));
    const graph = domain
      ? readKnowledgeGraph(join(this.benchmarkDataDirectory, `gb-${domain}-kg.json`))
      : emptyKnowledgeGraph(docs);
    const indexDirectory = join(options.workDirectory, bundle.id, this.id, "index", retrievalMode);
    const cacheOnly = deterministicCoverage?.cacheOnly ?? false;
    const cacheIndexDirectory = cacheOnly
      ? requiredPrecomputedIndexDirectory(this.precomputedIndexDirectory)
      : indexDirectory;
    const indexDigest = kgDocumentDigest(docs);
    const documentEmbeddings = await embedWithCheckpoints(
      embeddingClient,
      docs.map((doc) => ({ id: doc.id, title: doc.title, text: doc.body })),
      "RETRIEVAL_DOCUMENT",
      join(cacheIndexDirectory, "document-embedding-batches"),
      indexDigest,
      cacheOnly,
    );
    const baseQueryDigest = kgQueryDigest(bundle, indexDigest);
    const queryEmbeddings = await embedWithCheckpoints(
      embeddingClient,
      bundle.queries.map((query) => ({ id: query.id, text: query.text })),
      "RETRIEVAL_QUERY",
      join(cacheIndexDirectory, "query-embedding-batches"),
      baseQueryDigest,
      cacheOnly,
    );
    const queryVectors = splitVectors(
      queryEmbeddings.vectors,
      bundle.queries.map((query) => query.id),
      embeddingClient.dimensions,
    );
    const multiQueryExpander =
      multiQuery && !cacheOnly
        ? new MultiQueryExpander(this.codexClient, {
            model: this.manifest.models.answer.model,
            reasoningEffort: this.manifest.models.answer.reasoningEffort ?? "medium",
          })
        : null;
    const queryExpansions = multiQuery
      ? await mapWithConcurrency(bundle.queries, LLM_RERANK_CONCURRENCY, (query) =>
          expandWithCheckpoint(
            multiQueryExpander,
            query.id,
            query.text,
            join(cacheIndexDirectory, "multi-query-expansions"),
            cacheOnly,
          ),
        )
      : bundle.queries.map((query) => emptyMultiQueryCheckpoint(query.id, query.text));
    const expansionByQueryId = new Map(
      queryExpansions.map((expansion) => [expansion.queryId, expansion]),
    );
    const expandedQueryInputs = queryExpansions.flatMap((expansion) =>
      expansion.queries.map((text, index) => ({
        id: multiQueryEmbeddingId(expansion.queryId, index),
        text,
      })),
    );
    const expandedQueryDigest = multiQueryQueryDigest(bundle, indexDigest, queryExpansions);
    const expandedQueryEmbeddings = await embedWithCheckpoints(
      embeddingClient,
      expandedQueryInputs,
      "RETRIEVAL_QUERY",
      join(cacheIndexDirectory, "multi-query-embedding-batches"),
      expandedQueryDigest,
      cacheOnly,
    );
    const expandedQueryVectors = splitVectors(
      expandedQueryEmbeddings.vectors,
      expandedQueryInputs.map((input) => input.id),
      embeddingClient.dimensions,
    );
    const vectorPerspectivesByQuery = new Map(
      bundle.queries.map((query) => {
        const queryVector = queryVectors.get(query.id);
        if (!queryVector) throw new Error(`Missing KG query embedding ${query.id}`);
        const originalIds = rankKgVectors(
          documentEmbeddings.vectors,
          queryVector,
          embeddingClient.dimensions,
          candidateCount,
        ).flatMap((ranked) => {
          const doc = docs[ranked.index];
          return doc ? [doc.id] : [];
        });
        const expansion = expansionByQueryId.get(query.id);
        const perspectiveIds = (expansion?.queries ?? []).map((_text, index) => {
          const vector = expandedQueryVectors.get(multiQueryEmbeddingId(query.id, index));
          if (!vector) throw new Error(`Missing expanded query embedding ${query.id}:${index}`);
          return rankKgVectors(
            documentEmbeddings.vectors,
            vector,
            embeddingClient.dimensions,
            candidateCount,
          ).flatMap((ranked) => {
            const doc = docs[ranked.index];
            return doc ? [doc.id] : [];
          });
        });
        const fusedIds = multiQuery
          ? fuseQueryPerspectives(originalIds, perspectiveIds, {
              limit: candidateCount,
              originalQueryWeight,
              expandedQueryWeight,
              reciprocalRankConstant: perspectiveReciprocalRankConstant,
            }).map((candidate) => candidate.id)
          : originalIds;
        return [query.id, { originalIds, perspectiveIds, fusedIds }] as const;
      }),
    );
    const vectorIdsByQuery = new Map(
      Array.from(vectorPerspectivesByQuery, ([queryId, perspectives]) => [
        queryId,
        perspectives.fusedIds,
      ]),
    );
    const seeds = bundle.queries.map((query) => ({
      question: query.text,
      chunkIds: vectorIdsByQuery.get(query.id) ?? [],
    }));
    const graphFanout = {
      seedChunks: 10,
      lexicalSeedChunks: 5,
      queryAware: true,
      resourceChunks: 10,
      entityChunks: 10,
      entityFacts: 10,
      chunkEntities: 10,
      chunkFacts: 10,
    } as const;
    const graphBudget = {
      topK: candidateCount,
      maxHops: 8,
      maxKgHops: 3,
      maxVisited: 400,
      maxCandidates: 1_000,
      timeBudgetMs: 1_200,
      minScore: 0.02,
    } as const;
    const searchGraph = new BidirectionalBenchmarkSearchGraph(graph, docs, seeds, graphFanout);
    const knowledgeRetriever = new BudgetedBidirectionalRetriever(searchGraph, graphBudget);
    const agent = await this.createBidirectionalAgent(bundle, indexDirectory, knowledgeRetriever);
    const bm25Ranker = new CorpusBm25Ranker(
      docs.map((doc) => ({ id: doc.id, text: `${doc.title} ${doc.body}` })),
    );
    const sourceHydrator = hydrateSourceContext
      ? new SourceContextHydrator(
          docs.map(toSourceChunk),
          recallSafeLlmRerank ? RECALL_SAFE_SOURCE_CONTEXT_POLICY : SOURCE_CONTEXT_POLICY,
        )
      : null;
    const llmReranker = rerankWithLlm
      ? new LlmEvidenceReranker(
          this.codexClient,
          {
            model: this.manifest.models.answer.model,
            reasoningEffort: this.manifest.models.answer.reasoningEffort ?? "medium",
          },
          { coverageAware: coverageAwareRerank },
        )
      : null;
    const digest = maxExistingStackDigest(
      this.manifest,
      hydrateSourceContext,
      rerankWithLlm,
      recallSafeLlmRerank,
      honorDeclaredCandidateK,
      adaptiveEece,
      coverageAwareRerank,
      multiQuery,
      planAwareCoverage,
      anchoredEvidenceAnswer,
      candidateCount,
      deterministicCoverage,
      completeCorpusCoverage,
    );
    const retrievalQueryDigest = multiQuery ? expandedQueryDigest : baseQueryDigest;
    const checkpointDirectory = join(indexDirectory, "retrieval-checkpoints", retrievalQueryDigest);
    writeJsonAtomic(join(indexDirectory, "kontext-kg-config.json"), {
      retrievalMode,
      frameworkVersion,
      answerPolicy,
      components: [
        ...(adaptiveEece ? ["adaptive Entity–Event–Claim–Evidence KG enrichment"] : []),
        ...(multiQuery && !cacheOnly ? ["local GPT multi-query expansion"] : []),
        ...(cacheOnly ? ["precomputed multi-query perspectives"] : []),
        cacheOnly
          ? "precomputed OpenAI cosine vector candidates"
          : "OpenAI cosine vector candidates",
        "production BidirectionalNLayerRetriever",
        "query-aware evidence-backed KG traversal",
        "corpus BM25 candidates",
        "graph/lexical context rerank",
        "weighted reciprocal-rank fusion",
      ],
      graphProjection: adaptiveEece
        ? "source chunks -> AdaptiveKnowledgeEnricher -> augmented production SearchGraphPort"
        : domain
          ? "GraphRAG-Bench KG -> production SearchGraphPort"
          : "canonical static corpus -> source/chunk production SearchGraphPort",
      agentEntryPoint: "KontextAgent.retrieve(question, principal)",
      embedding: {
        provider: "openai",
        model: embeddingClient.model,
        dimensions: embeddingClient.dimensions,
        cacheOnly,
        vectorSeedCount: graphFanout.seedChunks,
        vectorCandidateCount: candidateCount,
        lexicalSeedCount: graphFanout.lexicalSeedChunks,
      },
      graphFanout,
      graphBudget,
      fusion: MAX_EXISTING_STACK_FUSION,
      multiQuery: multiQuery
        ? {
            policyVersion: MULTI_QUERY_POLICY_VERSION,
            execution: cacheOnly ? "precomputed-cache" : "codex-cli",
            ...(cacheOnly
              ? {}
              : {
                  model: this.manifest.models.answer.model,
                  reasoningEffort: this.manifest.models.answer.reasoningEffort,
                }),
            originalQueryPreserved: true,
            maximumExpandedQueries: 3,
            perspectiveFusion: {
              method: "reciprocal-rank-fusion",
              reciprocalRankConstant: perspectiveReciprocalRankConstant,
              ...(anchoredEvidenceAnswer || deterministicCoverage
                ? {
                    originalQueryWeight,
                    expandedQueryWeight,
                  }
                : { equalWeight: true }),
            },
            expandedQueries: expandedQueryInputs.length,
            failedExpansions: queryExpansions.filter((expansion) => expansion.error !== null)
              .length,
            ...(cacheOnly
              ? {
                  cachedOriginalQueryOnlyFallbacks: queryExpansions.filter(
                    (expansion) => expansion.error !== null && expansion.queries.length === 0,
                  ).length,
                }
              : {}),
            goldAccess: false,
          }
        : null,
      deterministicCoverage: deterministicCoverage
        ? {
            policy: deterministicCoverage.selectionPolicy,
            topWindow: deterministicCoverage.topWindow,
            originalQuota: deterministicCoverage.originalQuota,
            perExpansionQuota: deterministicCoverage.perExpansionQuota,
            goldAccess: false,
          }
        : null,
      corpusCoverage: corpusCoverage
        ? {
            policy: V15_STACK_DESCRIPTOR.corpusCoveragePolicy,
            policyVersion: V15_STACK_DESCRIPTOR.corpusCoveragePolicyVersion,
            originalSourceCount: corpusCoverage.originalSourceCount,
            artifactCoveredSourceCount: corpusCoverage.artifactCoveredSourceCount,
            supplementedSourceCount: corpusCoverage.supplementedSourceIds.length,
            supplementedSourceIds: corpusCoverage.supplementedSourceIds,
            supplementedSourceDigest: corpusCoverage.supplementedSourceDigest,
            queryAware: false,
            goldAccess: false,
          }
        : null,
      cacheReuse: cacheOnly
        ? {
            readOnly: true,
            newCodexCalls: 0,
            newEmbeddingCalls: 0,
            newInputTokens: 0,
          }
        : null,
      sourceHydration: hydrateSourceContext
        ? recallSafeLlmRerank
          ? RECALL_SAFE_SOURCE_CONTEXT_POLICY
          : SOURCE_CONTEXT_POLICY
        : null,
      llmReranker: rerankWithLlm
        ? {
            execution: "codex-cli",
            model: this.manifest.models.answer.model,
            reasoningEffort: this.manifest.models.answer.reasoningEffort,
            candidateCount,
            outputCount: recallSafeLlmRerank ? candidateCount : options.topK,
            recallSafe: recallSafeLlmRerank,
            coverageAware: coverageAwareRerank,
            queryPlanAware: planAwareCoverage,
            goldAccess: false,
            concurrency: LLM_RERANK_CONCURRENCY,
          }
        : null,
      outputTopK: options.topK,
      graph: { chunks: docs.length, entities: graph.entities.size, facts: graph.edges.length },
    });
    const totalEmbeddingInputTokens =
      documentEmbeddings.inputTokens +
      queryEmbeddings.inputTokens +
      expandedQueryEmbeddings.inputTokens;
    if (!cacheOnly) {
      writeJsonAtomic(join(indexDirectory, "embedding-usage.json"), {
        provider: "openai",
        model: embeddingClient.model,
        dimensions: embeddingClient.dimensions,
        indexInputTokens: documentEmbeddings.inputTokens,
        queryInputTokens: queryEmbeddings.inputTokens,
        expandedQueryInputTokens: expandedQueryEmbeddings.inputTokens,
        totalInputTokens: totalEmbeddingInputTokens,
        inputPriceUsdPerMillionTokens: 0.02,
        estimatedCostUsd: (totalEmbeddingInputTokens * 0.02) / 1_000_000,
      });
    }

    return mapWithConcurrency(
      bundle.queries,
      rerankWithLlm ? LLM_RERANK_CONCURRENCY : 1,
      async (query) => {
        const checkpointPath = join(
          checkpointDirectory,
          `${createHash("sha256").update(query.id).update("\0").update(query.text).digest("hex")}.json`,
        );
        if (existsSync(checkpointPath)) {
          try {
            const checkpoint = JSON.parse(readFileSync(checkpointPath, "utf8")) as RetrievalResult;
            if (
              checkpoint.datasetId === bundle.id &&
              checkpoint.frameworkId === this.id &&
              checkpoint.queryId === query.id &&
              checkpoint.status === "ok" &&
              checkpoint.configDigest === digest &&
              checkpoint.frameworkVersion === frameworkVersion &&
              checkpoint.answerPolicy === answerPolicy
            ) {
              return checkpoint;
            }
          } catch {
            // An invalid or interrupted checkpoint is recomputed below.
          }
        }
        const startedAt = performance.now();
        try {
          const retrieval = await agent.retrieve(query.text, BIDIRECTIONAL_PRINCIPAL);
          if (retrieval.retrievalMode !== "bidirectional") {
            throw new Error(
              `Expected bidirectional retrieval, received ${retrieval.retrievalMode}`,
            );
          }
          const trace = retrieval.searchTrace;
          const graphHits = retrieval.evidence ?? [];
          const graphByChunk = new Map(graphHits.map((hit) => [hit.chunkId, hit]));
          const graphIds = graphHits.map((hit) => hit.chunkId);
          const vectorIds = vectorIdsByQuery.get(query.id) ?? [];
          const expansion = expansionByQueryId.get(query.id);
          const bm25Perspectives = [
            query.text,
            ...(multiQuery ? (expansion?.queries ?? []) : []),
          ].map((perspective) => bm25Ranker.rank(perspective, candidateCount));
          const bm25Ids = multiQuery
            ? fuseQueryPerspectives(bm25Perspectives[0] ?? [], bm25Perspectives.slice(1), {
                limit: candidateCount,
                originalQueryWeight,
                expandedQueryWeight,
                reciprocalRankConstant: perspectiveReciprocalRankConstant,
              }).map((candidate) => candidate.id)
            : (bm25Perspectives[0] ?? []);
          const candidateIds = Array.from(new Set([...graphIds, ...vectorIds, ...bm25Ids]));
          const contextRerankedIds = searchGraph.rankContextChunkIds(query.text, candidateIds);
          const fusedCandidates = fuseRankings(
            [
              { name: "vector", ids: vectorIds, weight: MAX_EXISTING_STACK_FUSION.vector },
              { name: "graph", ids: graphIds, weight: MAX_EXISTING_STACK_FUSION.graph },
              { name: "bm25", ids: bm25Ids, weight: MAX_EXISTING_STACK_FUSION.bm25 },
              {
                name: "context-rerank",
                ids: contextRerankedIds,
                weight: MAX_EXISTING_STACK_FUSION.contextRerank,
              },
            ],
            rerankWithLlm || deterministicCoverage ? candidateCount : options.topK,
            MAX_EXISTING_STACK_FUSION.reciprocalRankConstant,
          );
          const vectorPerspectives = vectorPerspectivesByQuery.get(query.id);
          const deterministicallySelected =
            deterministicCoverage?.selectionPolicy === "quota" && vectorPerspectives
              ? applyOriginalAndExpansionQuota(
                  fusedCandidates,
                  fuseVectorAndBm25Perspective(
                    vectorPerspectives.originalIds,
                    bm25Perspectives[0] ?? [],
                    candidateCount,
                    deterministicCoverage.reciprocalRankConstant,
                  ),
                  vectorPerspectives.perspectiveIds.map((vectorIds, index) =>
                    fuseVectorAndBm25Perspective(
                      vectorIds,
                      bm25Perspectives[index + 1] ?? [],
                      candidateCount,
                      deterministicCoverage.reciprocalRankConstant,
                    ),
                  ),
                  {
                    topWindow: deterministicCoverage.topWindow,
                    originalQuota: deterministicCoverage.originalQuota,
                    perExpansionQuota: deterministicCoverage.perExpansionQuota,
                  },
                )
              : fusedCandidates;
          const fused = llmReranker
            ? await llmReranker.rerank(
                query.text,
                fusedCandidates.flatMap((candidate) => {
                  const doc = docsById.get(candidate.id);
                  return doc ? [{ ...candidate, text: doc.body }] : [];
                }),
                recallSafeLlmRerank ? candidateCount : options.topK,
                planAwareCoverage ? (expansion?.queries ?? []) : [],
              )
            : deterministicCoverage
              ? deterministicallySelected.slice(0, options.topK)
              : deterministicallySelected;
          const chunkEvidence = fused.flatMap((candidate, rank) => {
            const graphHit = graphByChunk.get(candidate.id);
            const doc = docsById.get(candidate.id);
            if (!graphHit && !doc) return [];
            return [
              {
                id: graphHit?.evidenceId ?? `chunk:${candidate.id}`,
                sourceId: graphHit?.resourceId ?? (doc ? sourceDocumentId(doc) : candidate.id),
                text: graphHit?.text ?? doc?.body ?? "",
                score: candidate.score,
                rank: rank + 1,
                metadata: {
                  retrievalMode: "v3-max-existing-stack",
                  chunkId: candidate.id,
                  factKey: graphHit?.factKey ?? null,
                  factStatus: graphHit?.factStatus ?? null,
                  path:
                    graphHit?.path
                      .map(
                        (edge) =>
                          `${edge.from.kind}:${edge.from.id}->${edge.to.kind}:${edge.to.id}`,
                      )
                      .join(" | ") ?? "",
                  fusionScore: candidate.score,
                  vectorRank: candidate.sourceRanks.vector ?? null,
                  graphRank: candidate.sourceRanks.graph ?? null,
                  bm25Rank: candidate.sourceRanks.bm25 ?? null,
                  contextRerank: candidate.sourceRanks["context-rerank"] ?? null,
                  visited: trace?.visited ?? null,
                  candidates: trace?.candidates ?? null,
                  stoppedBy: trace?.stoppedBy ?? null,
                },
              },
            ];
          });
          const evidence = sourceHydrator
            ? sourceHydrator
                .hydrate(
                  fused.map((candidate, rank) => ({
                    chunkId: candidate.id,
                    score: candidate.score,
                    rank: rank + 1,
                  })),
                )
                .map((window) => ({
                  id: window.id,
                  sourceId: window.sourceId,
                  text: window.text,
                  score: window.score,
                  rank: window.rank,
                  metadata: {
                    retrievalMode,
                    anchorChunkIds: window.anchorIds.join(","),
                    sourceChunkIds: window.chunkIds.join(","),
                    startOrdinal: window.startOrdinal,
                    endOrdinal: window.endOrdinal,
                    windowCharacters: window.text.length,
                  },
                }))
            : chunkEvidence;
          const result: RetrievalResult = {
            datasetId: bundle.id,
            frameworkId: this.id,
            queryId: query.id,
            status: "ok",
            evidence,
            latencyMs: performance.now() - startedAt + (expansion?.latencyMs ?? 0),
            inputTokens: Math.ceil(
              evidence.reduce((total, item) => total + item.text.length, 0) / 4,
            ),
            error: null,
            frameworkVersion,
            answerPolicy,
            configDigest: digest,
          };
          writeJsonAtomic(checkpointPath, result);
          return result;
        } catch (error) {
          const result: RetrievalResult = {
            datasetId: bundle.id,
            frameworkId: this.id,
            queryId: query.id,
            status: "error",
            evidence: [],
            latencyMs: performance.now() - startedAt,
            inputTokens: null,
            error: (error as Error).message,
            frameworkVersion,
            answerPolicy,
            configDigest: digest,
          };
          writeJsonAtomic(checkpointPath, result);
          return result;
        }
      },
    );
  }

  private async createBidirectionalAgent(
    bundle: DatasetBundle,
    indexDirectory: string,
    knowledgeRetriever: BidirectionalNLayerRetriever,
  ): Promise<KontextAgent> {
    const store = new FileOntologyStore(join(indexDirectory, "ontology-schema"));
    const vectorStore = new InMemoryVectorStore(async () => new Float32Array(0));
    const llm = new CodexLlmAdapter(this.codexClient, this.manifest);
    const agent = new KontextAgent({
      ontologySchemaGraph: new OntologyGraph(new Map(), [], {
        maxDepth: 2,
        maxTokens: 2_000,
        strategy: TraversalStrategy.WEIGHTED_DFS,
      }),
      router: new RouterLLMAdapter(llm, llm),
      mcpConnectors: [],
      mcpLayerAdapters: [],
      metaIndexStore: new InMemoryMetaIndexStore(),
      fetcherRegistry: new ContentFetcherRegistry(),
      vectorStore,
      mappingStrategy: new KeywordMappingStrategy(),
      metaSelector: new ScoreBasedSelector(),
      ingestPipeline: new IngestPipeline(llm, store, vectorStore),
      pipeline: DEFAULT_PIPELINE,
      legacySnapshotStore: store,
      stateId: `rag-eval-bidirectional-${documentDigest(bundle.documents).slice(0, 24)}`,
      organizationId: BIDIRECTIONAL_ORGANIZATION_ID,
      knowledgeRetriever,
    });
    await agent.initialize();
    return agent;
  }
}

interface EmbeddingCheckpointResult {
  readonly vectors: Float32Array;
  readonly inputTokens: number;
}

interface MultiQueryExpansionCheckpoint extends MultiQueryExpansion {
  readonly schemaVersion: 1;
  readonly policyVersion: typeof MULTI_QUERY_POLICY_VERSION;
  readonly queryId: string;
  readonly questionDigest: string;
}

async function expandWithCheckpoint(
  expander: Pick<MultiQueryExpander, "expand"> | null,
  queryId: string,
  question: string,
  directory: string,
  cacheOnly = false,
): Promise<MultiQueryExpansionCheckpoint> {
  const questionDigest = createHash("sha256").update(question).digest("hex");
  const checkpointPath = join(
    directory,
    `${createHash("sha256").update(queryId).update("\0").update(question).digest("hex")}.json`,
  );
  if (existsSync(checkpointPath)) {
    try {
      const cached = JSON.parse(
        readFileSync(checkpointPath, "utf8"),
      ) as MultiQueryExpansionCheckpoint;
      const structurallyValid =
        cached.schemaVersion === 1 &&
        cached.policyVersion === MULTI_QUERY_POLICY_VERSION &&
        cached.queryId === queryId &&
        cached.questionDigest === questionDigest &&
        Array.isArray(cached.queries) &&
        cached.queries.length <= 3 &&
        cached.queries.every((query) => typeof query === "string") &&
        (cached.error === null || typeof cached.error === "string");
      const cacheOnlySuccessfulExpansion =
        structurallyValid &&
        cached.error === null &&
        cached.queries.length > 0 &&
        validCachedExpandedQueries(cached.queries, question);
      const cacheOnlyOriginalQueryFallback =
        structurallyValid &&
        cached.queries.length === 0 &&
        typeof cached.error === "string" &&
        cached.error.trim().length > 0;
      const cacheOnlyValid = cacheOnlySuccessfulExpansion || cacheOnlyOriginalQueryFallback;
      if (structurallyValid && (!cacheOnly || cacheOnlyValid)) {
        return cached;
      }
    } catch {
      // Invalid or interrupted expansion checkpoints are regenerated below.
    }
  }
  if (cacheOnly) {
    throw new Error(`Required cached multi-query expansion missing or invalid for ${queryId}`);
  }
  if (!expander) throw new Error("Multi-query expansion requires a configured expander");
  const expansion = await expander.expand(question);
  const checkpoint: MultiQueryExpansionCheckpoint = {
    schemaVersion: 1,
    policyVersion: MULTI_QUERY_POLICY_VERSION,
    queryId,
    questionDigest,
    ...expansion,
  };
  writeJsonAtomic(checkpointPath, checkpoint);
  return checkpoint;
}

function validCachedExpandedQueries(queries: readonly string[], question: string): boolean {
  const seen = new Set<string>([normalizedQueryKey(question)]);
  for (const query of queries) {
    const trimmed = query.trim();
    const key = normalizedQueryKey(trimmed);
    if (!trimmed || trimmed.length > 500 || seen.has(key)) return false;
    seen.add(key);
  }
  return true;
}

function normalizedQueryKey(value: string): string {
  return value.toLocaleLowerCase().replace(/\s+/g, " ").trim();
}

function emptyMultiQueryCheckpoint(
  queryId: string,
  question: string,
): MultiQueryExpansionCheckpoint {
  return {
    schemaVersion: 1,
    policyVersion: MULTI_QUERY_POLICY_VERSION,
    queryId,
    questionDigest: createHash("sha256").update(question).digest("hex"),
    queries: [],
    latencyMs: 0,
    inputTokens: 0,
    outputTokens: 0,
    error: null,
  };
}

function multiQueryEmbeddingId(queryId: string, index: number): string {
  return `${queryId}::multi-query-${index + 1}`;
}

function multiQueryQueryDigest(
  bundle: DatasetBundle,
  indexDigest: string,
  expansions: readonly MultiQueryExpansionCheckpoint[],
): string {
  const hash = createHash("sha256")
    .update(kgQueryDigest(bundle, indexDigest))
    .update("\0")
    .update(MULTI_QUERY_POLICY_VERSION)
    .update("\0");
  for (const expansion of expansions) {
    hash.update(expansion.queryId).update("\0");
    for (const query of expansion.queries) hash.update(query).update("\0");
  }
  return hash.digest("hex");
}

class BudgetedBidirectionalRetriever extends BidirectionalNLayerRetriever {
  constructor(
    graph: SearchGraphPort,
    private readonly defaultBudget: Readonly<Partial<SearchBudget>>,
  ) {
    super(graph);
  }

  override retrieve(input: BidirectionalRetrievalInput): Promise<BidirectionalRetrievalResult> {
    return super.retrieve({
      ...input,
      budget: { ...this.defaultBudget, ...input.budget },
    });
  }
}

interface EmbeddingBatchMetadata {
  readonly schemaVersion: 1;
  readonly model: string;
  readonly dimensions: number;
  readonly digest: string;
  readonly task: "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY";
  readonly offset: number;
  readonly ids: readonly string[];
  readonly inputTokens: number;
}

async function embedWithCheckpoints(
  client: EmbeddingClient,
  inputs: readonly EmbeddingInput[],
  task: "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY",
  directory: string,
  digest: string,
  cacheOnly = false,
): Promise<EmbeddingCheckpointResult> {
  const batchSize = 100;
  const vectors = new Float32Array(inputs.length * client.dimensions);
  let inputTokens = 0;
  if (!cacheOnly) mkdirSync(directory, { recursive: true });
  for (let offset = 0; offset < inputs.length; offset += batchSize) {
    const batch = inputs.slice(offset, offset + batchSize);
    const stem = `batch-${String(offset).padStart(8, "0")}`;
    const metadataPath = join(directory, `${stem}.json`);
    const vectorsPath = join(directory, `${stem}.f32`);
    const ids = batch.map((input) => input.id);
    const cached = loadEmbeddingBatch(metadataPath, vectorsPath, client, digest, task, offset, ids);
    if (cached) {
      vectors.set(cached.vectors, offset * client.dimensions);
      inputTokens += cached.inputTokens;
      continue;
    }
    if (cacheOnly) {
      throw new Error(
        `Required cached ${task.toLowerCase()} embedding batch missing or invalid at ${stem}`,
      );
    }
    const usageBefore = client.getUsage();
    const embeddings = await client.embed(batch, task);
    const usageAfter = client.getUsage();
    const batchInputTokens = usageAfter.inputTokens - usageBefore.inputTokens;
    const batchVectors = new Float32Array(batch.length * client.dimensions);
    embeddings.forEach((embedding, index) =>
      batchVectors.set(embedding.values, index * client.dimensions),
    );
    writeFileSync(
      vectorsPath,
      Buffer.from(batchVectors.buffer, batchVectors.byteOffset, batchVectors.byteLength),
    );
    const metadata: EmbeddingBatchMetadata = {
      schemaVersion: 1,
      model: client.model,
      dimensions: client.dimensions,
      digest,
      task,
      offset,
      ids,
      inputTokens: batchInputTokens,
    };
    writeFileSync(metadataPath, `${JSON.stringify(metadata)}\n`, "utf8");
    vectors.set(batchVectors, offset * client.dimensions);
    inputTokens += batchInputTokens;
  }
  return { vectors, inputTokens };
}

function loadEmbeddingBatch(
  metadataPath: string,
  vectorsPath: string,
  client: EmbeddingClient,
  digest: string,
  task: "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY",
  offset: number,
  ids: readonly string[],
): EmbeddingCheckpointResult | null {
  if (!existsSync(metadataPath) || !existsSync(vectorsPath)) return null;
  try {
    const metadata = JSON.parse(readFileSync(metadataPath, "utf8")) as EmbeddingBatchMetadata;
    if (
      metadata.schemaVersion !== 1 ||
      metadata.model !== client.model ||
      metadata.dimensions !== client.dimensions ||
      metadata.digest !== digest ||
      metadata.task !== task ||
      metadata.offset !== offset ||
      metadata.ids.length !== ids.length ||
      metadata.ids.some((id, index) => id !== ids[index])
    )
      return null;
    const buffer = readFileSync(vectorsPath);
    if (buffer.byteLength !== ids.length * client.dimensions * Float32Array.BYTES_PER_ELEMENT)
      return null;
    const view = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
    return { vectors: new Float32Array(view), inputTokens: metadata.inputTokens };
  } catch {
    return null;
  }
}

function graphRagDomain(datasetId: DatasetBundle["id"]): "medical" | "novel" | null {
  if (datasetId === "graphrag-bench-medical") return "medical";
  if (datasetId === "graphrag-bench-novel") return "novel";
  return null;
}

function chunkTitle(id: string): string {
  const match = /^(.*)-(\d+)$/.exec(id);
  return match ? `${match[1]} chunk ${match[2]}` : id;
}

function chunkCanonicalDocuments(
  documents: readonly CorpusDocument[],
  chunkCharacters = 5_000,
  overlapCharacters = 400,
): BenchDoc[] {
  const chunks: BenchDoc[] = [];
  for (const document of documents) {
    const sourceId = document.sourceId || document.id;
    let start = 0;
    let index = 0;
    while (start < document.text.length) {
      const hardEnd = Math.min(document.text.length, start + chunkCharacters);
      const boundary =
        hardEnd < document.text.length ? document.text.lastIndexOf(" ", hardEnd) : -1;
      const end = boundary > start + Math.floor(chunkCharacters * 0.7) ? boundary : hardEnd;
      const sourceText = document.text.slice(start, end).trim();
      if (sourceText) {
        chunks.push({
          id: `${document.id}-${index}`,
          title: `${sourceId} chunk ${index}`,
          body:
            index === 0 && document.title.trim()
              ? `${document.title.trim()}\n\n${sourceText}`
              : sourceText,
        });
        index += 1;
      }
      if (end >= document.text.length) break;
      start = Math.max(start + 1, end - overlapCharacters);
    }
  }
  return chunks;
}

interface CorpusCoverageResult {
  readonly docs: readonly BenchDoc[];
  readonly originalSourceCount: number;
  readonly artifactCoveredSourceCount: number;
  readonly supplementedSourceIds: readonly string[];
  readonly supplementedSourceDigest: string;
}

function completeCorpusFromArtifact(
  artifactDocs: readonly BenchDoc[],
  documents: readonly CorpusDocument[],
): CorpusCoverageResult {
  const artifactSourceIds = new Set(
    artifactDocs.map((doc) => normalizeCoverageText(sourceDocumentId(doc))),
  );
  const artifactTextFingerprints = artifactDocs
    .map((doc) => coverageFingerprint(normalizeCoverageText(doc.body)))
    .filter((probes) => probes.length > 0);
  const uncovered = documents.filter((document) => {
    const sourceId = document.sourceId || document.id;
    if (artifactSourceIds.has(normalizeCoverageText(sourceId))) return false;
    const sourceText = normalizeCoverageText(`${document.title}\n${document.text}`);
    return !artifactTextFingerprints.some((probes) =>
      probes.every((probe) => sourceText.includes(probe)),
    );
  });
  const supplementalDocs = chunkCanonicalDocuments(uncovered);
  const seenIds = new Set(artifactDocs.map((doc) => doc.id));
  for (const doc of supplementalDocs) {
    if (seenIds.has(doc.id)) {
      throw new Error(`Canonical corpus gap-fill chunk conflicts with artifact chunk ${doc.id}`);
    }
    seenIds.add(doc.id);
  }
  return {
    docs: [...artifactDocs, ...supplementalDocs],
    originalSourceCount: documents.length,
    artifactCoveredSourceCount: documents.length - uncovered.length,
    supplementedSourceIds: uncovered.map((document) => document.sourceId || document.id),
    supplementedSourceDigest: documentDigest(uncovered),
  };
}

function normalizeCoverageText(value: string): string {
  return value.toLocaleLowerCase().replace(/\s+/gu, " ").trim();
}

function coverageFingerprint(normalizedText: string): readonly string[] {
  if (normalizedText.length < 32) return [];
  if (normalizedText.length <= 256) return [normalizedText];
  const width = 128;
  const starts = [
    0,
    Math.floor((normalizedText.length - width) / 2),
    normalizedText.length - width,
  ];
  return Array.from(new Set(starts.map((start) => normalizedText.slice(start, start + width))));
}

function emptyKnowledgeGraph(docs: readonly BenchDoc[]): KGStore {
  return {
    entities: new Map(),
    edges: [],
    chunkToEntities: new Map(docs.map((doc) => [doc.id, []])),
  };
}

function sourceDocumentId(doc: BenchDoc): string {
  return doc.title.replace(/\s+chunk\s+\d+$/i, "");
}

function toSourceChunk(doc: BenchDoc, fallbackOrdinal: number): SourceChunk {
  const match = /^(.*)-(\d+)$/.exec(doc.id);
  return {
    id: doc.id,
    sourceId: sourceDocumentId(doc),
    ordinal: match ? Number(match[2]) : fallbackOrdinal,
    text: doc.body,
  };
}

function readKnowledgeGraph(path: string): KGStore {
  const serialized = JSON.parse(readFileSync(path, "utf8")) as KGSerialized;
  return {
    entities: new Map(serialized.entities.map((entity) => [entity.id, entity])),
    edges: serialized.edges,
    chunkToEntities: new Map(serialized.chunkToEntities),
  };
}

function kgDocumentDigest(docs: readonly BenchDoc[]): string {
  const hash = createHash("sha256");
  for (const doc of docs) hash.update(doc.id).update("\0").update(doc.body).update("\0");
  return hash.digest("hex");
}

function kgQueryDigest(bundle: DatasetBundle, indexDigest: string): string {
  const hash = createHash("sha256").update(indexDigest).update("\0");
  for (const query of bundle.queries)
    hash.update(query.id).update("\0").update(query.text).update("\0");
  return hash.digest("hex");
}

function splitVectors(
  vectors: Float32Array,
  ids: readonly string[],
  dimensions: number,
): ReadonlyMap<string, Float32Array> {
  return new Map(
    ids.map((id, index) => [id, vectors.slice(index * dimensions, (index + 1) * dimensions)]),
  );
}

function rankKgVectors(
  vectors: Float32Array,
  queryVector: ArrayLike<number>,
  dimensions: number,
  limit: number,
): Array<{ readonly index: number; readonly score: number }> {
  if (vectors.length % dimensions !== 0) throw new Error("Corrupt KG embedding index length");
  const ranked: Array<{ index: number; score: number }> = [];
  for (let index = 0; index < vectors.length / dimensions; index += 1) {
    ranked.push({
      index,
      score: cosineSimilarity(
        vectors.subarray(index * dimensions, (index + 1) * dimensions),
        queryVector,
      ),
    });
  }
  return ranked
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .slice(0, limit);
}

function summarizeForClassification(text: string): string {
  return text.replace(/\s+/g, " ").trim().slice(0, 500);
}

function fuseVectorAndBm25Perspective(
  vectorIds: readonly string[],
  bm25Ids: readonly string[],
  limit: number,
  reciprocalRankConstant: number,
): string[] {
  return fuseRankings(
    [
      { name: "vector", ids: vectorIds, weight: MAX_EXISTING_STACK_FUSION.vector },
      { name: "bm25", ids: bm25Ids, weight: MAX_EXISTING_STACK_FUSION.bm25 },
    ],
    limit,
    reciprocalRankConstant,
  ).map((candidate) => candidate.id);
}

function isCacheOnlyCoverageMode(mode: KontextRetrievalMode): boolean {
  return (
    mode === "v14a-anchored-deterministic-soft-coverage-stack" ||
    mode === "v14b-anchored-deterministic-quota-coverage-stack"
  );
}

function requiredPrecomputedIndexDirectory(directory: string | null): string {
  if (!directory || !existsSync(directory)) {
    throw new Error(
      "KONTEXT_RAG_EVAL_PRECOMPUTED_INDEX must reference a readable precomputed index",
    );
  }
  return directory;
}

function validEmbeddingDimensions(dimensions: number | undefined): dimensions is number {
  return typeof dimensions === "number" && Number.isInteger(dimensions) && dimensions > 0;
}

function cacheOnlyEmbeddingClient(manifest: RagEvalManifest): EmbeddingClient {
  const dimensions = manifest.models.embedding.dimensions;
  if (!validEmbeddingDimensions(dimensions)) {
    throw new Error("Cache-only retrieval requires a positive manifest embedding dimension");
  }
  return {
    model: manifest.models.embedding.model,
    dimensions,
    async embed(): Promise<never> {
      throw new Error("Cache-only retrieval cannot call the embedding API");
    },
    getUsage() {
      return { requests: 0, inputTokens: 0, totalTokens: 0 };
    },
  };
}

function frameworkVersion(mode: KontextRetrievalMode): string {
  if (mode === "bidirectional-kg") return BIDIRECTIONAL_FRAMEWORK_VERSION;
  if (mode === "max-existing-stack") return MAX_EXISTING_STACK_FRAMEWORK_VERSION;
  if (mode === "source-hydrated-stack") return SOURCE_HYDRATED_STACK_FRAMEWORK_VERSION;
  if (mode === "source-hydrated-llm-stack") return SOURCE_HYDRATED_LLM_STACK_FRAMEWORK_VERSION;
  if (mode === "source-hydrated-llm-recall-safe-stack") {
    return SOURCE_HYDRATED_LLM_RECALL_SAFE_STACK_FRAMEWORK_VERSION;
  }
  if (mode === "source-hydrated-llm-candidate-safe-stack") {
    return SOURCE_HYDRATED_LLM_CANDIDATE_SAFE_STACK_FRAMEWORK_VERSION;
  }
  if (mode === "source-hydrated-llm-coverage-aware-stack") {
    return SOURCE_HYDRATED_LLM_COVERAGE_AWARE_STACK_FRAMEWORK_VERSION;
  }
  if (mode === "multi-query-coverage-aware-stack") {
    return MULTI_QUERY_COVERAGE_AWARE_STACK_FRAMEWORK_VERSION;
  }
  if (mode === "multi-query-standard-rerank-stack") {
    return MULTI_QUERY_STANDARD_RERANK_STACK_FRAMEWORK_VERSION;
  }
  if (mode === "multi-query-plan-aware-coverage-stack") {
    return MULTI_QUERY_PLAN_AWARE_COVERAGE_STACK_FRAMEWORK_VERSION;
  }
  if (mode === "multi-query-anchored-evidence-answer-stack") {
    return V13_STACK_DESCRIPTOR.frameworkVersion;
  }
  if (mode === "corpus-complete-anchored-evidence-answer-stack") {
    return V15_STACK_DESCRIPTOR.frameworkVersion;
  }
  if (mode === "v14a-anchored-deterministic-soft-coverage-stack") {
    return V14A_STACK_DESCRIPTOR.frameworkVersion;
  }
  if (mode === "v14b-anchored-deterministic-quota-coverage-stack") {
    return V14B_STACK_DESCRIPTOR.frameworkVersion;
  }
  if (mode === "adaptive-eece-stack") return ADAPTIVE_EECE_STACK_FRAMEWORK_VERSION;
  return "workspace-0.1.0";
}

function maxExistingStackDigest(
  manifest: RagEvalManifest,
  hydrateSourceContext = false,
  rerankWithLlm = false,
  recallSafeLlmRerank = false,
  honorDeclaredCandidateK = false,
  adaptiveEece = false,
  coverageAwareRerank = false,
  multiQuery = false,
  planAwareCoverage = false,
  anchoredEvidenceAnswer = false,
  candidateCount = MAX_EXISTING_STACK_CANDIDATES,
  deterministicCoverage: CacheOnlyCoverageDescriptor | null = null,
  completeCorpusCoverage = false,
): string {
  const inheritedStackIdentity = planAwareCoverage
    ? `\0v12-multi-query-plan-aware-coverage-stack\0${MULTI_QUERY_POLICY_VERSION}\0`
    : multiQuery
      ? coverageAwareRerank
        ? `\0v11b-multi-query-coverage-aware-stack\0${MULTI_QUERY_POLICY_VERSION}\0`
        : `\0v11a-multi-query-standard-rerank-stack\0${MULTI_QUERY_POLICY_VERSION}\0`
      : coverageAwareRerank
        ? "\0v10-source-hydrated-llm-coverage-aware-stack\0"
        : adaptiveEece
          ? "\0adaptive-eece-stack-v9\0"
          : honorDeclaredCandidateK
            ? "\0v7-source-hydrated-llm-candidate-safe-stack\0"
            : recallSafeLlmRerank
              ? "\0v6-source-hydrated-llm-recall-safe-stack\0"
              : rerankWithLlm
                ? "\0v5-source-hydrated-llm-stack\0"
                : hydrateSourceContext
                  ? "\0v4-source-hydrated-stack\0"
                  : "\0v3-max-existing-stack\0";
  const stackIdentity = deterministicCoverage
    ? `\0${JSON.stringify({
        retrievalMode: deterministicCoverage.retrievalMode,
        frameworkVersion: deterministicCoverage.frameworkVersion,
        answerPolicy: deterministicCoverage.answerPolicy,
        selectionPolicy: deterministicCoverage.selectionPolicy,
        cacheOnly: deterministicCoverage.cacheOnly,
        candidateCount: deterministicCoverage.candidateCount,
        originalQueryWeight: deterministicCoverage.originalQueryWeight,
        expandedQueryWeight: deterministicCoverage.expandedQueryWeight,
        reciprocalRankConstant: deterministicCoverage.reciprocalRankConstant,
        topWindow: deterministicCoverage.topWindow,
        originalQuota: deterministicCoverage.originalQuota,
        perExpansionQuota: deterministicCoverage.perExpansionQuota,
        multiQueryPolicyVersion: MULTI_QUERY_POLICY_VERSION,
      })}\0`
    : completeCorpusCoverage
      ? `\0${V15_STACK_DESCRIPTOR.retrievalMode}\0${MULTI_QUERY_POLICY_VERSION}\0original-query-weight=${V13_ORIGINAL_QUERY_WEIGHT}\0expanded-query-weight=${EXPANDED_QUERY_WEIGHT}\0coverage-policy=${V15_STACK_DESCRIPTOR.corpusCoveragePolicy}\0coverage-policy-version=${V15_STACK_DESCRIPTOR.corpusCoveragePolicyVersion}\0`
      : anchoredEvidenceAnswer
        ? `\0v13-anchored-evidence-answer-stack\0${MULTI_QUERY_POLICY_VERSION}\0original-query-weight=${V13_ORIGINAL_QUERY_WEIGHT}\0expanded-query-weight=${EXPANDED_QUERY_WEIGHT}\0`
        : inheritedStackIdentity;
  const hash = createHash("sha256")
    .update(manifestDigest(manifest))
    .update(stackIdentity)
    .update(String(candidateCount))
    .update("\0")
    .update(JSON.stringify(MAX_EXISTING_STACK_FUSION));
  if (hydrateSourceContext) {
    hash
      .update("\0")
      .update(
        JSON.stringify(
          recallSafeLlmRerank ? RECALL_SAFE_SOURCE_CONTEXT_POLICY : SOURCE_CONTEXT_POLICY,
        ),
      );
  }
  if (rerankWithLlm) {
    hash
      .update("\0")
      .update(manifest.models.answer.model)
      .update("\0")
      .update(manifest.models.answer.reasoningEffort ?? "");
  }
  return hash.digest("hex");
}

function documentDigest(documents: readonly CorpusDocument[]): string {
  const hash = createHash("sha256");
  for (const document of documents)
    hash.update(document.id).update("\0").update(document.text).update("\0");
  return hash.digest("hex");
}

async function mapWithConcurrency<T, R>(
  values: readonly T[],
  concurrency: number,
  operation: (value: T) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(values.length);
  let nextIndex = 0;
  const workers = Array.from(
    { length: Math.min(Math.max(1, concurrency), values.length) },
    async () => {
      while (true) {
        const index = nextIndex;
        nextIndex += 1;
        if (index >= values.length) return;
        const value = values[index];
        if (value === undefined) return;
        results[index] = await operation(value);
      }
    },
  );
  await Promise.all(workers);
  return results;
}
