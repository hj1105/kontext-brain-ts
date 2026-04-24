/**
 * kontext-brain runners for the benchmark.
 *
 * Provides multiple variants so we can compare different configurations:
 *   - V1: DEFAULT_PIPELINE + KeywordMappingStrategy (original kontext, known broken for flat ontologies)
 *   - V2: Custom direct pipeline — LLM maps query to nodes, then meta search + content fetch per node
 *   - V3: Same as V2 but with VectorMappingStrategy (real embeddings)
 */

import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import {
  AliasEntityExtractor,
  BM25BodyExtractor,
  Bm25NodeMappingStrategy,
  CentroidNodeEmbedder,
  ContentFetcherRegistry,
  DEFAULT_PIPELINE,
  DataSource,
  DefaultPromptTemplates,
  EdgeAwareMappingStrategy,
  EntityRetriever,
  HybridMappingStrategy,
  InMemoryEntityIndex,
  InMemoryMetaIndexStore,
  KeywordMappingStrategy,
  LLMMappingStrategy,
  LayeredQueryPipeline,
  MmrSelector,
  OntologyGraph,
  PERNODE_PIPELINE,
  RouterLLMAdapter,
  ScoreBasedSelector,
  TraversalStrategy,
  VectorMappingStrategy,
  createEntity,
  createMetaDocument,
  createNode,
  type ContentFetcher,
  type Entity,
  type MetaDocument,
  type NodeMappingStrategy,
  type OntologyNode,
} from "@kontext-brain/core";
import { LangChainLLMAdapter, LangChainVectorStore } from "@kontext-brain/llm";
import type { BenchDoc } from "./corpus.js";

function assignDocToNode(docId: string): string {
  if (docId.startsWith("backend-")) return "backend";
  if (docId.startsWith("frontend-")) return "frontend";
  if (docId.startsWith("ops-")) return "ops";
  if (docId.startsWith("sec-")) return "security";
  if (docId.startsWith("data-")) return "data";
  if (docId.startsWith("ml-")) return "ml";
  return "misc";
}

function buildGraph(): OntologyGraph {
  const nodes = new Map<string, OntologyNode>();
  nodes.set(
    "backend",
    createNode({
      id: "backend",
      description: "REST API server database postgres JWT authentication tokens authorization",
      weight: 0.9,
      level: 0,
    }),
  );
  nodes.set(
    "frontend",
    createNode({
      id: "frontend",
      description: "React TypeScript components UI rendering performance web vitals",
      weight: 0.9,
      level: 0,
    }),
  );
  nodes.set(
    "ops",
    createNode({
      id: "ops",
      description: "Docker Kubernetes CI CD deployment container operations infrastructure images",
      weight: 0.9,
      level: 0,
    }),
  );
  nodes.set(
    "security",
    createNode({
      id: "security",
      description: "OWASP secrets TLS HTTPS encryption vulnerabilities certificates rotation RBAC CSRF XSS supply chain",
      weight: 0.9,
      level: 0,
    }),
  );
  nodes.set(
    "data",
    createNode({
      id: "data",
      description: "ETL warehouse pipeline schema dimensions facts star snowflake transformations",
      weight: 0.8,
      level: 0,
    }),
  );
  nodes.set(
    "ml",
    createNode({
      id: "ml",
      description: "machine learning feature store model training serving drift lineage",
      weight: 0.8,
      level: 0,
    }),
  );
  // Cross-domain edges so EdgeAwareMappingStrategy has work to do
  const edges = [
    { from: "backend", to: "security", weight: 0.7 },
    { from: "ops", to: "security", weight: 0.6 },
    { from: "frontend", to: "backend", weight: 0.5 },
  ];
  return new OntologyGraph(nodes, edges, {
    maxDepth: 2,
    maxTokens: 4000,
    strategy: TraversalStrategy.WEIGHTED_DFS,
  });
}

interface RunResult {
  answer: string;
  retrievedDocIds: string[];
  contextChars: number;
}

interface SharedState {
  chat: ChatOllama;
  embeddings: OllamaEmbeddings;
  adapter: LangChainLLMAdapter;
  vectorStore: LangChainVectorStore;
  corpusById: Map<string, BenchDoc>;
  graph: OntologyGraph;
  metaIndex: InMemoryMetaIndexStore;
  fetcherRegistry: ContentFetcherRegistry;
}

async function buildSharedState(corpus: BenchDoc[]): Promise<SharedState> {
  const baseUrl = "http://localhost:11434";
  const chat = new ChatOllama({
    baseUrl,
    model: "qwen2.5:1.5b",
    temperature: 0,
    numGpu: 0,
    numCtx: 2048,
    numPredict: 256,
  });
  const embeddings = new OllamaEmbeddings({ baseUrl, model: "nomic-embed-text" });
  const adapter = new LangChainLLMAdapter(chat);
  const vectorStore = new LangChainVectorStore(embeddings);
  const graph = buildGraph();
  const corpusById = new Map<string, BenchDoc>();
  for (const d of corpus) corpusById.set(d.id, d);

  // Embed node descriptions for VectorMappingStrategy
  for (const node of graph.nodes.values()) {
    const v = await vectorStore.embed(node.description);
    await vectorStore.upsert(node.id, v);
  }

  // Index meta docs
  const metaIndex = new InMemoryMetaIndexStore();
  const byNode = new Map<string, BenchDoc[]>();
  for (const doc of corpus) {
    const nodeId = assignDocToNode(doc.id);
    const list = byNode.get(nodeId) ?? [];
    list.push(doc);
    byNode.set(nodeId, list);
  }
  for (const [nodeId, docs] of byNode) {
    const metaDocs: MetaDocument[] = docs.map((d) =>
      createMetaDocument({
        id: d.id,
        title: d.title,
        source: DataSource.CUSTOM,
        ontologyNodeId: nodeId,
      }),
    );
    await metaIndex.index(nodeId, metaDocs);
  }

  // Fetcher resolves body by id
  const fetcherRegistry = new ContentFetcherRegistry();
  const fetcher: ContentFetcher = {
    source: DataSource.CUSTOM,
    async fetch(meta) {
      const doc = corpusById.get(meta.id);
      return {
        metaDocumentId: meta.id,
        title: meta.title,
        body: doc?.body ?? "",
        source: meta.source,
        sectionContent: null,
        fetchedAt: new Date(),
      };
    },
  };
  fetcherRegistry.register(fetcher);

  return { chat, embeddings, adapter, vectorStore, corpusById, graph, metaIndex, fetcherRegistry };
}

// ── V1: original default pipeline ─────────────────────────────

export class KontextV1 {
  private state!: SharedState;
  private pipeline!: LayeredQueryPipeline;

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
    const router = new RouterLLMAdapter(this.state.adapter, this.state.adapter);
    this.pipeline = new LayeredQueryPipeline(
      this.state.graph,
      router,
      this.state.metaIndex,
      this.state.fetcherRegistry,
      {
        mappingStrategy: new KeywordMappingStrategy(),
        metaSelector: new ScoreBasedSelector(),
        pipeline: DEFAULT_PIPELINE,
      },
    );
  }

  async query(question: string): Promise<RunResult> {
    const res = await this.pipeline.execute(question);
    return {
      answer: res.answer,
      retrievedDocIds: Array.from(new Set(res.selectedMetaDocs.map((d) => d.id))),
      contextChars: res.contextTokensUsed * 4,
    };
  }
}

/** V1-fixed: LayeredQueryPipeline + PERNODE_PIPELINE + collector fix. */
export class KontextV1Fixed {
  private state!: SharedState;
  private pipeline!: LayeredQueryPipeline;

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
    const router = new RouterLLMAdapter(this.state.adapter, this.state.adapter);
    this.pipeline = new LayeredQueryPipeline(
      this.state.graph,
      router,
      this.state.metaIndex,
      this.state.fetcherRegistry,
      {
        mappingStrategy: new KeywordMappingStrategy(),
        metaSelector: new ScoreBasedSelector(),
        pipeline: PERNODE_PIPELINE,
      },
    );
  }

  async query(question: string): Promise<RunResult> {
    const res = await this.pipeline.execute(question);
    return {
      answer: res.answer,
      retrievedDocIds: Array.from(new Set(res.fetchedContents.map((d) => d.metaDocumentId))),
      contextChars: res.contextTokensUsed * 4,
    };
  }
}

// ── V2: custom direct pipeline with chosen mapping strategy ──

interface DirectQueryOptions {
  topNodes?: number;
  topDocsPerNode?: number;
  /** Compress bodies via BM25 (keeps top-N relevant sentences). Default: false (full body). */
  bm25Compress?: boolean;
  bm25Sentences?: number;
}

async function directQuery(
  state: SharedState,
  mappingStrategy: NodeMappingStrategy,
  question: string,
  opts: DirectQueryOptions = {},
): Promise<RunResult> {
  const topNodes = opts.topNodes ?? 2;
  const topDocsPerNode = opts.topDocsPerNode ?? 3;

  // 1. L1: map query to relevant ontology nodes
  const candidates = await mappingStrategy.findStartNodes(question, state.graph.nodes);
  const selectedNodes = candidates.slice(0, topNodes);

  // 2. L2: for each node, search meta index
  const allMeta: MetaDocument[] = [];
  for (const nodeId of selectedNodes) {
    const metas = await state.metaIndex.search(nodeId, question, topDocsPerNode);
    allMeta.push(...metas);
  }

  // 3. L3: fetch bodies (optionally BM25-compressed) and build context
  const parts: string[] = [];
  const retrievedIds: string[] = [];
  for (const meta of allMeta) {
    const content = await state.fetcherRegistry.fetch(meta);
    const body = opts.bm25Compress
      ? BM25BodyExtractor.extract(content.body, question, opts.bm25Sentences ?? 3)
      : content.body;
    parts.push(`### ${content.title}\n${body}`);
    retrievedIds.push(meta.id);
  }
  const context = parts.join("\n\n");

  // 4. Final reasoning
  const prompt = `You are a technical assistant. Answer the question based only on the provided context. Be concise.\n\nContext:\n${context}\n\nQuestion: ${question}\n\nAnswer:`;
  const response = await state.chat.invoke(prompt);
  const answer =
    typeof response.content === "string" ? response.content : JSON.stringify(response.content);

  return {
    answer,
    retrievedDocIds: Array.from(new Set(retrievedIds)),
    contextChars: context.length,
  };
}

export class KontextV2Keyword {
  private state!: SharedState;
  private strategy = new KeywordMappingStrategy();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
  }

  async query(question: string): Promise<RunResult> {
    return directQuery(this.state, this.strategy, question);
  }
}

export class KontextV3Vector {
  private state!: SharedState;
  private strategy!: VectorMappingStrategy;

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
    this.strategy = new VectorMappingStrategy(this.state.vectorStore);
  }

  async query(question: string): Promise<RunResult> {
    return directQuery(this.state, this.strategy, question);
  }
}

export class KontextV4LLM {
  private state!: SharedState;
  private strategy!: LLMMappingStrategy;

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
    this.strategy = new LLMMappingStrategy(this.state.adapter, DefaultPromptTemplates);
  }

  async query(question: string): Promise<RunResult> {
    return directQuery(this.state, this.strategy, question);
  }
}

/** V5: keyword mapping + BM25 body compression — minimize tokens. */
export class KontextV5Compressed {
  private state!: SharedState;
  private strategy = new KeywordMappingStrategy();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
  }

  async query(question: string): Promise<RunResult> {
    return directQuery(this.state, this.strategy, question, {
      bm25Compress: true,
      bm25Sentences: 3,
    });
  }
}

/** V6: hybrid keyword+vector mapping + direct pipeline. */
export class KontextV6Hybrid {
  private state!: SharedState;
  private strategy!: HybridMappingStrategy;

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
    this.strategy = new HybridMappingStrategy(this.state.vectorStore, 0.5, 3);
  }

  async query(question: string): Promise<RunResult> {
    return directQuery(this.state, this.strategy, question);
  }
}

/**
 * V7: HyDE (Hypothetical Document Embeddings).
 * 1. Ask LLM to generate a plausible answer to the question
 * 2. Embed that hypothetical answer (not the question)
 * 3. Use the resulting vector for similarity search over node descriptions
 *
 * Idea: question and answer live in different vector regions; embedding the
 * *answer* shape gets closer to the right docs in semantic space.
 */
export class KontextV7HyDE {
  private state!: SharedState;

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
  }

  async query(question: string): Promise<RunResult> {
    // Generate hypothetical answer (short, technical)
    const hydePrompt = `Write a single short paragraph (2-3 sentences) that would answer this question. Use technical keywords. No preamble.\n\nQuestion: ${question}\n\nAnswer:`;
    const hypo = await this.state.chat.invoke(hydePrompt);
    const hypoText =
      typeof hypo.content === "string" ? hypo.content : JSON.stringify(hypo.content);

    // Embed the hypothetical answer and search nodes
    const queryVec = await this.state.vectorStore.embed(hypoText);
    await this.state.vectorStore.upsert(`__hyde_query_${Date.now()}`, queryVec);
    const ranked = await this.state.vectorStore.similaritySearch(hypoText, 3);
    const selectedNodes = ranked.filter((id) => this.state.graph.nodes.has(id)).slice(0, 2);

    // Retrieve meta docs under selected nodes
    const allMeta: MetaDocument[] = [];
    for (const nodeId of selectedNodes) {
      const metas = await this.state.metaIndex.search(nodeId, question, 3);
      allMeta.push(...metas);
    }

    const parts: string[] = [];
    const retrievedIds: string[] = [];
    for (const meta of allMeta) {
      const content = await this.state.fetcherRegistry.fetch(meta);
      parts.push(`### ${content.title}\n${content.body}`);
      retrievedIds.push(meta.id);
    }
    const context = parts.join("\n\n");
    const prompt = `You are a technical assistant. Answer the question based only on the provided context. Be concise.\n\nContext:\n${context}\n\nQuestion: ${question}\n\nAnswer:`;
    const response = await this.state.chat.invoke(prompt);
    const answer =
      typeof response.content === "string" ? response.content : JSON.stringify(response.content);
    return {
      answer,
      retrievedDocIds: Array.from(new Set(retrievedIds)),
      contextChars: context.length,
    };
  }
}

/**
 * V8: Query expansion. Ask LLM to produce keyword variants of the query first,
 * then search with the expanded term set. Helps when user wording differs from
 * corpus wording.
 */
export class KontextV8QueryExpand {
  private state!: SharedState;
  private strategy = new KeywordMappingStrategy();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
  }

  async query(question: string): Promise<RunResult> {
    const expandPrompt = `Given the question, output 8 related technical keywords separated by spaces. No other text.\n\nQuestion: ${question}\n\nKeywords:`;
    const resp = await this.state.chat.invoke(expandPrompt);
    const keywords =
      typeof resp.content === "string" ? resp.content : JSON.stringify(resp.content);
    const expandedQuery = `${question} ${keywords.trim()}`;
    return directQuery(this.state, this.strategy, expandedQuery, {
      topNodes: 2,
      topDocsPerNode: 3,
    });
  }
}

/**
 * V9: Over-retrieve then LLM rerank. Pull 6 candidate docs, ask the LLM to
 * pick the 3 most relevant, then answer with just those.
 */
export class KontextV9Rerank {
  private state!: SharedState;
  private strategy = new KeywordMappingStrategy();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
  }

  async query(question: string): Promise<RunResult> {
    // Over-retrieve
    const candidates = await this.strategy.findStartNodes(question, this.state.graph.nodes);
    const selectedNodes = candidates.slice(0, 3);
    const allMeta: MetaDocument[] = [];
    for (const nodeId of selectedNodes) {
      const metas = await this.state.metaIndex.search(nodeId, question, 3);
      allMeta.push(...metas);
    }
    const unique = Array.from(new Map(allMeta.map((m) => [m.id, m])).values()).slice(0, 6);

    // Rerank with LLM
    const listing = unique.map((d, i) => `${i}: ${d.title}`).join("\n");
    const rerankPrompt = `Given the question, output the 3 most relevant document numbers as comma-separated values. Example: 0,2,4\n\nDocuments:\n${listing}\n\nQuestion: ${question}\n\nAnswer:`;
    const resp = await this.state.chat.invoke(rerankPrompt);
    const text =
      typeof resp.content === "string" ? resp.content : JSON.stringify(resp.content);
    const picks = text
      .split(/[,\s]+/)
      .map((s) => Number.parseInt(s.trim(), 10))
      .filter((n) => Number.isInteger(n) && n >= 0 && n < unique.length);
    const top =
      picks.length > 0
        ? picks.slice(0, 3).map((i) => unique[i]!).filter(Boolean)
        : unique.slice(0, 3);

    // Fetch and answer
    const parts: string[] = [];
    const retrievedIds: string[] = [];
    for (const meta of top) {
      const content = await this.state.fetcherRegistry.fetch(meta);
      parts.push(`### ${content.title}\n${content.body}`);
      retrievedIds.push(meta.id);
    }
    const context = parts.join("\n\n");
    const prompt = `You are a technical assistant. Answer the question based only on the provided context. Be concise.\n\nContext:\n${context}\n\nQuestion: ${question}\n\nAnswer:`;
    const answerResp = await this.state.chat.invoke(prompt);
    const answer =
      typeof answerResp.content === "string"
        ? answerResp.content
        : JSON.stringify(answerResp.content);
    return {
      answer,
      retrievedDocIds: Array.from(new Set(retrievedIds)),
      contextChars: context.length,
    };
  }
}

/**
 * V10: combine v9 (LLM rerank) + v5 (BM25 compression).
 * Over-retrieve 6, rerank to 3, then BM25-compress bodies before answer.
 */
export class KontextV10RerankCompress {
  private state!: SharedState;
  private strategy = new KeywordMappingStrategy();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
  }

  async query(question: string): Promise<RunResult> {
    const candidates = await this.strategy.findStartNodes(question, this.state.graph.nodes);
    const selectedNodes = candidates.slice(0, 3);
    const allMeta: MetaDocument[] = [];
    for (const nodeId of selectedNodes) {
      const metas = await this.state.metaIndex.search(nodeId, question, 3);
      allMeta.push(...metas);
    }
    const unique = Array.from(new Map(allMeta.map((m) => [m.id, m])).values()).slice(0, 6);

    const listing = unique.map((d, i) => `${i}: ${d.title}`).join("\n");
    const rerankPrompt = `Given the question, output the 3 most relevant document numbers as comma-separated values. Example: 0,2,4\n\nDocuments:\n${listing}\n\nQuestion: ${question}\n\nAnswer:`;
    const resp = await this.state.chat.invoke(rerankPrompt);
    const text =
      typeof resp.content === "string" ? resp.content : JSON.stringify(resp.content);
    const picks = text
      .split(/[,\s]+/)
      .map((s) => Number.parseInt(s.trim(), 10))
      .filter((n) => Number.isInteger(n) && n >= 0 && n < unique.length);
    const top =
      picks.length > 0
        ? picks.slice(0, 3).map((i) => unique[i]!).filter(Boolean)
        : unique.slice(0, 3);

    const parts: string[] = [];
    const retrievedIds: string[] = [];
    for (const meta of top) {
      const content = await this.state.fetcherRegistry.fetch(meta);
      const body = BM25BodyExtractor.extract(content.body, question, 3);
      parts.push(`### ${content.title}\n${body}`);
      retrievedIds.push(meta.id);
    }
    const context = parts.join("\n\n");
    const prompt = `You are a technical assistant. Answer the question based only on the provided context. Be concise.\n\nContext:\n${context}\n\nQuestion: ${question}\n\nAnswer:`;
    const answerResp = await this.state.chat.invoke(prompt);
    const answer =
      typeof answerResp.content === "string"
        ? answerResp.content
        : JSON.stringify(answerResp.content);
    return {
      answer,
      retrievedDocIds: Array.from(new Set(retrievedIds)),
      contextChars: context.length,
    };
  }
}

/**
 * V11: hybrid mapping with keyword-heavy weight + compression.
 * Tunes HybridMappingStrategy keywordWeight up to reduce vector noise.
 */
export class KontextV11HybridHeavy {
  private state!: SharedState;
  private strategy!: HybridMappingStrategy;

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
    this.strategy = new HybridMappingStrategy(this.state.vectorStore, 0.75, 2);
  }

  async query(question: string): Promise<RunResult> {
    return directQuery(this.state, this.strategy, question, {
      topNodes: 2,
      topDocsPerNode: 3,
      bm25Compress: true,
    });
  }
}

/**
 * V12 Extractive: skip the final LLM entirely. Retrieve best doc via keyword map,
 * extract top-1 BM25 sentence, return that sentence as the "answer".
 *
 * Hypothesis: for factual short-answer queries, the corpus already contains the
 * literal answer. Sending it through another LLM adds cost but not info.
 */
export class KontextV12Extractive {
  private state!: SharedState;
  private strategy = new KeywordMappingStrategy();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
  }

  async query(question: string): Promise<RunResult> {
    const candidates = await this.strategy.findStartNodes(question, this.state.graph.nodes);
    const selectedNodes = candidates.slice(0, 2);
    const allMeta: MetaDocument[] = [];
    for (const nodeId of selectedNodes) {
      const metas = await this.state.metaIndex.search(nodeId, question, 3);
      allMeta.push(...metas);
    }
    // Pool all sentences across selected docs, rank globally by BM25, take top 2
    const sentences: Array<{ doc: MetaDocument; sentence: string }> = [];
    const retrievedIds: string[] = [];
    for (const meta of allMeta) {
      const content = await this.state.fetcherRegistry.fetch(meta);
      retrievedIds.push(meta.id);
      for (const s of content.body.split(/[.!?]\s+/).map((x) => x.trim()).filter((x) => x.length > 20)) {
        sentences.push({ doc: meta, sentence: s });
      }
    }
    // Rank by BM25 (reuse the helper from BM25BodyExtractor pattern)
    const scored = sentences.map((s) => {
      const extracted = BM25BodyExtractor.extract(s.sentence, question, 1);
      // extract() returns the best sentence; if it kept it fully, score is high
      return { ...s, score: extracted.length / Math.max(s.sentence.length, 1) };
    });
    // Simpler: use word overlap
    const q = question.toLowerCase().split(/\s+/).filter((w) => w.length > 2);
    for (const s of scored) {
      s.score =
        q.filter((w) => s.sentence.toLowerCase().includes(w)).length / Math.max(q.length, 1);
    }
    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, 2);

    const answer = top.map((x) => x.sentence).join(". ");
    return {
      answer,
      retrievedDocIds: Array.from(new Set(retrievedIds)),
      contextChars: answer.length,
    };
  }
}

/**
 * V13 Sentence-level + tiny-LLM: pool sentences, BM25 to top 3, short LLM answer.
 */
export class KontextV13SentenceLevel {
  private state!: SharedState;
  private strategy = new KeywordMappingStrategy();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
  }

  async query(question: string): Promise<RunResult> {
    const candidates = await this.strategy.findStartNodes(question, this.state.graph.nodes);
    const selectedNodes = candidates.slice(0, 2);
    const allMeta: MetaDocument[] = [];
    for (const nodeId of selectedNodes) {
      const metas = await this.state.metaIndex.search(nodeId, question, 3);
      allMeta.push(...metas);
    }
    const retrievedIds = new Set<string>();
    const sentencePool: string[] = [];
    for (const meta of allMeta) {
      const content = await this.state.fetcherRegistry.fetch(meta);
      retrievedIds.add(meta.id);
      // Keep only sentences with at least one query word
      const q = question.toLowerCase().split(/\s+/).filter((w) => w.length > 2);
      const sentences = content.body
        .split(/[.!?]\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 15);
      for (const s of sentences) {
        const lc = s.toLowerCase();
        const overlap = q.filter((w) => lc.includes(w)).length;
        if (overlap > 0) sentencePool.push(s);
      }
    }

    // Top 3 globally-ranked sentences
    const q = question.toLowerCase().split(/\s+/).filter((w) => w.length > 2);
    const ranked = sentencePool
      .map((s) => ({
        s,
        score: q.filter((w) => s.toLowerCase().includes(w)).length,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map((x) => x.s);
    const context = ranked.join(". ");

    const prompt = `Answer in one short sentence using only the facts below.\n\nFacts: ${context}\n\nQ: ${question}\nA:`;
    const response = await this.state.chat.invoke(prompt);
    const answer =
      typeof response.content === "string" ? response.content : JSON.stringify(response.content);
    return {
      answer,
      retrievedDocIds: Array.from(retrievedIds),
      contextChars: context.length,
    };
  }
}

/**
 * V14 Parallel retrieval: fetch all candidate docs concurrently, everything
 * else is compressed. Aims to cut retrieval latency.
 */
export class KontextV14Parallel {
  private state!: SharedState;
  private strategy = new KeywordMappingStrategy();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
  }

  async query(question: string): Promise<RunResult> {
    const candidates = await this.strategy.findStartNodes(question, this.state.graph.nodes);
    const selectedNodes = candidates.slice(0, 2);
    const metaResults = await Promise.all(
      selectedNodes.map((nodeId) => this.state.metaIndex.search(nodeId, question, 3)),
    );
    const allMeta = metaResults.flat();
    const contents = await Promise.all(allMeta.map((m) => this.state.fetcherRegistry.fetch(m)));
    const parts: string[] = [];
    const retrievedIds: string[] = [];
    for (let i = 0; i < contents.length; i++) {
      const content = contents[i]!;
      const meta = allMeta[i]!;
      const body = BM25BodyExtractor.extract(content.body, question, 2);
      parts.push(`### ${content.title}\n${body}`);
      retrievedIds.push(meta.id);
    }
    const context = parts.join("\n\n");
    const prompt = `Answer in 1-2 sentences using only the context.\n\nContext:\n${context}\n\nQuestion: ${question}\nAnswer:`;
    const response = await this.state.chat.invoke(prompt);
    const answer =
      typeof response.content === "string" ? response.content : JSON.stringify(response.content);
    return {
      answer,
      retrievedDocIds: Array.from(new Set(retrievedIds)),
      contextChars: context.length,
    };
  }
}

/**
 * V15 Adaptive: if keyword mapping finds exactly one confident node with a
 * single high-overlap doc, return its top BM25 sentence (skip LLM). Otherwise
 * fall back to v10.
 */
export class KontextV15Adaptive {
  private state!: SharedState;
  private strategy = new KeywordMappingStrategy();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
  }

  async query(question: string): Promise<RunResult> {
    const candidates = await this.strategy.findStartNodes(question, this.state.graph.nodes);
    const topNode = candidates[0];
    if (!topNode) return this.fallback(question, candidates);

    const metas = await this.state.metaIndex.search(topNode, question, 3);
    const q = question.toLowerCase().split(/\s+/).filter((w) => w.length > 2);
    // Confidence = title word overlap on top doc
    const top = metas[0];
    if (!top) return this.fallback(question, candidates);

    const confidence = q.filter((w) => top.title.toLowerCase().includes(w)).length / Math.max(q.length, 1);
    if (confidence >= 0.4) {
      // High confidence path — extractive answer
      const content = await this.state.fetcherRegistry.fetch(top);
      const sentences = content.body
        .split(/[.!?]\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 15);
      const ranked = sentences
        .map((s) => ({
          s,
          score: q.filter((w) => s.toLowerCase().includes(w)).length,
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 2)
        .map((x) => x.s);
      const answer = ranked.join(". ");
      return {
        answer,
        retrievedDocIds: [top.id],
        contextChars: answer.length,
      };
    }
    return this.fallback(question, candidates);
  }

  private async fallback(question: string, candidates: string[]): Promise<RunResult> {
    const selectedNodes = candidates.slice(0, 2);
    const allMeta: MetaDocument[] = [];
    for (const nodeId of selectedNodes) {
      const metas = await this.state.metaIndex.search(nodeId, question, 3);
      allMeta.push(...metas);
    }
    const parts: string[] = [];
    const retrievedIds: string[] = [];
    for (const meta of allMeta.slice(0, 3)) {
      const content = await this.state.fetcherRegistry.fetch(meta);
      parts.push(`### ${content.title}\n${BM25BodyExtractor.extract(content.body, question, 2)}`);
      retrievedIds.push(meta.id);
    }
    const context = parts.join("\n\n");
    const prompt = `Answer in 1-2 sentences using only the context.\n\nContext:\n${context}\n\nQuestion: ${question}\nAnswer:`;
    const response = await this.state.chat.invoke(prompt);
    const answer =
      typeof response.content === "string" ? response.content : JSON.stringify(response.content);
    return {
      answer,
      retrievedDocIds: Array.from(new Set(retrievedIds)),
      contextChars: context.length,
    };
  }
}

// ── Ontology improvements (V18-V21) ─────────────────────────

/** V18: BM25 node mapping (IDF-weighted) replaces plain keyword overlap. */
export class KontextV18Bm25Map {
  private state!: SharedState;
  private strategy = new Bm25NodeMappingStrategy();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
  }

  async query(question: string): Promise<RunResult> {
    return directQuery(this.state, this.strategy, question, {
      topNodes: 2,
      topDocsPerNode: 3,
    });
  }
}

/** V19: keyword mapping + MMR document selector for diversity. */
export class KontextV19Mmr {
  private state!: SharedState;
  private strategy = new KeywordMappingStrategy();
  private mmr = new MmrSelector(0.7);

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
  }

  async query(question: string): Promise<RunResult> {
    const candidates = await this.strategy.findStartNodes(question, this.state.graph.nodes);
    const selectedNodes = candidates.slice(0, 2);
    const allMeta: MetaDocument[] = [];
    for (const nodeId of selectedNodes) {
      const metas = await this.state.metaIndex.search(nodeId, question, 6);
      allMeta.push(...metas);
    }
    const top = await this.mmr.select(question, allMeta, 3);
    const parts: string[] = [];
    const retrievedIds: string[] = [];
    for (const meta of top) {
      const content = await this.state.fetcherRegistry.fetch(meta);
      parts.push(`### ${content.title}\n${content.body}`);
      retrievedIds.push(meta.id);
    }
    const context = parts.join("\n\n");
    const prompt = `You are a technical assistant. Answer the question based only on the provided context. Be concise.\n\nContext:\n${context}\n\nQuestion: ${question}\n\nAnswer:`;
    const response = await this.state.chat.invoke(prompt);
    const answer =
      typeof response.content === "string" ? response.content : JSON.stringify(response.content);
    return {
      answer,
      retrievedDocIds: Array.from(new Set(retrievedIds)),
      contextChars: context.length,
    };
  }
}

/** V20: edge-aware expansion — wraps BM25 mapping with edge follow-through. */
export class KontextV20EdgeAware {
  private state!: SharedState;
  private strategy!: EdgeAwareMappingStrategy;

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
    this.strategy = new EdgeAwareMappingStrategy(
      new Bm25NodeMappingStrategy(),
      this.state.graph,
      0.5,
      2,
    );
  }

  async query(question: string): Promise<RunResult> {
    return directQuery(this.state, this.strategy, question, {
      topNodes: 3,
      topDocsPerNode: 2,
    });
  }
}

// ── Entity vocabulary for the bench corpus ─────────────────
const BENCH_ENTITIES: Entity[] = [
  createEntity({ id: "rest", name: "REST", type: "concept", aliases: ["REST API"] }),
  createEntity({ id: "graphql", name: "GraphQL", type: "concept", aliases: ["GQL"] }),
  createEntity({ id: "grpc", name: "gRPC", type: "concept" }),
  createEntity({ id: "jwt", name: "JWT", type: "concept", aliases: ["JSON Web Token"] }),
  createEntity({ id: "postgres", name: "PostgreSQL", type: "tool", aliases: ["Postgres"] }),
  createEntity({ id: "redis", name: "Redis", type: "tool" }),
  createEntity({ id: "kafka", name: "Kafka", type: "tool", aliases: ["Apache Kafka"] }),
  createEntity({ id: "rabbitmq", name: "RabbitMQ", type: "tool" }),
  createEntity({ id: "react", name: "React", type: "framework" }),
  createEntity({ id: "typescript", name: "TypeScript", type: "language", aliases: ["TS"] }),
  createEntity({ id: "docker", name: "Docker", type: "tool" }),
  createEntity({ id: "kubernetes", name: "Kubernetes", type: "tool", aliases: ["K8s"] }),
  createEntity({ id: "terraform", name: "Terraform", type: "tool" }),
  createEntity({ id: "owasp", name: "OWASP", type: "org" }),
  createEntity({ id: "tls", name: "TLS", type: "protocol", aliases: ["Transport Layer Security"] }),
  createEntity({ id: "vault", name: "Vault", type: "tool", aliases: ["HashiCorp Vault"] }),
  createEntity({ id: "dataloader", name: "DataLoader", type: "tool" }),
  createEntity({ id: "web-vitals", name: "Web Vitals", type: "concept", aliases: ["LCP", "FID", "CLS"] }),
  createEntity({ id: "rbac", name: "RBAC", type: "concept", aliases: ["Role-Based Access Control"] }),
  createEntity({ id: "csrf", name: "CSRF", type: "concept", aliases: ["Cross-Site Request Forgery"] }),
  createEntity({ id: "xss", name: "XSS", type: "concept", aliases: ["Cross-Site Scripting"] }),
  createEntity({ id: "etl", name: "ETL", type: "concept", aliases: ["ELT"] }),
  createEntity({ id: "dbt", name: "dbt", type: "tool" }),
  createEntity({ id: "airflow", name: "Airflow", type: "tool" }),
  createEntity({ id: "scd", name: "SCD", type: "concept", aliases: ["slowly changing dimensions"] }),
  createEntity({ id: "feature-store", name: "Feature Store", type: "concept" }),
  createEntity({ id: "red-metrics", name: "RED Metrics", type: "concept", aliases: ["Rate Errors Duration"] }),
  createEntity({ id: "use-metrics", name: "USE Metrics", type: "concept", aliases: ["Utilization Saturation"] }),
];

const BENCH_ENTITY_RELATIONS = [
  { from: "react", to: "typescript", type: "uses", weight: 0.8 },
  { from: "kafka", to: "rabbitmq", type: "alternative_to", weight: 0.7 },
  { from: "jwt", to: "rest", type: "part_of", weight: 0.6 },
  { from: "graphql", to: "dataloader", type: "uses", weight: 0.8 },
  { from: "kubernetes", to: "docker", type: "uses", weight: 0.9 },
  { from: "terraform", to: "kubernetes", type: "provisions", weight: 0.7 },
  { from: "vault", to: "rbac", type: "uses", weight: 0.6 },
  { from: "rbac", to: "owasp", type: "related_to", weight: 0.5 },
];

async function buildEntityIndexForCorpus(
  corpus: BenchDoc[],
): Promise<{ index: InMemoryEntityIndex; metaByDoc: Map<string, MetaDocument> }> {
  const index = new InMemoryEntityIndex();
  for (const e of BENCH_ENTITIES) await index.addEntity(e);
  for (const r of BENCH_ENTITY_RELATIONS) await index.addRelation(r);

  const extractor = new AliasEntityExtractor(BENCH_ENTITIES);
  const metaByDoc = new Map<string, MetaDocument>();
  for (const doc of corpus) {
    const meta = createMetaDocument({
      id: doc.id,
      title: doc.title,
      source: DataSource.CUSTOM,
      ontologyNodeId: "unused",
    });
    metaByDoc.set(doc.id, meta);
    const found = await extractor.extract(`${doc.title}\n${doc.body}`);
    for (const e of found.entities) {
      await index.addMention({ entityId: e.id, docId: doc.id });
    }
  }
  return { index, metaByDoc };
}

/** V24: entity-aware retrieval — ranks docs by entity mentions in the query. */
export class KontextV24Entity {
  private entityIndex!: InMemoryEntityIndex;
  private metaByDoc!: Map<string, MetaDocument>;
  private corpusById = new Map<string, BenchDoc>();

  async index(corpus: BenchDoc[]): Promise<void> {
    const built = await buildEntityIndexForCorpus(corpus);
    this.entityIndex = built.index;
    this.metaByDoc = built.metaByDoc;
    for (const d of corpus) this.corpusById.set(d.id, d);
  }

  async query(question: string): Promise<RunResult> {
    const retriever = new EntityRetriever(this.entityIndex, async () => this.metaByDoc, 1);
    const ranked = await retriever.retrieve(question, 4);
    const retrievedIds: string[] = [];
    const parts: string[] = [];
    for (const { doc } of ranked) {
      const bodyDoc = this.corpusById.get(doc.id);
      if (!bodyDoc) continue;
      retrievedIds.push(doc.id);
      parts.push(`### ${bodyDoc.title}\n${bodyDoc.body}`);
    }
    const context = parts.join("\n\n");
    const answer = context.length > 0
      ? extractBestSentences(context, question, 3)
      : "No relevant entities found in query.";
    return {
      answer,
      retrievedDocIds: retrievedIds,
      contextChars: answer.length,
    };
  }
}

/** V25: entity-aware + full RAG pipeline (entities for L2 candidates, LLM for final answer). */
export class KontextV25EntityLLM {
  private state!: SharedState;
  private entityIndex!: InMemoryEntityIndex;
  private metaByDoc!: Map<string, MetaDocument>;
  private corpusById = new Map<string, BenchDoc>();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
    const built = await buildEntityIndexForCorpus(corpus);
    this.entityIndex = built.index;
    this.metaByDoc = built.metaByDoc;
    for (const d of corpus) this.corpusById.set(d.id, d);
  }

  async query(question: string): Promise<RunResult> {
    const retriever = new EntityRetriever(this.entityIndex, async () => this.metaByDoc, 1);
    const ranked = await retriever.retrieve(question, 3);
    const retrievedIds: string[] = [];
    const parts: string[] = [];
    for (const { doc } of ranked) {
      const bodyDoc = this.corpusById.get(doc.id);
      if (!bodyDoc) continue;
      retrievedIds.push(doc.id);
      parts.push(`### ${bodyDoc.title}\n${BM25BodyExtractor.extract(bodyDoc.body, question, 3)}`);
    }
    const context = parts.join("\n\n");
    const prompt = `You are a technical assistant. Answer based only on the context. Be concise.\n\nContext:\n${context}\n\nQuestion: ${question}\n\nAnswer:`;
    const response = await this.state.chat.invoke(prompt);
    const answer =
      typeof response.content === "string" ? response.content : JSON.stringify(response.content);
    return {
      answer,
      retrievedDocIds: retrievedIds,
      contextChars: context.length,
    };
  }
}

function extractBestSentences(body: string, query: string, topK: number): string {
  const q = query.toLowerCase().split(/\s+/).filter((w) => w.length > 2);
  const sentences = body
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 15);
  const scored = sentences
    .map((s) => {
      const lc = s.toLowerCase();
      const hits = q.filter((w) => lc.includes(w)).length;
      const density = hits / Math.max(s.length / 100, 1);
      return { s, score: hits + density * 0.5 };
    })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map((x) => x.s);
  return scored.join(". ");
}

/** V22: BM25 mapping + MMR + LLM rerank + BM25 body compress. */
export class KontextV22Compose {
  private state!: SharedState;
  private strategy = new Bm25NodeMappingStrategy();
  private mmr = new MmrSelector(0.7);

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
  }

  async query(question: string): Promise<RunResult> {
    const candidates = await this.strategy.findStartNodes(question, this.state.graph.nodes);
    const selectedNodes = candidates.slice(0, 3);
    const allMeta: MetaDocument[] = [];
    for (const nodeId of selectedNodes) {
      const metas = await this.state.metaIndex.search(nodeId, question, 4);
      allMeta.push(...metas);
    }
    const unique = Array.from(new Map(allMeta.map((m) => [m.id, m])).values());
    const top = await this.mmr.select(question, unique, 3);

    const parts: string[] = [];
    const retrievedIds: string[] = [];
    for (const meta of top) {
      const content = await this.state.fetcherRegistry.fetch(meta);
      const body = BM25BodyExtractor.extract(content.body, question, 3);
      parts.push(`### ${content.title}\n${body}`);
      retrievedIds.push(meta.id);
    }
    const context = parts.join("\n\n");
    const prompt = `Answer in 1-2 sentences using only the context.\n\nContext:\n${context}\n\nQuestion: ${question}\nAnswer:`;
    const response = await this.state.chat.invoke(prompt);
    const answer =
      typeof response.content === "string" ? response.content : JSON.stringify(response.content);
    return {
      answer,
      retrievedDocIds: Array.from(new Set(retrievedIds)),
      contextChars: context.length,
    };
  }
}

/** V23: BM25 mapping + extractive (no LLM). v16 with BM25 routing. */
export class KontextV23Bm25Extractive {
  private state!: SharedState;
  private strategy = new Bm25NodeMappingStrategy();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
  }

  async query(question: string): Promise<RunResult> {
    const candidates = await this.strategy.findStartNodes(question, this.state.graph.nodes);
    const selectedNodes = candidates.slice(0, 2);
    const allMeta: MetaDocument[] = [];
    for (const nodeId of selectedNodes) {
      const metas = await this.state.metaIndex.search(nodeId, question, 3);
      allMeta.push(...metas);
    }
    const q = question.toLowerCase().split(/\s+/).filter((w) => w.length > 2);
    const retrievedIds = new Set<string>();
    type Scored = { sentence: string; score: number };
    const scored: Scored[] = [];
    for (const meta of allMeta) {
      const content = await this.state.fetcherRegistry.fetch(meta);
      retrievedIds.add(meta.id);
      const sentences = content.body
        .split(/(?<=[.!?])\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 15);
      for (const s of sentences) {
        const lc = s.toLowerCase();
        const hits = q.filter((w) => lc.includes(w)).length;
        if (hits === 0) continue;
        const density = hits / Math.max(s.length / 100, 1);
        scored.push({ sentence: s, score: hits + density * 0.5 });
      }
    }
    scored.sort((a, b) => b.score - a.score);
    const answer = scored.slice(0, 3).map((x) => x.sentence).join(". ");
    return {
      answer,
      retrievedDocIds: Array.from(retrievedIds),
      contextChars: answer.length,
    };
  }
}

/** V21: centroid-refined embeddings + vector mapping. */
export class KontextV21Centroid {
  private state!: SharedState;
  private strategy!: VectorMappingStrategy;
  private refined = false;

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
    // Refine node embeddings using doc centroids
    const refiner = new CentroidNodeEmbedder(
      this.state.vectorStore,
      this.state.metaIndex,
      this.state.fetcherRegistry,
      0.7,
      10,
      800,
    );
    await refiner.refine(this.state.graph);
    this.refined = true;
    this.strategy = new VectorMappingStrategy(this.state.vectorStore);
  }

  async query(question: string): Promise<RunResult> {
    if (!this.refined) throw new Error("call index() first");
    return directQuery(this.state, this.strategy, question, {
      topNodes: 2,
      topDocsPerNode: 3,
    });
  }
}

/**
 * V16 Proximity-ranked extractive: score sentences by query-term density,
 * addressing v12's q5 miss where the right words were scattered.
 */
export class KontextV16Proximity {
  private state!: SharedState;
  private strategy = new KeywordMappingStrategy();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
  }

  async query(question: string): Promise<RunResult> {
    const candidates = await this.strategy.findStartNodes(question, this.state.graph.nodes);
    const selectedNodes = candidates.slice(0, 2);
    const allMeta: MetaDocument[] = [];
    for (const nodeId of selectedNodes) {
      const metas = await this.state.metaIndex.search(nodeId, question, 3);
      allMeta.push(...metas);
    }
    const q = question.toLowerCase().split(/\s+/).filter((w) => w.length > 2);
    const retrievedIds = new Set<string>();

    type Scored = { sentence: string; score: number };
    const scored: Scored[] = [];
    for (const meta of allMeta) {
      const content = await this.state.fetcherRegistry.fetch(meta);
      retrievedIds.add(meta.id);
      const sentences = content.body
        .split(/(?<=[.!?])\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 15);
      for (const s of sentences) {
        const lc = s.toLowerCase();
        const hits = q.filter((w) => lc.includes(w)).length;
        if (hits === 0) continue;
        const density = hits / Math.max(s.length / 100, 1);
        scored.push({ sentence: s, score: hits + density * 0.5 });
      }
    }
    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, 3).map((x) => x.sentence);
    const answer = top.join(". ");
    return {
      answer,
      retrievedDocIds: Array.from(retrievedIds),
      contextChars: answer.length,
    };
  }
}

/**
 * V17 Hybrid extractive + LLM fallback: extractive if top-doc title strongly
 * matches query; otherwise v13-sentence path.
 */
export class KontextV17HybridExtract {
  private state!: SharedState;
  private strategy = new KeywordMappingStrategy();
  private v13!: KontextV13SentenceLevel;

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
    this.v13 = new KontextV13SentenceLevel();
    await this.v13.index(corpus);
  }

  async query(question: string): Promise<RunResult> {
    const candidates = await this.strategy.findStartNodes(question, this.state.graph.nodes);
    const topNode = candidates[0];
    if (!topNode) return this.v13.query(question);
    const metas = await this.state.metaIndex.search(topNode, question, 3);
    const q = question.toLowerCase().split(/\s+/).filter((w) => w.length > 2);
    const top = metas[0];
    if (!top) return this.v13.query(question);
    const titleOverlap =
      q.filter((w) => top.title.toLowerCase().includes(w)).length / Math.max(q.length, 1);

    if (titleOverlap < 0.3) return this.v13.query(question);

    const content = await this.state.fetcherRegistry.fetch(top);
    const sentences = content.body
      .split(/(?<=[.!?])\s+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 15);
    const ranked = sentences
      .map((s) => ({
        s,
        score: q.filter((w) => s.toLowerCase().includes(w)).length,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 2)
      .map((x) => x.s);
    const answer = ranked.join(". ");
    return { answer, retrievedDocIds: [top.id], contextChars: answer.length };
  }
}

/**
 * Control: "flat" retrieval — no ontology. Equivalent to putting all meta docs
 * in one bucket and keyword-scoring them. Compares what the ontology layer buys us.
 */
export class FlatKeywordControl {
  private state!: SharedState;
  private allDocs: MetaDocument[] = [];

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSharedState(corpus);
    for (const d of corpus) {
      this.allDocs.push(
        createMetaDocument({
          id: d.id,
          title: d.title,
          source: DataSource.CUSTOM,
          ontologyNodeId: "flat",
        }),
      );
    }
  }

  async query(question: string): Promise<RunResult> {
    const q = question.toLowerCase();
    const qWords = new Set(q.split(/\s+/).filter((w) => w.length > 1));
    const scored = this.allDocs
      .map((d) => {
        const tw = d.title.toLowerCase().split(/\s+/);
        let overlap = 0;
        for (const w of tw) if (qWords.has(w)) overlap++;
        return { doc: d, score: overlap };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, 4);

    const parts: string[] = [];
    const retrievedIds: string[] = [];
    for (const { doc } of scored) {
      const content = await this.state.fetcherRegistry.fetch(doc);
      parts.push(`### ${content.title}\n${content.body}`);
      retrievedIds.push(doc.id);
    }
    const context = parts.join("\n\n");
    const prompt = `You are a technical assistant. Answer the question based only on the provided context. Be concise.\n\nContext:\n${context}\n\nQuestion: ${question}\n\nAnswer:`;
    const response = await this.state.chat.invoke(prompt);
    const answer =
      typeof response.content === "string" ? response.content : JSON.stringify(response.content);
    return { answer, retrievedDocIds: Array.from(new Set(retrievedIds)), contextChars: context.length };
  }
}
