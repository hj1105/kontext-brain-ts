/**
 * Adapts kontext variants to run on a SQuAD-style corpus.
 *
 * SQuAD has no hand-crafted ontology, so we auto-build one from article
 * titles (each unique Wikipedia article becomes a top-level node, and docs
 * are assigned to their article's node). This tests whether the core
 * retrieval machinery transfers from the tech-docs bench.
 */

import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import {
  BM25BodyExtractor,
  Bm25NodeMappingStrategy,
  ContentFetcherRegistry,
  DataSource,
  EntityRetriever,
  HybridRetriever,
  InMemoryEntityIndex,
  InMemoryMetaIndexStore,
  KeywordMappingStrategy,
  OntologyGraph,
  TraversalStrategy,
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

interface RunResult {
  answer: string;
  retrievedDocIds: string[];
  contextChars: number;
}

interface SquadState {
  chat: ChatOllama;
  embeddings: OllamaEmbeddings;
  vectorStore: LangChainVectorStore;
  graph: OntologyGraph;
  metaIndex: InMemoryMetaIndexStore;
  fetcherRegistry: ContentFetcherRegistry;
  corpusById: Map<string, BenchDoc>;
  docToNode: Map<string, string>;
}

/** Derive node id from SQuAD doc id: "normans-2" → "normans". */
function nodeForDoc(docId: string): string {
  const idx = docId.lastIndexOf("-");
  return idx >= 0 ? docId.slice(0, idx) : docId;
}

async function buildSquadState(corpus: BenchDoc[]): Promise<SquadState> {
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
  const vectorStore = new LangChainVectorStore(embeddings);

  // Auto-build ontology: one node per unique article
  const docToNode = new Map<string, string>();
  const nodeDocs = new Map<string, BenchDoc[]>();
  for (const doc of corpus) {
    const nodeId = nodeForDoc(doc.id);
    docToNode.set(doc.id, nodeId);
    const list = nodeDocs.get(nodeId) ?? [];
    list.push(doc);
    nodeDocs.set(nodeId, list);
  }

  const nodes = new Map<string, OntologyNode>();
  for (const [nodeId, docs] of nodeDocs) {
    // Description = concatenation of first 100 chars of each doc in this node
    const desc = docs
      .map((d) => `${d.title}: ${d.body.slice(0, 100)}`)
      .join(" ")
      .slice(0, 500);
    nodes.set(
      nodeId,
      createNode({ id: nodeId, description: desc, weight: 1.0 }),
    );
  }
  const graph = new OntologyGraph(nodes, [], {
    maxDepth: 2,
    maxTokens: 4000,
    strategy: TraversalStrategy.WEIGHTED_DFS,
  });

  // Embed node descriptions
  for (const node of graph.nodes.values()) {
    const v = await vectorStore.embed(node.description);
    await vectorStore.upsert(node.id, v);
  }

  // Meta index: assign each doc to its node
  const metaIndex = new InMemoryMetaIndexStore();
  for (const [nodeId, docs] of nodeDocs) {
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

  // Fetcher backed by corpus map
  const corpusById = new Map<string, BenchDoc>();
  for (const d of corpus) corpusById.set(d.id, d);
  const fetcherRegistry = new ContentFetcherRegistry();
  const fetcher: ContentFetcher = {
    source: DataSource.CUSTOM,
    async fetch(meta: MetaDocument) {
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

  return { chat, embeddings, vectorStore, graph, metaIndex, fetcherRegistry, corpusById, docToNode };
}

async function directQueryWithBody(
  state: SquadState,
  mapping: NodeMappingStrategy,
  question: string,
  topNodes = 2,
  topDocs = 2,
): Promise<RunResult> {
  const candidates = await mapping.findStartNodes(question, state.graph.nodes);
  const selectedNodes = candidates.slice(0, topNodes);
  const allMeta: MetaDocument[] = [];
  for (const nodeId of selectedNodes) {
    const metas = await state.metaIndex.search(nodeId, question, topDocs);
    allMeta.push(...metas);
  }
  const parts: string[] = [];
  const retrievedIds: string[] = [];
  for (const meta of allMeta) {
    const content = await state.fetcherRegistry.fetch(meta);
    parts.push(`### ${content.title}\n${BM25BodyExtractor.extract(content.body, question, 3)}`);
    retrievedIds.push(meta.id);
  }
  const context = parts.join("\n\n");
  const prompt = `Answer the question in one short sentence, using only the context.\n\nContext:\n${context}\n\nQuestion: ${question}\nAnswer:`;
  const response = await state.chat.invoke(prompt);
  const answer =
    typeof response.content === "string" ? response.content : JSON.stringify(response.content);
  return { answer, retrievedDocIds: Array.from(new Set(retrievedIds)), contextChars: context.length };
}

// ── Baseline: standard vector RAG over chunks ────────────────

export class SquadBaselineRAG {
  private chunks: Array<{ docId: string; title: string; text: string; embedding: number[] }> = [];
  private readonly chat: ChatOllama;
  private readonly embeddings: OllamaEmbeddings;

  constructor() {
    const baseUrl = "http://localhost:11434";
    this.chat = new ChatOllama({ baseUrl, model: "qwen2.5:1.5b", temperature: 0, numGpu: 0, numCtx: 2048, numPredict: 256 });
    this.embeddings = new OllamaEmbeddings({ baseUrl, model: "nomic-embed-text" });
  }

  async index(corpus: BenchDoc[]): Promise<void> {
    const texts: Array<{ docId: string; title: string; text: string }> = [];
    for (const doc of corpus) {
      // Chunk at 400 chars with 50 overlap
      const size = 400;
      const overlap = 50;
      for (let i = 0; i < doc.body.length; i += size - overlap) {
        texts.push({ docId: doc.id, title: doc.title, text: doc.body.slice(i, i + size) });
        if (i + size >= doc.body.length) break;
      }
    }
    const vectors = await this.embeddings.embedDocuments(texts.map((x) => x.text));
    this.chunks = texts.map((x, i) => ({ ...x, embedding: vectors[i] ?? [] }));
  }

  async query(question: string): Promise<RunResult> {
    const qv = await this.embeddings.embedQuery(question);
    const scored = this.chunks.map((c) => ({ c, score: cosine(qv, c.embedding) }));
    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, 4).map((x) => x.c);
    const context = top.map((c) => `### ${c.title}\n${c.text}`).join("\n\n");
    const prompt = `Answer the question in one short sentence, using only the context.\n\nContext:\n${context}\n\nQuestion: ${question}\nAnswer:`;
    const res = await this.chat.invoke(prompt);
    const answer = typeof res.content === "string" ? res.content : JSON.stringify(res.content);
    return { answer, retrievedDocIds: Array.from(new Set(top.map((c) => c.docId))), contextChars: context.length };
  }
}

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += (a[i] ?? 0) * (b[i] ?? 0);
    na += (a[i] ?? 0) ** 2;
    nb += (b[i] ?? 0) ** 2;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ── kontext variants for SQuAD ────────────────────────────────

/** V2-keyword equivalent — keyword node routing + LLM answer */
export class SquadKontextKeyword {
  private state!: SquadState;
  async index(corpus: BenchDoc[]): Promise<void> { this.state = await buildSquadState(corpus); }
  async query(q: string): Promise<RunResult> { return directQueryWithBody(this.state, new KeywordMappingStrategy(), q); }
}

/** V13 equivalent — BM25 mapping + compressed answer */
export class SquadKontextBm25 {
  private state!: SquadState;
  async index(corpus: BenchDoc[]): Promise<void> { this.state = await buildSquadState(corpus); }
  async query(q: string): Promise<RunResult> { return directQueryWithBody(this.state, new Bm25NodeMappingStrategy(), q); }
}

/** V16 equivalent — proximity extractive (no LLM) */
export class SquadKontextExtractive {
  private state!: SquadState;
  private strategy = new Bm25NodeMappingStrategy();
  async index(corpus: BenchDoc[]): Promise<void> { this.state = await buildSquadState(corpus); }

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
    const answer = scored.slice(0, 2).map((x) => x.sentence).join(". ");
    return { answer, retrievedDocIds: Array.from(retrievedIds), contextChars: answer.length };
  }
}

/**
 * Extract entity vocabulary from a corpus: capitalized phrases (proper nouns)
 * + TF-IDF-ranked common-noun phrases. Covers SQuAD-style queries like
 * "biomass", "complexity theory" where proper-noun-only extractors fail.
 */
function extractCorpusVocab(corpus: BenchDoc[]): Entity[] {
  // Proper nouns (capitalized phrases) — high confidence
  const properNouns = new Map<string, Set<string>>();
  for (const doc of corpus) {
    for (const m of doc.body.matchAll(/\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+){0,3})\b/g)) {
      const p = m[1]!.trim();
      if (p.length > 50) continue;
      const s = properNouns.get(p) ?? new Set<string>();
      s.add(doc.id);
      properNouns.set(p, s);
    }
  }

  // Common noun phrases (2-3 content words). Rough heuristic: bigrams/
  // trigrams excluding stopwords, ranked by document frequency.
  const STOP = new Set([
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "this", "that", "these", "those", "it", "its",
    "they", "them", "their", "there", "where", "when", "what", "which", "who",
    "whom", "whose", "how", "why", "with", "without", "from", "into", "onto",
    "over", "under", "above", "below", "up", "down", "out", "in", "on", "at",
    "to", "of", "for", "by", "as", "if", "than", "then", "so", "not", "no", "yes",
    "all", "any", "each", "every", "some", "many", "much", "few", "more", "most",
    "other", "another", "same", "such", "own", "only", "just", "also", "very",
    "still", "well", "one", "two", "three", "four", "five", "first", "second",
    "third", "last", "new", "old", "about", "after", "before", "during", "while",
    "between", "among", "through", "because", "although", "though",
  ]);
  const phraseDocFreq = new Map<string, Set<string>>();
  for (const doc of corpus) {
    const tokens = doc.body
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((w) => w.length > 2 && !STOP.has(w));
    for (let i = 0; i < tokens.length; i++) {
      // unigram meaningful noun-like
      const uni = tokens[i]!;
      if (uni.length >= 4) {
        const set = phraseDocFreq.get(uni) ?? new Set<string>();
        set.add(doc.id);
        phraseDocFreq.set(uni, set);
      }
      // bigram
      if (i + 1 < tokens.length) {
        const bi = `${tokens[i]} ${tokens[i + 1]}`;
        const set = phraseDocFreq.get(bi) ?? new Set<string>();
        set.add(doc.id);
        phraseDocFreq.set(bi, set);
      }
    }
  }

  const totalDocs = corpus.length;
  const commonVocab: Array<{ phrase: string; idf: number; df: number }> = [];
  for (const [phrase, docs] of phraseDocFreq) {
    const df = docs.size;
    if (df < 1 || df > totalDocs * 0.5) continue; // skip too common / too rare
    const idf = Math.log(totalDocs / df);
    commonVocab.push({ phrase, idf, df });
  }
  // Keep top ~1000 by IDF * sqrt(df) — favors discriminative mid-frequency phrases
  commonVocab.sort((a, b) => b.idf * Math.sqrt(b.df) - a.idf * Math.sqrt(a.df));
  const topCommon = commonVocab.slice(0, 800);

  // Merge: proper nouns first, then common phrases (dedup by normalized form)
  const vocab: Entity[] = [];
  const seen = new Set<string>();
  for (const [phrase, docSet] of properNouns) {
    const id = phrase.toLowerCase().replace(/\s+/g, "-");
    if (seen.has(id)) continue;
    seen.add(id);
    vocab.push(createEntity({
      id,
      name: phrase,
      type: "proper_noun",
      weight: Math.min(1.0, docSet.size / 3),
    }));
  }
  for (const { phrase, df } of topCommon) {
    const id = phrase.replace(/\s+/g, "-");
    if (seen.has(id)) continue;
    seen.add(id);
    vocab.push(createEntity({
      id,
      name: phrase,
      type: "noun_phrase",
      weight: Math.min(0.8, df / 5),
    }));
  }
  return vocab.slice(0, 1500);
}

/**
 * V24 equivalent — auto-extracts entities from the SQuAD corpus (capitalized
 * sequences as candidate entities, dedup across docs) and retrieves by entity
 * mention overlap with the question.
 */
export class SquadKontextEntity {
  private entityIndex!: InMemoryEntityIndex;
  private metaByDoc!: Map<string, MetaDocument>;
  private corpusById = new Map<string, BenchDoc>();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.entityIndex = new InMemoryEntityIndex();
    this.metaByDoc = new Map();
    for (const d of corpus) this.corpusById.set(d.id, d);

    // Auto-extract candidate entities: capitalized word sequences (proper nouns),
    // filtered to those appearing in >=2 docs (reduces noise).
    const mentionCount = new Map<string, Set<string>>();
    for (const doc of corpus) {
      for (const match of doc.body.matchAll(/\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+){0,3})\b/g)) {
        const phrase = match[1]!;
        const norm = phrase.trim();
        const set = mentionCount.get(norm) ?? new Set<string>();
        set.add(doc.id);
        mentionCount.set(norm, set);
      }
    }

    // Keep phrases mentioned in at least one doc; prioritize those in multiple
    const vocab: Entity[] = [];
    for (const [phrase, docSet] of mentionCount) {
      if (docSet.size < 1) continue;
      if (phrase.length > 50) continue; // skip overly long matches
      vocab.push(createEntity({
        id: phrase.toLowerCase().replace(/\s+/g, "-"),
        name: phrase,
        type: "proper_noun",
        weight: Math.min(1.0, docSet.size / 5),
      }));
    }
    // Cap vocab for performance
    vocab.sort((a, b) => (b.weight ?? 0) - (a.weight ?? 0));
    const top = vocab.slice(0, 500);
    for (const e of top) await this.entityIndex.addEntity(e);

    // Create meta docs + index mentions
    for (const doc of corpus) {
      const meta = createMetaDocument({
        id: doc.id,
        title: doc.title,
        source: DataSource.CUSTOM,
        ontologyNodeId: "squad",
      });
      this.metaByDoc.set(doc.id, meta);
      const foundEntities = await this.entityIndex.findEntitiesInText(`${doc.title}\n${doc.body}`);
      for (const e of foundEntities) {
        await this.entityIndex.addMention({ entityId: e.id, docId: doc.id });
      }
    }
  }

  async query(question: string): Promise<RunResult> {
    const retriever = new EntityRetriever(this.entityIndex, async () => this.metaByDoc, 0);
    const ranked = await retriever.retrieve(question, 3);
    if (ranked.length === 0) {
      return { answer: "No entity matches.", retrievedDocIds: [], contextChars: 0 };
    }
    const q = question.toLowerCase().split(/\s+/).filter((w) => w.length > 2);
    type Scored = { sentence: string; score: number };
    const scored: Scored[] = [];
    const retrievedIds: string[] = [];
    for (const { doc } of ranked) {
      const bodyDoc = this.corpusById.get(doc.id);
      if (!bodyDoc) continue;
      retrievedIds.push(doc.id);
      const sentences = bodyDoc.body
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
    const answer = scored.slice(0, 2).map((x) => x.sentence).join(". ");
    return { answer, retrievedDocIds: retrievedIds, contextChars: answer.length };
  }
}

/**
 * Hybrid retriever — vector similarity + entity matching ensemble.
 * Designed to close the gap to baseline vector RAG on generic QA.
 *
 *   1. Index: embed each doc body via Ollama, store under key "doc:{id}".
 *      Extract proper-noun + IDF-ranked common-noun entities.
 *   2. Query: combine vector top-K and entity matches via weighted score,
 *      send top 3 docs (BM25-compressed) to LLM for final answer.
 */
export class SquadKontextHybrid {
  private entityIndex!: InMemoryEntityIndex;
  private metaByDoc!: Map<string, MetaDocument>;
  private corpusById = new Map<string, BenchDoc>();
  private state!: SquadState;

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSquadState(corpus);
    this.entityIndex = new InMemoryEntityIndex();
    this.metaByDoc = new Map();
    for (const d of corpus) this.corpusById.set(d.id, d);

    // Build richer entity vocabulary: proper nouns + common noun phrases
    const vocab = extractCorpusVocab(corpus);
    for (const e of vocab) await this.entityIndex.addEntity(e);

    for (const doc of corpus) {
      const meta = createMetaDocument({
        id: doc.id,
        title: doc.title,
        source: DataSource.CUSTOM,
        ontologyNodeId: "squad",
      });
      this.metaByDoc.set(doc.id, meta);
      const found = await this.entityIndex.findEntitiesInText(`${doc.title}\n${doc.body}`);
      for (const e of found) {
        await this.entityIndex.addMention({ entityId: e.id, docId: doc.id });
      }

      // Embed doc for vector similarity. Use full body truncated.
      const embedText = `${doc.title}\n${doc.body.slice(0, 1500)}`;
      const vec = await this.state.vectorStore.embed(embedText);
      await this.state.vectorStore.upsert(`doc:${doc.id}`, vec, { docId: doc.id });
    }
  }

  async query(question: string): Promise<RunResult> {
    const hybrid = new HybridRetriever(
      this.entityIndex,
      this.state.vectorStore,
      async () => this.metaByDoc,
      0.4, // entity weight
      0.6, // vector weight
      1,   // expansion depth
      "doc:",
      20,
    );
    const ranked = await hybrid.retrieve(question, 3);
    const retrievedIds: string[] = [];
    const parts: string[] = [];
    for (const { doc } of ranked) {
      const bodyDoc = this.corpusById.get(doc.id);
      if (!bodyDoc) continue;
      retrievedIds.push(doc.id);
      parts.push(`### ${bodyDoc.title}\n${BM25BodyExtractor.extract(bodyDoc.body, question, 3)}`);
    }
    const context = parts.join("\n\n");
    const prompt = `Answer the question in one short sentence, using only the context.\n\nContext:\n${context}\n\nQuestion: ${question}\nAnswer:`;
    const response = await this.state.chat.invoke(prompt);
    const answer =
      typeof response.content === "string" ? response.content : JSON.stringify(response.content);
    return { answer, retrievedDocIds: retrievedIds, contextChars: context.length };
  }
}

/**
 * Pure vector baseline with doc-level embeddings (not chunks) — for fair
 * comparison against hybrid's vector component.
 */
export class SquadKontextVectorDocs {
  private state!: SquadState;
  private corpusById = new Map<string, BenchDoc>();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSquadState(corpus);
    for (const d of corpus) this.corpusById.set(d.id, d);
    for (const doc of corpus) {
      const embedText = `${doc.title}\n${doc.body.slice(0, 1500)}`;
      const vec = await this.state.vectorStore.embed(embedText);
      await this.state.vectorStore.upsert(`doc:${doc.id}`, vec, { docId: doc.id });
    }
  }

  async query(question: string): Promise<RunResult> {
    const hits = await this.state.vectorStore.similaritySearchWithPrefix(question, "doc:", 3);
    const retrievedIds: string[] = [];
    const parts: string[] = [];
    for (const docId of hits) {
      const bodyDoc = this.corpusById.get(docId);
      if (!bodyDoc) continue;
      retrievedIds.push(docId);
      parts.push(`### ${bodyDoc.title}\n${BM25BodyExtractor.extract(bodyDoc.body, question, 3)}`);
    }
    const context = parts.join("\n\n");
    const prompt = `Answer the question in one short sentence, using only the context.\n\nContext:\n${context}\n\nQuestion: ${question}\nAnswer:`;
    const response = await this.state.chat.invoke(prompt);
    const answer =
      typeof response.content === "string" ? response.content : JSON.stringify(response.content);
    return { answer, retrievedDocIds: retrievedIds, contextChars: context.length };
  }
}

// ── Round 10 improvements ────────────────────────────────────

/**
 * Ensemble retriever: union of hybrid + baseline vector top-K, then
 * single LLM answer on merged context.
 *
 * Intent: recover cases where baseline found the right paragraph but
 * hybrid didn't (and vice versa). Trades some context size for coverage.
 */
export class SquadKontextEnsemble {
  private entityIndex!: InMemoryEntityIndex;
  private metaByDoc!: Map<string, MetaDocument>;
  private corpusById = new Map<string, BenchDoc>();
  private state!: SquadState;

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSquadState(corpus);
    this.entityIndex = new InMemoryEntityIndex();
    this.metaByDoc = new Map();
    for (const d of corpus) this.corpusById.set(d.id, d);

    const vocab = extractCorpusVocab(corpus);
    for (const e of vocab) await this.entityIndex.addEntity(e);

    for (const doc of corpus) {
      const meta = createMetaDocument({
        id: doc.id, title: doc.title, source: DataSource.CUSTOM, ontologyNodeId: "squad",
      });
      this.metaByDoc.set(doc.id, meta);
      const found = await this.entityIndex.findEntitiesInText(`${doc.title}\n${doc.body}`);
      for (const e of found) await this.entityIndex.addMention({ entityId: e.id, docId: doc.id });
      const embedText = `${doc.title}\n${doc.body.slice(0, 1500)}`;
      const vec = await this.state.vectorStore.embed(embedText);
      await this.state.vectorStore.upsert(`doc:${doc.id}`, vec, { docId: doc.id });
    }
  }

  async query(question: string): Promise<RunResult> {
    const hybridRetriever = new HybridRetriever(
      this.entityIndex, this.state.vectorStore, async () => this.metaByDoc, 0.4, 0.6, 1, "doc:", 20,
    );
    const hybridTop = await hybridRetriever.retrieve(question, 3);
    const vectorOnly = await this.state.vectorStore.similaritySearchWithPrefix(question, "doc:", 3);
    const ids = new Set<string>();
    const retrievedIds: string[] = [];
    const parts: string[] = [];
    for (const { doc } of hybridTop) {
      if (ids.has(doc.id)) continue;
      ids.add(doc.id);
      const bd = this.corpusById.get(doc.id);
      if (!bd) continue;
      retrievedIds.push(doc.id);
      parts.push(`### ${bd.title}\n${BM25BodyExtractor.extract(bd.body, question, 3)}`);
    }
    for (const docId of vectorOnly) {
      if (ids.has(docId)) continue;
      ids.add(docId);
      const bd = this.corpusById.get(docId);
      if (!bd) continue;
      retrievedIds.push(docId);
      parts.push(`### ${bd.title}\n${BM25BodyExtractor.extract(bd.body, question, 3)}`);
    }
    const context = parts.join("\n\n");
    const prompt = `Answer the question in one short sentence, using only the context.\n\nContext:\n${context}\n\nQuestion: ${question}\nAnswer:`;
    const response = await this.state.chat.invoke(prompt);
    const answer =
      typeof response.content === "string" ? response.content : JSON.stringify(response.content);
    return { answer, retrievedDocIds: retrievedIds, contextChars: context.length };
  }
}

/**
 * Extract-then-answer: 2-stage LLM. First extracts the single most relevant
 * sentence from the retrieved context; second generates the concise answer
 * from only that sentence. Reduces LLM distraction from surrounding text.
 */
export class SquadKontextExtractThenAnswer {
  private entityIndex!: InMemoryEntityIndex;
  private metaByDoc!: Map<string, MetaDocument>;
  private corpusById = new Map<string, BenchDoc>();
  private state!: SquadState;

  async index(corpus: BenchDoc[]): Promise<void> {
    this.state = await buildSquadState(corpus);
    this.entityIndex = new InMemoryEntityIndex();
    this.metaByDoc = new Map();
    for (const d of corpus) this.corpusById.set(d.id, d);
    const vocab = extractCorpusVocab(corpus);
    for (const e of vocab) await this.entityIndex.addEntity(e);
    for (const doc of corpus) {
      const meta = createMetaDocument({
        id: doc.id, title: doc.title, source: DataSource.CUSTOM, ontologyNodeId: "squad",
      });
      this.metaByDoc.set(doc.id, meta);
      const found = await this.entityIndex.findEntitiesInText(`${doc.title}\n${doc.body}`);
      for (const e of found) await this.entityIndex.addMention({ entityId: e.id, docId: doc.id });
      const embedText = `${doc.title}\n${doc.body.slice(0, 1500)}`;
      const vec = await this.state.vectorStore.embed(embedText);
      await this.state.vectorStore.upsert(`doc:${doc.id}`, vec, { docId: doc.id });
    }
  }

  async query(question: string): Promise<RunResult> {
    const hybrid = new HybridRetriever(
      this.entityIndex, this.state.vectorStore, async () => this.metaByDoc, 0.4, 0.6, 1, "doc:", 20,
    );
    const ranked = await hybrid.retrieve(question, 3);
    const retrievedIds: string[] = [];
    const parts: string[] = [];
    for (const { doc } of ranked) {
      const bd = this.corpusById.get(doc.id);
      if (!bd) continue;
      retrievedIds.push(doc.id);
      parts.push(`### ${bd.title}\n${bd.body.slice(0, 1200)}`);
    }
    const fullContext = parts.join("\n\n");

    // Stage 1: extract the single most relevant sentence
    const extractPrompt = `From the context, copy the single sentence that best answers the question. Output only that sentence, nothing else.\n\nContext:\n${fullContext}\n\nQuestion: ${question}\nSentence:`;
    const extractRes = await this.state.chat.invoke(extractPrompt);
    const extracted =
      typeof extractRes.content === "string" ? extractRes.content.trim() : JSON.stringify(extractRes.content);

    // Stage 2: concise answer from the extracted sentence
    const answerPrompt = `From this single sentence, answer the question in as few words as possible.\n\nSentence: ${extracted}\n\nQuestion: ${question}\nAnswer:`;
    const answerRes = await this.state.chat.invoke(answerPrompt);
    const answer =
      typeof answerRes.content === "string" ? answerRes.content.trim() : JSON.stringify(answerRes.content);
    return { answer, retrievedDocIds: retrievedIds, contextChars: extracted.length };
  }
}

/** Factory to build a SquadKontextHybrid with a specific chat model name. */
export function buildHybridWithModel(modelName: string): SquadKontextHybrid {
  const h = new SquadKontextHybrid();
  // Swap the chat model after index (hacky but works for bench)
  const originalIndex = h.index.bind(h);
  h.index = async (corpus: BenchDoc[]) => {
    await originalIndex(corpus);
    const baseUrl = "http://localhost:11434";
    (h as unknown as { state: SquadState }).state.chat = new ChatOllama({
      baseUrl, model: modelName, temperature: 0, numGpu: 0, numCtx: 2048, numPredict: 256,
    });
  };
  return h;
}

/**
 * Answer-selection ensemble: runs hybrid (1.5b) and hybrid-3b in parallel,
 * then asks a third LLM call to pick the better answer given the retrieved
 * context. Targets 90% accuracy ceiling on SQuAD (oracle of the two).
 */
export class SquadKontextAnswerEnsemble {
  private hybridSmall!: SquadKontextHybrid;
  private hybridLarge!: SquadKontextHybrid;
  private judge!: ChatOllama;
  private corpusById = new Map<string, BenchDoc>();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.hybridSmall = new SquadKontextHybrid();
    this.hybridLarge = buildHybridWithModel("qwen2.5:3b");
    await this.hybridSmall.index(corpus);
    await this.hybridLarge.index(corpus);
    for (const d of corpus) this.corpusById.set(d.id, d);
    // Use the 3b model as judge
    this.judge = new ChatOllama({
      baseUrl: "http://localhost:11434",
      model: "qwen2.5:3b",
      temperature: 0,
      numGpu: 0,
      numCtx: 2048,
      numPredict: 64,
    });
  }

  async query(question: string): Promise<RunResult> {
    const [sm, lg] = await Promise.all([
      this.hybridSmall.query(question),
      this.hybridLarge.query(question),
    ]);
    // If both answers are the same, trust and return the small one (faster latency attribution)
    if (sm.answer.trim() === lg.answer.trim()) return sm;

    // Gather retrieved doc bodies (union) for the judge to ground on
    const ids = Array.from(new Set([...sm.retrievedDocIds, ...lg.retrievedDocIds]));
    const context = ids
      .slice(0, 4)
      .map((id) => {
        const d = this.corpusById.get(id);
        return d ? `### ${d.title}\n${d.body.slice(0, 500)}` : "";
      })
      .filter(Boolean)
      .join("\n\n");

    const judgePrompt = `Given the context, which answer is correct for the question? Respond with ONLY "A" or "B", nothing else.\n\nContext:\n${context}\n\nQuestion: ${question}\n\nAnswer A: ${sm.answer}\nAnswer B: ${lg.answer}\n\nWhich is correct?`;
    const verdict = await this.judge.invoke(judgePrompt);
    const verdictText =
      typeof verdict.content === "string" ? verdict.content : JSON.stringify(verdict.content);
    const pickB = /\bB\b/.test(verdictText.trim().slice(0, 20));
    const chosen = pickB ? lg : sm;
    return {
      answer: chosen.answer,
      retrievedDocIds: chosen.retrievedDocIds,
      contextChars: chosen.contextChars,
    };
  }
}

/**
 * Extractive-on-hybrid: runs the hybrid retriever to find top docs, then
 * extracts the single best sentence from those docs (no LLM). Useful as a
 * 3rd synthesis candidate for queries where the LLM compresses away the
 * actual answer sentence (e.g. sq-14 aerobic which full-text extractive
 * solved in Round 7 but hybrid's BM25-compressed context lost).
 */
export class SquadKontextExtractiveOnHybrid {
  private hybrid!: SquadKontextHybrid;
  private corpusById = new Map<string, BenchDoc>();

  async index(corpus: BenchDoc[]): Promise<void> {
    this.hybrid = new SquadKontextHybrid();
    await this.hybrid.index(corpus);
    for (const d of corpus) this.corpusById.set(d.id, d);
  }

  async query(question: string): Promise<RunResult> {
    const hState = (this.hybrid as unknown as {
      entityIndex: InMemoryEntityIndex;
      metaByDoc: Map<string, MetaDocument>;
      state: SquadState;
    });
    const retriever = new HybridRetriever(
      hState.entityIndex,
      hState.state.vectorStore,
      async () => hState.metaByDoc,
      0.4,
      0.6,
      1,
      "doc:",
      20,
    );
    const ranked = await retriever.retrieve(question, 3);
    const retrievedIds: string[] = [];
    const qTokens = question
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((w) => w.length > 2);

    type Scored = { sentence: string; score: number };
    const scored: Scored[] = [];
    for (const { doc } of ranked) {
      const bodyDoc = this.corpusById.get(doc.id);
      if (!bodyDoc) continue;
      retrievedIds.push(doc.id);
      const sents = bodyDoc.body
        .split(/(?<=[.!?])\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 15 && s.length < 500);
      for (const s of sents) {
        const lc = s.toLowerCase();
        const hits = qTokens.filter((w) => lc.includes(w)).length;
        if (hits === 0) continue;
        const density = hits / Math.max(s.length / 100, 1);
        scored.push({ sentence: s, score: hits + density * 0.5 });
      }
    }
    scored.sort((a, b) => b.score - a.score);
    const answer = scored.slice(0, 1).map((x) => x.sentence).join(" ");
    return { answer: answer || "No match.", retrievedDocIds: retrievedIds, contextChars: answer.length };
  }
}

/**
 * Detect exclusion patterns in a question: "besides X", "other than X,Y",
 * "except X", "apart from X". Returns the list of excluded terms (lowercased),
 * or [] if no exclusion pattern.
 *
 * Handles both "besides X, what..." and "besides X,what..." (missing space).
 */
export function extractExclusions(question: string): string[] {
  // Match exclusion keyword through to before the interrogative (what/which/...)
  const patterns = [
    /\bbesides\s+(.+?),?\s*\b(what|which|who|when|where|how|why)\b/i,
    /\bother\s+than\s+(.+?),?\s*\b(what|which|who|when|where|how|why)\b/i,
    /\bexcept\s+(?:for\s+)?(.+?),?\s*\b(what|which|who|when|where|how|why)\b/i,
    /\bapart\s+from\s+(.+?),?\s*\b(what|which|who|when|where|how|why)\b/i,
    /\bin\s+addition\s+to\s+(.+?),?\s*\b(what|which|who|when|where|how|why)\b/i,
  ];
  for (const re of patterns) {
    const m = question.match(re);
    if (m && m[1]) {
      return m[1]
        .replace(/[,?]+$/, "")
        .split(/\s*,\s*|\s+and\s+/)
        .map((s) => s.trim().toLowerCase())
        .filter((s) => s.length > 0 && s.length < 60);
    }
  }
  return [];
}

/**
 * Expand query terms by prefix match against corpus vocabulary. Handles
 * morphology mismatches like "cestida" -> "cestids", "aerobic" -> "aerobically".
 * Returns the expanded query string with additional variants OR'd in.
 */
export function expandQueryTerms(question: string, corpusText: string): string {
  const qTokens = question
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((w) => w.length > 4);
  const corpusTokens = new Set(
    corpusText
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((w) => w.length > 4),
  );
  const expansions = new Set<string>();
  for (const qt of qTokens) {
    // If exact match in corpus, skip (no expansion needed)
    if (corpusTokens.has(qt)) continue;
    // Look for corpus tokens sharing a 4+ char prefix
    const prefix = qt.slice(0, Math.max(4, qt.length - 2));
    for (const ct of corpusTokens) {
      if (ct !== qt && ct.startsWith(prefix) && Math.abs(ct.length - qt.length) <= 4) {
        expansions.add(ct);
      }
    }
  }
  if (expansions.size === 0) return question;
  return `${question} ${Array.from(expansions).join(" ")}`;
}

/**
 * Multi-candidate synthesis targeting 90%+ SQuAD accuracy.
 *
 * Generates 3 candidate answers in parallel (hybrid@1.5b, hybrid@3b,
 * extractive-on-hybrid) plus query preprocessing (exclusion detection +
 * morphology expansion), then synthesizes the final answer using a judge
 * LLM with exclusion-aware prompting.
 *
 * Targets: sq-14 (aerobic — fixed by extractive candidate), sq-24 (cestida
 * — fixed by query expansion), sq-28 (besides X,Y — fixed by exclusion-aware
 * synthesis prompt).
 */
export class SquadKontextSynthesis {
  private hybridSmall!: SquadKontextHybrid;
  private hybridLarge!: SquadKontextHybrid;
  private extractive!: SquadKontextExtractiveOnHybrid;
  private judge!: ChatOllama;
  private corpusById = new Map<string, BenchDoc>();
  private corpusText = "";

  async index(corpus: BenchDoc[]): Promise<void> {
    this.hybridSmall = new SquadKontextHybrid();
    this.hybridLarge = buildHybridWithModel("qwen2.5:3b");
    this.extractive = new SquadKontextExtractiveOnHybrid();
    await this.hybridSmall.index(corpus);
    await this.hybridLarge.index(corpus);
    await this.extractive.index(corpus);
    for (const d of corpus) this.corpusById.set(d.id, d);
    this.corpusText = corpus.map((d) => d.body).join(" ");
    this.judge = new ChatOllama({
      baseUrl: "http://localhost:11434",
      model: "qwen2.5:3b",
      temperature: 0,
      numGpu: 0,
      numCtx: 3072,
      numPredict: 96,
    });
  }

  async query(question: string): Promise<RunResult> {
    const exclusions = extractExclusions(question);
    const expanded = expandQueryTerms(question, this.corpusText);
    const qForRetrieval = expanded !== question ? expanded : question;

    const [sm, lg, ex] = await Promise.all([
      this.hybridSmall.query(qForRetrieval),
      this.hybridLarge.query(qForRetrieval),
      this.extractive.query(qForRetrieval),
    ]);

    // If small and large agree, return directly (high confidence)
    const normalizedEq = (a: string, b: string) =>
      a.trim().toLowerCase() === b.trim().toLowerCase();
    if (normalizedEq(sm.answer, lg.answer)) {
      return sm;
    }

    const ids = Array.from(
      new Set([...sm.retrievedDocIds, ...lg.retrievedDocIds, ...ex.retrievedDocIds]),
    );
    const context = ids
      .slice(0, 5)
      .map((id) => {
        const d = this.corpusById.get(id);
        return d ? `### ${d.title}\n${d.body.slice(0, 600)}` : "";
      })
      .filter(Boolean)
      .join("\n\n");

    const exclusionNote =
      exclusions.length > 0
        ? `\nIMPORTANT: The question excludes these: ${exclusions.join(", ")}. Reject any candidate whose answer is one of these terms or just restates them.`
        : "";

    // The extractive sentence is shown as "evidence from source" rather than a
    // peer candidate — the judge uses it to ground A vs B.
    const judgePrompt = `You must pick the correct answer to the question. You are given two candidate answers and a grounding sentence extracted verbatim from the source. Respond with ONLY a single letter: A or B. Nothing else.${exclusionNote}

Context:
${context}

Grounding sentence from source:
${ex.answer}

Question: ${question}

A: ${sm.answer}
B: ${lg.answer}

Which candidate (A or B) is better supported by the grounding sentence and context? Letter only:`;
    const res = await this.judge.invoke(judgePrompt);
    const raw =
      typeof res.content === "string" ? res.content.trim() : JSON.stringify(res.content);
    const head = raw.toUpperCase().slice(0, 40);
    const pickMatch = head.match(/\b([AB])\b/);
    const pick = pickMatch ? pickMatch[1] : "B";
    const chosen = pick === "A" ? sm : lg;

    // Final override: if an exclusion was detected and the chosen answer
    // mentions an excluded term but the grounding sentence names a different
    // concrete noun, prefer the grounding sentence's noun.
    if (exclusions.length > 0) {
      const chosenLc = chosen.answer.toLowerCase();
      const mentionsExcluded = exclusions.some((e) =>
        e.split(/\s+/).some((t) => t.length > 3 && chosenLc.includes(t)),
      );
      if (mentionsExcluded) {
        return {
          answer: ex.answer,
          retrievedDocIds: ids,
          contextChars: context.length,
        };
      }
    }
    return {
      answer: chosen.answer,
      retrievedDocIds: ids,
      contextChars: context.length,
    };
  }
}
