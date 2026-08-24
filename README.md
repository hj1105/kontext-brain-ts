# kontext-brain

> N-layer ontology-graph RAG framework for AI agents — TypeScript / Node.js.

[![node](https://img.shields.io/badge/node-%3E%3D20-brightgreen)](https://nodejs.org)
[![pnpm](https://img.shields.io/badge/pnpm-9-orange)](https://pnpm.io)
[![typescript](https://img.shields.io/badge/typescript-5.x-blue)](https://www.typescriptlang.org/)

A retrieval framework that organizes documents under a hierarchical ontology
graph instead of a flat vector index. The current public evaluation profile is
the **v13 anchored-evidence stack**: original-query-anchored multi-query
retrieval, graph/vector/BM25 fusion, coverage-aware reranking, source hydration,
and evidence-needs-constrained answers. The RAG evaluation harness selects v13
when no experimental mode is configured. See
[RAG evaluation v2](./bench/src/rag-eval-v2/README.md) for the frozen protocol
and the
[dataset-by-dataset cross-framework report](./bench/data/rag-eval-v2/cross-framework-all-datasets-2026-08-23.md);
the superseded experiments are indexed in
[Benchmark history](./bench/data/BENCHMARK_HISTORY.md), not presented as the
primary quality claim.

The idea: most production RAG indexes documents into a single semantic vector
space. kontext routes queries first through a small **ontology graph** (e.g.
"backend → REST APIs → JWT") and only then searches inside the matched
subspace. This (a) prunes irrelevant docs early, (b) gives you a natural place
to plug multiple data sources (Notion, Slack, GitHub) under one knowledge
structure, and (c) lets you swap retrieval strategies per layer without
rewriting the whole pipeline.

---

## Current benchmark snapshot

This is the section to read for current performance. The old Round-by-Round
research log lives separately in [Benchmark history](./bench/data/BENCHMARK_HISTORY.md).

- **Default, no configuration:** v13 anchored-evidence stack.
- **Latest validated candidate:** v15, which keeps the v13 ranking and answer
  policy but repairs missing original resources in incomplete precomputed KGs.
- **Promotion status:** v15 is not silently made the default yet. Novel supplied
  the development signal and SciFact recall moved slightly, so v13 remains the
  conservative default until another held-out gate confirms the change.

All rows below use the frozen shared answer/judge contract. Retrieval is scored
over the full dataset; answer quality uses the same deterministic 200-query
sample. Higher is better.

| Dataset | System | Recall@10 | nDCG@10 | Correctness | Strict faithfulness | Claim F1 | Citation F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Medical | **Kontext v15** | 0.8914 | 0.9689 | **0.9499** | **0.9614** | **0.8612** | **0.9583** |
| Medical | Kontext v13 default | 0.8923 | 0.9704 | 0.9461 | 0.9534 | 0.8550 | 0.9541 |
| Medical | LightRAG 1.5.6 | **0.9326** | 0.9990* | 0.8939 | 0.9417 | 0.8575 | 0.9477 |
| Medical | Microsoft GraphRAG 3.1.1 | 0.8303 | 0.9971* | 0.7817 | 0.8740 | 0.7336 | 0.8518 |
| Novel | **Kontext v15** | 0.8209 | 0.9349 | **0.8566** | **0.9290** | **0.8234** | 0.9369 |
| Novel | Kontext v13 default | 0.5259 | 0.6662 | 0.4654 | 0.7922 | 0.5181 | 0.5521 |
| Novel | LightRAG 1.5.6 | **0.8567** | 0.9945* | 0.8498 | 0.9272 | 0.8201 | **0.9407** |
| Novel | Microsoft GraphRAG 3.1.1 | 0.7716 | 0.9816* | 0.7668 | 0.8651 | 0.7434 | 0.8763 |

Context precision is deliberately omitted from this compact table. `*` marks
package-sensitive nDCG: LightRAG and Microsoft GraphRAG package a large native
context as one evidence record, while Kontext exposes separately scored evidence
windows, so their raw ranking/noise values are not directly comparable. The
detailed report includes that caveat, latency, confidence intervals, token use,
embedding cost, and the public BEIR
SciFact/NFCorpus retrieval guardrails:
[cross-framework evaluation](./bench/data/rag-eval-v2/cross-framework-all-datasets-2026-08-23.md).

### Official evaluation contract

No single aggregate “overall score” is used. A system must report the layers
separately so a retrieval gain cannot hide a grounding or abstention regression.

| Layer | Primary metric | What it checks |
|---|---|---|
| Retrieval | Evidence Recall@K | Whether the evidence required for the answer was actually found |
| Retrieval order | nDCG@K | Whether important evidence was ranked near the front |
| Retrieval noise | Context Precision | What fraction of retrieved evidence is relevant |
| Answer coverage | Claim Recall | Whether required answer claims were omitted |
| Grounding | Strict Faithfulness / Claim Support Precision | Whether every generated claim is entailed by retrieved evidence |
| Citations | Citation Precision / Recall / F1 | Whether citations support their claims and cover the claims that need them |
| Out-of-scope handling | Answerable/Unanswerable Joint Accuracy | Whether the system answers supported questions and abstains outside the KB |
| Stability | Robustness Drop | Performance loss after document-order, paraphrase, or distractor perturbations |
| Writing quality | Clarity / Conciseness / Fluency | Whether the response is readable without unnecessary wording |

Metrics that require labels or paired perturbations are reported as unavailable,
not fabricated. Historical score files retain the fields available under their
original frozen judge contract.

---

## What this project is

A modular monorepo with eight published packages and a benchmark harness:

| package | purpose |
|---------|---------|
| `@kontext-brain/core` | data model, retrieval pipelines, mapping strategies, extractive QA — pure TypeScript, no LLM dependencies |
| `@kontext-brain/llm` | LangChain.js adapters for Claude, OpenAI, Ollama |
| `@kontext-brain/mcp` | client connectors using the official `@modelcontextprotocol/sdk` (stdio + SSE), plus layer adapters for Notion / Jira / GitHub PR / Slack |
| `@kontext-brain/loader` | YAML/zod config loader + `KontextAgent` (the high-level entry point) including `autoSetup()` |
| `@kontext-brain/tool-server` | MCP server exposing kontext as 6 tools to any MCP client (Claude Desktop, Claude Code, Cursor, etc.) |
| `@kontext-brain/postgres` | PostgreSQL/pgvector KG, RLS-aware retrieval, ontology deployments, proposal queue, and extraction jobs |
| `@kontext-brain/object-storage` | S3-compatible compressed Resource content storage |
| `@kontext-brain/github` | accumulated ontology-proposal draft PR publisher |

There is no Python in the project — it is end-to-end TypeScript / Node.js.

### Architecture in one diagram

```
                    ┌─ Notion MCP ──┐
   user query ─►    │  GitHub MCP   │ ──►  kontext.autoSetup()  ──►  ontology graph
                    │  Slack MCP    │                                     │
                    └─ ... ─────────┘                                     │
                                                                          ▼
                                                ┌─────── L1: route query to nodes ───────┐
                                                │  KeywordMapping / VectorMapping /      │
                                                │  LLMMapping / HybridMapping            │
                                                └────────────────────┬───────────────────┘
                                                                     ▼
                                                ┌─────── L2: meta search per node ───────┐
                                                │  ScoreBasedSelector / LLMSelector      │
                                                └────────────────────┬───────────────────┘
                                                                     ▼
                                                ┌─────── L3: fetch + compress body ──────┐
                                                │  Full body / BM25 top-N sentences /    │
                                                │  ExtractiveRetriever (no LLM)          │
                                                └────────────────────┬───────────────────┘
                                                                     ▼
                                                ┌─────── L4: final reasoning LLM ────────┐
                                                │  RouterLLMAdapter (cheap+expensive)    │
                                                └─────────────────────────────────────────┘
```

Every layer is a port (TypeScript interface) with default implementations and
a registry pattern, so you can plug in any embedding model, vector store, MCP
server, chunker, or LLM without modifying core code.

The diagram above is the backward-compatible staged pipeline. The production
KG path uses a typed bidirectional graph instead: multi-source seeds can start
at an Ontology Node, Resource, Chunk, Entity, or Fact; bounded best-first
search then performs adaptive **Lift → Expand → Ground** until it has ranked,
ACL-accessible Evidence. It does not assume a DAG or a fixed number of lifts.

---

## Why use it

- **Multi-source from day one**: Notion + GitHub + Slack documents end up
  organized under one ontology, not in three disconnected vector indexes.
- **Predictable retrieval**: ontology routing is auditable — you can see
  exactly which nodes a query matched, then which docs under those nodes.
- **Cost-tunable**: choose between fast extractive (no final LLM, ~1ms
  latency, ~200 char context) and richer LLM-generated answers (1-8s,
  200–1700 char context) per query.
- **MCP-native**: built on the official Model Context Protocol SDK both as
  client (consume MCP servers) and as server (expose to AI agent hosts).
- **Governed auto-setup**: the first setup session can build a small ontology.
  Later unmatched documents enter a deduplicated proposal queue and draft PR;
  they never mutate the active ontology directly.

### Where it fits

| use case | recommended integration |
|----------|-------------------------|
| Company knowledge across Notion, Slack, GitHub, Jira, or internal MCP servers | Connect source MCP servers, synchronize them into the Evidence KG, and use the PostgreSQL runtime when ACL/RLS enforcement is required. |
| Existing AI clients and coding agents | Run `@kontext-brain/tool-server`; clients call the six kontext MCP tools without embedding the library. |
| A TypeScript application or service | Load a YAML configuration with `KontextLoader`, then call `retrieve()` for grounded context or `answer()` for a cited response. |
| Local or small-team knowledge base | Use the filesystem-backed store and local Ollama providers; no PostgreSQL or hosted completion API is required. |
| RAG research and regression testing | Use `bench/src/rag-eval-v2`; it freezes datasets, models, samples, metrics, manifests, and resumable checkpoints. |

The production `KontextAgent` remains configuration-driven because company
deployments have different stores, ACLs, models, and source connectors. The
**v13 default applies to the comparable RAG-evaluation adapter**, where omitting
`KONTEXT_RAG_EVAL_MODE` now selects the promoted stack. This distinction keeps
a benchmark policy from silently overriding a production security or storage
configuration.

### What “v13” does

1. Preserve the literal user question and generate at most three question-only
   retrieval perspectives.
2. Fuse vector and BM25 rankings with weighted reciprocal-rank fusion: the
   original question has weight 2 and each expansion has weight 1.
3. Add graph and context candidates, then apply a coverage-aware LLM reranker
   to the shared 50-candidate pool.
4. Hydrate selected sources into bounded 5,000-character windows under a
   50,000-character context budget.
5. Answer only supported question-derived evidence needs, with at most one
   atomic claim and one best citation per need (maximum eight claims).

Dataset names, reference answers, gold evidence, and judge outputs are not
available to these decisions. The newer v15 experiment adds corpus-completeness
repair when a precomputed KG omitted original resources. It passed the Medical
and public retrieval regression gates but remains an explicit candidate because
Novel supplied the original development signal and SciFact moved slightly.

---

## Quick start

### Prerequisites

```bash
node --version    # >= 20
corepack enable   # enables pnpm
```

You also need **either** local LLM access (Ollama) **or** an API key for
Claude / OpenAI.

```bash
# Local LLM (free, slower):
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text

# OR Claude:
export ANTHROPIC_API_KEY=sk-ant-...
```

### Install + build

```bash
git clone <repo>
cd kontext-brain-ts
pnpm install
pnpm -r build
pnpm test            # unit, contract, and integration tests
```

### Run the example

```bash
pnpm --filter @kontext-brain/example-basic start       # in-process toy
pnpm --filter @kontext-brain/example-auto-setup start  # mock MCP servers + autoSetup
pnpm --filter @kontext-brain/bench start               # full 14-system benchmark (needs Ollama)
pnpm --filter @kontext-brain/bench ralph               # short-form "Ralph loop"
```

---

## Library usage

### Pattern A — full agent from YAML (most common)

```typescript
import { KontextLoader } from "@kontext-brain/loader";

const agent = await KontextLoader.fromFile("kontext.yaml");
await agent.autoSetup();   // first time only — builds ontology + indexes docs
const retrieval = await agent.retrieve("How should I version my REST API?");
console.log(retrieval.context);          // no final reasoning LLM call

const result = await agent.answer("How should I version my REST API?");

console.log(result.answer);
console.log(result.selectedMetaDocs);  // sourced documents
console.log(result.contextTokensUsed);
```

### Production Evidence KG

The production path keeps structured state in PostgreSQL/pgvector and one
compressed current object per Resource in S3-compatible storage. External MCP
systems remain the source of truth. Source changes stale old derived Evidence
and atomically activate the replacement; stable Facts record lifecycle events
instead of accumulating versions.

```typescript
import { Pool } from "pg";
import { S3Client } from "@aws-sdk/client-s3";
import { S3ResourceContentStore } from "@kontext-brain/object-storage";
import { createPostgresKnowledgeRuntime, migratePostgres } from "@kontext-brain/postgres";
import { GenericMCPResourceSnapshotAdapter } from "@kontext-brain/mcp";
import { KontextLoader } from "@kontext-brain/loader";
import { AdaptiveKnowledgeEnricher } from "@kontext-brain/core";
import { LangChainLLMAdapter, LLMProviderRegistry } from "@kontext-brain/llm";

const pool = new Pool({ connectionString: process.env.DATABASE_URL });
await migratePostgres(pool);

const contentStore = new S3ResourceContentStore(new S3Client({}), {
  bucket: process.env.KONTEXT_CONTENT_BUCKET!,
});
const candidateReindexer = {
  // Build and regression-check the candidate KG; resolve only when it is safe to activate.
  async prepare(candidate) { await rebuildCandidateKnowledgeGraph(candidate); },
};
const llmProviders = new LLMProviderRegistry();
const extractionModel = llmProviders.createChat({
  provider: "ollama",
  model: "qwen2.5:7b",
});
const snapshotEnricher = new AdaptiveKnowledgeEnricher(
  new LangChainLLMAdapter(extractionModel),
);
const runtime = createPostgresKnowledgeRuntime(pool, contentStore, [
  // Use source-native adapters for Slack messages, Notion block subtrees, etc.
  // This generic adapter is the recursive-chunk fallback.
  new GenericMCPResourceSnapshotAdapter("notion", "notion", {
    groupIds: ["knowledge-users"],
  }),
], candidateReindexer, snapshotEnricher);

const agent = await new KontextLoader({
  knowledgeRuntime: {
    organizationId: "acme",
    knowledgeRetriever: runtime.knowledgeRetriever,
    mcpKnowledgeSynchronizer: runtime.mcpKnowledgeSynchronizer,
    ontologyProposalQueue: runtime.ontologyProposalQueue,
    ontologyActivation: runtime.ontologyActivation,
  },
}).fromFile("kontext.yaml");

const principal = {
  organizationId: "acme",
  subjectId: "user-123",
  groupIds: ["knowledge-users"],
};
const evidence = await agent.retrieve("Was order 42 paid?", principal);
const answer = await agent.answer("Was order 42 paid?", principal);
```

The database migration enables organization RLS, normalized many-to-many
Resource/Chunk–Ontology links, Facts, Evidence, Fact events, pgvector columns,
idempotent extraction jobs, ontology deployments, proposals, and structured
audit rows. `answer()` fails closed when no accessible active Evidence exists
or the generated answer does not cite an Evidence ID.

`AdaptiveKnowledgeEnricher` is optional. When enabled, it examines literal
source chunks—not corpus or dataset names—to select and dispatch
identity-resolution, event, temporal, causal, and cross-chunk extraction
capabilities. It keeps entities resource-scoped by default, requires an exact
source quote for every Mention and Claim, independently verifies that an
explicit Claim is directly supported, anchors Entity identity to stable source
addresses rather than display names, and resolves identity across all windows
of a Resource. On updates, `SyncResourceUseCase` supplies prior active identity
records separately from the new snapshot, so IDs can be reused without
reactivating disappeared Mentions. Inferred Claims remain Hypotheses and any
invalid window rejects the whole enrichment before synchronization.

`kontext.yaml` for an Ollama-only setup:

```yaml
llm:
  traversal: { provider: ollama, model: qwen2.5:1.5b, baseUrl: http://localhost:11434 }
  reasoning: { provider: ollama, model: qwen2.5:1.5b, baseUrl: http://localhost:11434 }

mcp:
  - { name: notion-docs,  url: http://localhost:8101, type: notion,    transport: sse }
  - { name: github-issues, command: "npx", args: ["@modelcontextprotocol/server-github"], transport: stdio }

# ontology can be omitted — autoSetup() will build one
ontology:
  - { id: backend,  description: REST API server database JWT, weight: 0.9 }
  - { id: frontend, description: React UI components,           weight: 0.9 }

storage:
  type: file
  path: ./.kontext-store  # graph + meta index + MCP assignments

graph:
  maxDepth: 2
  maxTokens: 4000
  strategy: WEIGHTED_DFS
```

### Pattern B — programmatic, no YAML

For full control over each component, build the agent directly:

```typescript
import {
  ContentFetcherRegistry,
  DEFAULT_PIPELINE,
  DataSource,
  InMemoryMetaIndexStore,
  InMemoryOntologyStore,
  IngestPipeline,
  KeywordMappingStrategy,
  OntologyGraph,
  RouterLLMAdapter,
  ScoreBasedSelector,
  TraversalStrategy,
  createMetaDocument,
  createNode,
} from "@kontext-brain/core";
import { LangChainLLMAdapter, LangChainVectorStore, LLMProviderRegistry } from "@kontext-brain/llm";
import { KontextAgent } from "@kontext-brain/loader";

const registry = new LLMProviderRegistry();
const chat = registry.createChat({ provider: "ollama", model: "qwen2.5:1.5b" });
const adapter = new LangChainLLMAdapter(chat);
const router = new RouterLLMAdapter(adapter, adapter);

const nodes = new Map([
  ["backend",  createNode({ id: "backend",  description: "REST API JWT", weight: 1 })],
  ["frontend", createNode({ id: "frontend", description: "React UI",     weight: 1 })],
]);
const graph = new OntologyGraph(nodes, [], {
  maxDepth: 2, maxTokens: 4000, strategy: TraversalStrategy.WEIGHTED_DFS,
});

const agent = new KontextAgent({
  ontologySchemaGraph: graph, router,
  mcpConnectors: [], mcpLayerAdapters: [],
  metaIndexStore: new InMemoryMetaIndexStore(),
  fetcherRegistry: new ContentFetcherRegistry(),
  vectorStore: null,
  mappingStrategy: new KeywordMappingStrategy(),
  metaSelector: new ScoreBasedSelector(),
  ingestPipeline: new IngestPipeline(adapter, new InMemoryOntologyStore(), null as any),
});

const res = await agent.query("backend authentication");
```

### Pattern C — extractive (no LLM at query time)

For tech-docs QA where answers are literal sentences and you need
sub-millisecond latency:

```typescript
import { ExtractiveRetriever } from "@kontext-brain/core";

const extractor = new ExtractiveRetriever(fetcherRegistry, /* topSentences */ 2);
const candidates = await metaIndex.search(nodeId, query, 3);
const result = await extractor.answer(query, candidates);
// result.answer is the top-scored sentences from the matched docs
// no LLM call; runs in <1ms on small corpora
```

### Pattern D — expose to any AI agent via MCP

```bash
# Start the kontext MCP tool server pointing at your config
pnpm --filter @kontext-brain/tool-server start kontext.yaml

# Or, after `pnpm -r build`, use the bin:
./packages/tool-server/dist/cli.js kontext.yaml
```

Register with Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "kontext": {
      "command": "node",
      "args": ["/abs/path/to/packages/tool-server/dist/cli.js", "/abs/path/to/kontext.yaml"]
    }
  }
}
```

The server exposes 6 tools to the host agent:

| tool | input | output |
|------|-------|--------|
| `kontext_query` | `{ question }` | reasoned answer + sources |
| `kontext_query_context` | `{ question }` | retrieved context only (no LLM reasoning) |
| `kontext_ingest` | `{ data, source? }` | extracts entities into the graph |
| `kontext_describe` | `{}` | dumps ontology / pipeline / MCP adapters |
| `kontext_sync` | `{ connectorName? }` | incrementally classify additions/changes and remove deleted resources |
| `kontext_auto_setup` | `{ targetNodeCount? }` | LLM builds/expands ontology + classifies docs |

---

## Pluggable interfaces

Every retrieval stage is a port. Default implementations ship in core, and you
can register your own without touching upstream:

| port | defaults | swap with |
|------|----------|-----------|
| `LLMAdapter` | `LangChainLLMAdapter` (Claude / OpenAI / Ollama) | any function returning `Promise<string>` |
| `VectorStore` | `InMemoryVectorStore`, `LangChainVectorStore` | Pinecone, Weaviate, Postgres pgvector, etc. |
| `MetaIndexStore` | `InMemoryMetaIndexStore`, `VectorMetaIndexStore` | DB-backed implementations |
| `ContentFetcher` | `MCPContentFetcherBridge` | HTTP, S3, filesystem, custom APIs |
| `NodeMappingStrategy` | `Keyword`, `Vector`, `LLM`, `Hybrid` | per-corpus tuning |
| `MetaDocumentSelector` | `ScoreBased`, `LLMMetaDocumentSelector` | reranker models |
| `StepExecutor` | `Ontology`, `Meta`, `Vector`, `Content`, `Section`, `Chunk` | new pipeline-step kinds |
| `Tokenizer` | `Whitespace`, `CharNGram`, `Composite`, `MultiLanguage` | language-specific |
| `ChunkingStrategy` | `RegexHeader`, `Paragraph`, `Recursive` | domain-specific splitters |
| `TokenEstimator` | `Default` (English), `Korean` | tiktoken, claude-tokenizer, etc. |
| `OntologyStore` | `InMemory`, `File` | DB persistence |
| `MCPConnector` | `Stdio`, `Sse` (official SDK) | custom transports |

Pipeline composition uses preset constants (`DEFAULT_PIPELINE`,
`VECTOR_PIPELINE`, `N_LAYER_PIPELINE`, `PERNODE_PIPELINE`) or user-defined
arrays of `PipelineStep` objects. Pipeline steps are ordered stages and run
for every traversed ontology node; graph depth is independent from retrieval
stage order, so a leaf node still executes META → CONTENT.

### State and persistence

`KontextAgent` remains the orchestration boundary. The lightweight file store
keeps only the legacy ontology-schema/meta-index/MCP-sync snapshot for local
development. It never contains production Resource, Chunk, Entity, Fact, or
Evidence rows. In production,
`@kontext-brain/postgres` is the canonical structured store and
`@kontext-brain/object-storage` holds normalized current bodies. The loader
compares a SHA-256 hash of configured ontology YAML with the active snapshot;
a changed candidate is validated before an atomic activation, while invalid
relations or parent cycles leave the old graph active.

The Agent keeps the small YAML-derived `ontologySchemaGraph` in memory. During
retrieval, the production search adapter loads only the accessible neighboring
KG rows needed by the bounded frontier; it does not hydrate the instance KG
into process memory. Resource content is fetched only after SQL ACL checks.

MCP resources are stored as documents classified under ontology nodes, not
as graph nodes themselves:

```
OntologyNode → MetaDocument(resource id + connector) → MCP resource body
```

`syncMCP()` keeps prior assignments for unchanged resources, classifies only
new or changed resources, and soft-removes deleted Resources. With an
`MCPKnowledgeSynchronizer`, it also persists Resource bodies, source-native or
fallback Chunks, many-to-many ontology links, and source Evidence. A failed
connector is skipped rather than interpreted as deleting all of its documents.

---

---

## Auto-setup flow (the killer feature)

When you don't yet have an ontology and just want to point kontext at MCP
sources:

```typescript
const agent = await KontextLoader.fromFile("kontext.yaml");
const result = await agent.autoSetup({ targetNodeCount: 8 });

console.log(`Built ${result.nodesCreated} ontology nodes`);
console.log(`Classified ${result.documentsClassified} documents`);
console.log(`${result.documentsUnmapped} unmapped`);
console.log(result.ontologyYaml);  // save this back to kontext.yaml for reuse
```

Internally:

1. `MCPConnector.listResources()` on every connector → `MCPResourceInfo[]`
2. `OntologyAutoBuilder.build()` — LLM extracts categories, designs N nodes
   with parent/level hierarchy, infers edges
3. `DocumentClassifier.classify()` — LLM maps each document to its best node;
   any unmappable docs spawn new nodes
4. `MetaIndexStore.index()` per node
5. `VectorStore.upsert()` of node descriptions and (optionally) document bodies

The whole flow takes one network round-trip per LLM call, parallelized where
safe. Total time: roughly 30 seconds for a 100-doc corpus on Claude Haiku;
proportionally slower on local Ollama.

---

## Project layout

```
kontext-brain-ts/
├── package.json                   # root workspace config (pnpm)
├── pnpm-workspace.yaml
├── tsconfig.base.json
├── biome.json                     # lint + format (Biome, not eslint+prettier)
├── vitest.config.ts
├── packages/
│   ├── core/                      # @kontext-brain/core    — pure TS, no LLM deps
│   │   └── src/
│   │       ├── graph/             # OntologyNode, OntologyGraph, GraphTraverser
│   │       ├── query/             # mapping strategies, pipelines, retrievers
│   │       ├── ingest/            # OntologyAutoBuilder, DocumentClassifier
│   │       └── store/             # OntologyStore (memory + file)
│   ├── llm/                       # @kontext-brain/llm      — LangChain.js wrappers
│   ├── mcp/                       # @kontext-brain/mcp      — official MCP SDK
│   ├── loader/                    # @kontext-brain/loader   — KontextAgent + YAML
│   └── tool-server/               # @kontext-brain/tool-server — MCP server (stdio)
├── examples/
│   ├── basic/                     # programmatic toy
│   └── auto-setup/                # mock Notion + Slack → autoSetup → query
├── tests/integration/             # vitest end-to-end
└── bench/                         # 14-system benchmark + Ralph loop
    ├── src/corpus.ts              # 12-doc tech corpus + 8 labeled queries
    ├── src/baseline.ts            # standard LangChain.js vector RAG
    ├── src/kontext-runner.ts      # all kontext variants (V1-V17)
    ├── src/run.ts                 # full 14-system run
    └── src/ralph.ts               # short-loop subset for fast iteration
```

---

## Tech stack

- **Language**: TypeScript 5.x, strict mode, `noUncheckedIndexedAccess`
- **Runtime**: Node.js 20+ (uses native `performance.now()`, ESM, fetch)
- **Package manager**: pnpm 9 (workspaces)
- **Build**: tsup (esbuild + dts)
- **Test**: vitest
- **Lint/format**: Biome (single binary, no eslint+prettier)
- **Validation**: zod (runtime parsing of YAML config)
- **YAML**: `yaml` package
- **HTTP / MCP**: `@modelcontextprotocol/sdk` (stdio + SSE transports)
- **LLMs**: `@langchain/anthropic`, `@langchain/openai`, `@langchain/ollama`
- **Embeddings**: any LangChain.js `Embeddings` (default: Ollama
  `nomic-embed-text`, OpenAI `text-embedding-3-small`)

No build dependencies on Python, Java, or Rust. No transpiled binaries shipped
in the repo. Everything runs on a stock Node 20 install plus pnpm.

---

## Status

**Honest current state:**

- ✅ Core, llm, mcp, loader, tool-server packages: typecheck + build clean
- ✅ Unit + integration coverage for retrieval, persistence, incremental MCP
  synchronization, graph traversal, entities, and the tool server
- ✅ Real Ollama benchmarked end-to-end on 14 retrieval variants
- ✅ Ralph-loop iterative optimization completed, exceeded 10x efficiency
  target by ~40,000x
- ✅ `DEFAULT_PIPELINE` leaf-node bug fixed; original Kotlin codebase had
  the same issue
- ⚠️ Real Notion / GitHub / Slack MCP servers not yet smoke-tested end-to-end
  (incremental synchronization is covered with mock connectors)
- ⚠️ Larger-corpus benchmarking pending (12 docs is small)
- ⚠️ LLM-as-judge quality scoring not implemented yet (currently using
  keyword-fragment matching as a weak proxy)

**Originally a Kotlin project**, ported to TypeScript because (a) the Model
Context Protocol ecosystem is TypeScript-first, (b) AI-agent OSS gravity is
on Node, (c) frontend developers can adopt it directly. The Kotlin reference
is preserved as `kb-clean/` in the parent directory.

---

## License

TBD (currently unlicensed — request before production use).
