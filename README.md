# kontext-brain

[English](./README.md) | [한국어](./README.ko.md)

> Evidence-backed N-layer knowledge-graph RAG for AI agents — TypeScript / Node.js.

[![node](https://img.shields.io/badge/node-%3E%3D20-brightgreen)](https://nodejs.org)
[![pnpm](https://img.shields.io/badge/pnpm-9-orange)](https://pnpm.io)
[![typescript](https://img.shields.io/badge/typescript-5.x-blue)](https://www.typescriptlang.org/)

A retrieval framework that structures multi-source knowledge as Resources,
source-native Chunks, Entities, Facts, and ACL-aware Evidence instead of treating
the corpus as a flat vector index. The included RAG evaluation harness uses the
**v13 anchored-evidence stack** by default: original-query-anchored multi-query
retrieval, graph/vector/BM25 fusion, coverage-aware reranking, source hydration,
and evidence-needs-constrained answers. See
[RAG evaluation v2](./bench/src/rag-eval-v2/README.md) and its
[development report](./bench/data/rag-eval-v2/cross-framework-all-datasets-2026-08-23.md).
The profile was iteratively tuned, some comparisons use a precomputed Kontext
KG, and the report's raw run directories are not committed, so these results
are regression evidence rather than an independently reproducible final
cross-framework benchmark.

The production path keeps external systems as the source of truth and maintains
an evidence-backed derived index. A query can seed accessible Resources, Chunks,
Entities, Facts, and optional Ontology anchors, then run bounded best-first
**Lift → Expand → Ground** traversal. Only the source Chunks selected as Evidence
are hydrated for answering. Ontology-first staged routing remains available for
backward compatibility; it is not the definition of production N-layer retrieval.

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

### Production architecture

```text
INGEST / STRUCTURE                          QUERY / ANSWER

Notion · Slack · GitHub · MCP              question + Principal
             │                                      │
             ▼                                      ▼
    normalize + synchronize              ACL/RLS-filtered seed fusion
             │                           Resource | Chunk | Entity | Fact
             ▼                                  | optional Ontology
Resource ──contains──► Chunk                       │
                         │                         ▼
                         ├─mentions──► Entity   bounded best-first search
                         │              │       Lift ↔ Expand ↔ Ground
                         │              ▼                  │
                         │             Fact                ▼
                         │              ▲       ranked accessible Evidence
                         └─grounds──► Evidence   + lazy Chunk hydration
                               optional factKey             │
                                                          ▼
                                               final reasoning LLM
                                                          │
                                                          ▼
                                               citation validation
                                               or fail closed

PostgreSQL: structured KG, ACL/RLS, scoring profiles
Object storage: current Resource and Chunk content
```

Ingestion and querying are separate flows: `autoSetup()` or synchronization does
not run for every question. The traversable node kinds are Ontology, Resource,
Chunk, Entity, and Fact. Evidence is the terminal, ranked grounding record bound
to a Resource and Chunk, optionally to a Fact; it is not another traversable node.
ACL filtering is a hard precondition for seeds, edges, and Evidence. Source text
is loaded only when an accessible Chunk is grounded, and the final reasoning LLM
is a downstream consumer rather than a graph layer. Missing Evidence or missing
Evidence-ID citations fail closed.

The legacy local/file-store path remains available as a linear compatibility
pipeline:

```text
Ontology routing → per-node meta search → body fetch/compression → final LLM
```

Its mapping, vector, metadata, content, chunking, and LLM components remain
replaceable ports, but this fixed L1–L4 sequence must not be confused with the
production Evidence KG. A source-grounded LLM Wiki is being evaluated as a
future replacement for predeclared Ontology routing; it is not an active
production route, and generated Wiki prose can never serve as answer Evidence.

---

## Why use it

- **Multi-source from day one**: Notion, GitHub, and Slack synchronize into one
  evidence-backed KG instead of three disconnected vector indexes.
- **Auditable grounding**: each selected context carries its Evidence, Resource,
  and Chunk identity plus the scored traversal path that reached it.
- **Cost-tunable**: choose between extractive retrieval with no final LLM and
  richer LLM-generated answers per query.
- **MCP-native**: built on the official Model Context Protocol SDK both as
  client (consume MCP servers) and as server (expose to AI agent hosts).
- **Governed optional Ontology path**: the first setup session can build a small
  Ontology. Later unmatched documents enter a deduplicated proposal queue and
  draft PR; they never mutate the active deployment directly.

### Where it fits

| use case | recommended integration |
|----------|-------------------------|
| Company knowledge across Notion, Slack, GitHub, Jira, or internal MCP servers | Connect source MCP servers, synchronize them into the Evidence KG, and use the PostgreSQL runtime when ACL/RLS enforcement is required. |
| Existing AI clients and coding agents | Run `@kontext-brain/tool-server`; clients call the six kontext MCP tools without embedding the library. |
| A TypeScript application or service | Load a YAML configuration with `KontextLoader`, then call `retrieve()` for grounded context or `answer()` for a cited response. |
| Local or small-team knowledge base | Use the filesystem-backed store and local Ollama providers; no PostgreSQL or hosted completion API is required. |
| RAG research and regression testing | Use `bench/src/rag-eval-v2`; it versions datasets, models, samples, metrics, manifests, and resumable checkpoints. |

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
available to the runtime decisions. The v13/v15 policies were nevertheless
selected through iterative development on reported datasets and are not an
untouched holdout result. The newer v15 experiment adds corpus-completeness
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
bench/node_modules/.bin/tsx bench/src/rag-eval-v2/cli.ts doctor  # inspect evaluation prerequisites
pnpm --filter @kontext-brain/bench start               # legacy local benchmark (needs Ollama)
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

N-Layer traversal scoring is observation-based. Search adapters report lexical/vector ranks,
neighbor-list fanout and rank, normalized query evidence, relationship provenance, ACL-filtered
evidence counts, conflicts, and freshness. A versioned base profile plus an optional query-bound
route policy turns those observations into priorities without branching on dataset or organization
identity. Every result trace records the profile and feature-schema digests, missing signals,
seed-provider counts, route decisions, path lengths, and a per-evidence score breakdown. PostgreSQL
profiles support staged evaluation, full shadow traversal, deterministic canaries, activation, and
atomic rollback:

```typescript
const staged = await runtime.scoringProfiles.stage("acme", candidateProfile, evaluationSummary);
await runtime.scoringProfiles.setShadow("acme", staged.profileDigest);
await runtime.scoringProfiles.setCanaryPercent("acme", 5);
await runtime.scoringProfiles.activate("acme", staged.profileDigest);
// await runtime.scoringProfiles.rollback("acme", previousProfileDigest);
```

See [ADR 0005](./docs/adr/0005-versioned-traversal-scoring.md),
[ADR 0006](./docs/adr/0006-query-adaptive-route-scoring.md), the
[adaptive evaluation](./bench/data/rag-eval-v2/adaptive-route-v3-reevaluation-2026-08-24.md), and
the [raw direct-only ablation](./bench/data/rag-eval-v2/adaptive-route-v3-direct-only-ablation-2026-08-25.md).
The subsequent [source-hydrated direct-only ablation](./bench/data/rag-eval-v2/source-hydrated-direct-only-ablation-2026-08-25.md)
found a small aggregate graph recall gain paired with a precision regression and no strict
two-dataset holdout win. Source-hydrated direct retrieval is therefore the current quality
candidate; adaptive graph traversal remains an explicit recall-first experiment and must not be
activated by default. See the [rollout runbook](./docs/runbooks/scoring-profile-rollout.md) for the
remaining gates.

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

For document QA where answers are literal sentences and you want to avoid an
LLM call at query time:

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

## Performance (current retrieval candidate)

The latest evaluation (2026-08-25) covers the full GraphRAG-Bench Medical
(2,062 queries) and Novel (2,010 queries) retrieval sets. The current quality
candidate is **source-hydrated direct hybrid retrieval**: vector and lexical
candidates are fused and reranked, then hydrated into contiguous
5,000-character source windows under a 36,000-character context cap. Graph
traversal is disabled (`maxHops: 0`) because the matched graph ablation did
not pass the default-promotion gate.

| Dataset | Queries | Evidence recall@10 | Lift vs raw direct | p95 retrieval |
|---|---:|---:|---:|---:|
| Medical | 2,062 | **0.80892** | **+0.09360 (+9.36pp)** | **4.18 ms** |
| Novel | 2,010 | **0.43980** | **+0.06915 (+6.92pp)** | **12.40 ms** |

Evidence recall measures how much of the required gold evidence is present in
the combined top-10 context. These are retrieval results, not answer accuracy
or citation scores. They use frozen OpenAI `text-embedding-3-small`
checkpoints, vector seeds 10, lexical seeds 5, and the same candidate and
reranking settings on both datasets.

`Context precision` is retained as a secondary diagnostic because its name is
easy to confuse with answer precision. For each query, it is the fraction of
returned source windows that individually cover at least 50% of a gold-evidence
text. It is sensitive to window packaging and does not mean that only 35.7% or
65.6% of answers are correct.

| Dataset | Raw direct | Current candidate | Absolute improvement |
|---|---:|---:|---:|
| Medical | 0.37410 | **0.65641** | **+0.28230** |
| Novel | 0.18483 | **0.35696** | **+0.17214** |

### Cross-framework comparison

The latest completed shared-protocol comparison uses the **Kontext v15**
evaluation profile. It is separate from the newer source-hydrated direct
candidate above, which has not yet been rerun against every external system.
Retrieval covers all 2,062 Medical and 2,010 Novel queries; answer and judge
metrics use the same deterministic 200-query sample per dataset.

| Dataset | System | Recall@10 | Answer correctness | Strict faithfulness | Citation F1 |
|---|---|---:|---:|---:|---:|
| Medical | **Kontext v15** | 89.1% | **95.0%** | **96.1%** | **95.8%** |
| Medical | LightRAG 1.5.6 | **93.3%** | 89.4% | 94.2% | 94.8% |
| Medical | Microsoft GraphRAG 3.1.1 | 83.0% | 78.2% | 87.4% | 85.2% |
| Medical | Vector + BM25-RRF | 70.7% | 87.4% | 89.5% | 90.0% |
| Novel | **Kontext v15** | 82.1% | **85.7%** | **92.9%** | 93.7% |
| Novel | LightRAG 1.5.6 | **85.7%** | 85.0% | 92.7% | **94.1%** |
| Novel | Microsoft GraphRAG 3.1.1 | 77.2% | 76.7% | 86.5% | 87.6% |

Kontext v15 does not have the highest retrieval recall—LightRAG leads both
datasets—but it produces the best answer correctness and strict faithfulness
in this run. On Novel, LightRAG retains a small citation-F1 lead.

This is a provisional development comparison, not an independent leaderboard.
Kontext uses a precomputed KG while the external systems build native indexes,
so index-build cost is not equivalent. Historical adapter timings used
different queue boundaries and are intentionally omitted. Packaged-context
precision is also omitted because LightRAG and Microsoft GraphRAG return large
native contexts as single evidence records, making that metric incomparable.
See the
[cross-framework report](./bench/data/rag-eval-v2/cross-framework-all-datasets-2026-08-23.md)
for the full protocol, raw scores, limitations, and unsupported systems.

### Matched graph-traversal ablation

The latest graph-enabled treatment changes only `maxHops` from 0 to 8.

| Dataset | Direct recall | Graph recall | Recall delta (95% CI) | Context-precision delta (95% CI) | p95 direct → graph |
|---|---:|---:|---:|---:|---:|
| Medical | 0.80892 | **0.81474** | +0.00582 [0.00048, 0.01115] | -0.00602 [-0.00934, -0.00269] | 4.18 → 24.41 ms |
| Novel | 0.43980 | **0.44478** | +0.00498 [0.00100, 0.00896] | -0.00277 [-0.00508, -0.00048] | 12.40 → 43.09 ms |

Graph traversal adds about 0.5 percentage points of aggregate recall while
slightly reducing context precision and materially increasing retrieval
latency. The recall gain did not become a strict win on both regression
holdouts, so graph traversal remains an explicit recall-first option rather
than the default.

See the
[source-hydrated direct-only ablation](./bench/data/rag-eval-v2/source-hydrated-direct-only-ablation-2026-08-25.md)
for the protocol, confidence intervals, holdout results, and artifact paths.

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
└── bench/                         # versioned RAG evaluation and regression harness
    ├── src/rag-eval-v2/           # datasets, adapters, metrics, and resumable runs
    ├── data/rag-eval-v2/          # reviewed evaluation reports and manifests
    └── src/run.ts                 # legacy local benchmark entry point
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
- ✅ Versioned retrieval evaluation covers all 4,072 Medical and Novel queries
- ✅ Latest source-hydrated direct candidate has a matched graph on/off ablation
- ✅ `DEFAULT_PIPELINE` leaf-node bug fixed; original Kotlin codebase had
  the same issue
- ⚠️ Real Notion / GitHub / Slack MCP servers not yet smoke-tested end-to-end
  (incremental synchronization is covered with mock connectors)
- ⚠️ Graph traversal remains opt-in because its recall gain trades away
  precision and does not pass the strict two-dataset holdout gate
- ⚠️ Answer-level faithfulness and citation evaluation is still required for
  the latest retrieval candidate before production activation

**Originally a Kotlin project**, ported to TypeScript because (a) the Model
Context Protocol ecosystem is TypeScript-first, (b) AI-agent OSS gravity is
on Node, (c) frontend developers can adopt it directly. The Kotlin reference
is preserved as `kb-clean/` in the parent directory.

---

## Product direction: self-hosted OSS and optional managed deployment

Every runtime and framework package in this repository is Apache-2.0 and
self-hostable. The product direction is a **governed knowledge layer for AI
agents**: point it at your Notion / Slack / GitHub, and every agent answer
respects who can see what (ACL), cites its source (Evidence), and refuses when
it has no grounding (fail-closed). A future managed deployment may package the
same open-source runtime with hosted operations and administration.

### The wedge, in one sentence

> Most RAG demos leak documents a user can't access, can't tell you why they
> answered, and hallucinate when they don't know. kontext's production path
> already fixes all three. The same runtime can be self-hosted or operated as a
> managed, governed deployment.

### What already exists vs. what the cloud needs

The repository-to-service boundary maps cleanly onto "runtime" vs "operations".
The retrieval and governance logic remains Apache-2.0; a managed deployment is
primarily a control-plane and operations wrapper around it.

| Layer | Status | Where |
|-------|--------|-------|
| Retrieval pipelines, ontology graph, pluggable retrievers | ✅ built | `@kontext-brain/core` (Apache-2.0) |
| MCP client + server, source adapters | ✅ built (⚠️ real-server E2E pending) | `@kontext-brain/mcp`, `tool-server` (Apache-2.0) |
| Multi-tenant KG, org RLS, ACL retrieval, Evidence, fail-closed `answer()` | ✅ built | `@kontext-brain/postgres` (Apache-2.0) |
| Compressed source-of-truth body storage | ✅ built | `@kontext-brain/object-storage` (Apache-2.0) |
| Ontology-proposal governance (draft PRs) | ✅ built | `@kontext-brain/github` (Apache-2.0) |
| **Control plane** (signup, org provisioning, connector OAuth, usage metering, billing) | ⬜ to build | future service layer |
| **Admin UI** (connect sources, browse ontology, audit log, ACL preview) | ⬜ to build | future service layer |
| **Agent endpoint** (hosted MCP server + REST per org) | ⬜ mostly wiring | wraps `tool-server` + `postgres` |
| SSO / SCIM, SOC2, on-prem installer | ⬜ enterprise phase | — |

### Managed-cloud architecture (target)

```
                         ┌──────────────────────────────────────────┐
   Notion / Slack /      │              kontext cloud                │
   GitHub / Jira  ──MCP──►  ┌────────────┐   ┌────────────────────┐  │
                         │  │ control    │   │  per-org runtime    │  │
   customer's agent      │  │ plane      │   │  (Apache-2.0):      │  │
   (Claude, Cursor) ─────►  │ • auth/org │   │  createPostgres     │  │
        (hosted MCP)     │  │ • connectors│  │   KnowledgeRuntime  │  │
                         │  │ • metering │   │  • RLS retrieval     │  │
   admin (browser) ──────►  │ • billing  │   │  • Evidence + ACL    │  │
        (admin UI)       │  └────────────┘   │  • fail-closed answer│  │
                         │       │           └─────────┬───────────┘  │
                         │  Postgres/pgvector  ◄────────┘              │
                         │  + S3 (object-storage)                     │
                         └──────────────────────────────────────────┘
```

The per-org runtime is literally the code in this repo
(`createPostgresKnowledgeRuntime` + `S3ResourceContentStore` +
`KontextLoader`). A future managed service would add only the multi-tenant
control plane and UI on top — no re-implementation of retrieval.

---

## License

The repository and all eight published packages are licensed under
**Apache License 2.0**. This includes `postgres`, `object-storage`, and `github`.
Commercial use, modification, distribution, self-hosting, and hosted services
are permitted subject to the license terms. See [`LICENSE`](./LICENSE) and
[`LICENSING.md`](./LICENSING.md).
