# RAG evaluation v2

This harness compares retrieval frameworks at the same boundary: raw corpus in,
ranked evidence out. A shared answer model then produces the final answer, and a
separate shared judge scores it. Framework-native automatic indexing is kept;
hand-authored ontologies, triples, and per-framework prompt tuning are forbidden
in the baseline run.

## Frozen baseline

- Frameworks: kontext-brain, Vector RAG + BM25-RRF reranker, Microsoft GraphRAG
  3.1.1, LightRAG 1.5.6, and HippoRAG 2 2.0.0a4.
- Embeddings: OpenAI `text-embedding-3-small`, its default 1,536 dimensions,
  with the same unprefixed text representation for documents and queries.
- Answer model: `gpt-5.6-terra`, medium reasoning, through local `codex exec`.
- Judge: `gpt-5.6-sol`, xhigh reasoning, through local `codex exec`.
- Retrieval cutoffs: candidate k=50 and output k=10 unless an official task
  interface fixes a different cutoff.
- Retrieval metrics run on every query. Answer and judge metrics use one shared,
  deterministic, proportional category-stratified sample of at most 200 queries
  per dataset (`seed=20260814`). The exact IDs, category counts, and content
  digest are frozen in each dataset's `evaluation-sample.json`.
- Answer and judge execution use one case per structured request because
  five-case structured requests repeatedly stalled before producing an output,
  while identical single-case requests completed normally. Both stages use one
  local `codex exec` process at a time because concurrent calls on the saved
  ChatGPT session stalled while a standalone call completed. Usage and latency from a
  batch are distributed across its cases without duplication. Judge requests use
  a 30-minute process timeout because xhigh reasoning has high latency variance;
  timeout changes execution recovery only, not model output settings.
- Retries: 3. Results are checkpointed after every query.
- No cross-dataset aggregate score. Every dataset/framework cell is reported
  independently with paired-bootstrap 95% confidence intervals where defined.
- Human audit: 100 blinded, framework-balanced outputs per dataset. Human labels
  are an audit only and cannot be used to tune the baseline.

The machine-readable source of truth is `manifest.ts`. Its SHA-256 digest is
written into each run report so a configuration change cannot silently mix with
earlier results.

## Dataset tracks

The primary static-KB track uses GraphRAG-Bench Medical, GraphRAG-Bench Novel,
FRAMES, GaRAGe, the kontext-specific UAEval4RAG-style set, and Stable-RAG
perturbations. All compatible frameworks receive the same raw documents and
build their own indexes automatically.

CRAG, TREC RAG, and RAGTIME are extension tracks. They keep their official
dynamic-API, very-large-corpus, or multilingual-report interfaces and are never
folded into the static-KB table.

Known data constraints are explicit benchmark outcomes:

- GaRAGe's paper describes private grounding sources and does not provide a
  complete redistributable corpus, so the full official cell remains `blocked`
  until the rights-bearing data is supplied. No similarly named substitute is
  allowed.
- CRAG needs its query-specific web pages and mock-KG resources.
- TREC RAG's MS MARCO v2 passage corpus is too large to treat as the local static
  corpus and must run on a separately provisioned large-corpus host.
- RAGTIME keeps its registered/live multilingual news workflow.
- `uaeval-kontext` and `stable-rag` are not fabricated automatically: their
  versioned `corpus.jsonl` and `queries.jsonl` must be reviewed and committed as
  benchmark assets before they become `ready`.

## External framework command contract

Microsoft GraphRAG, LightRAG, and HippoRAG 2 are invoked through isolated `uv`
projects under `bench/framework-adapters`, without modifying framework internals.
The pinned lockfiles and bundled commands are used by default. To override an
adapter command, set its environment variable to a JSON array, not a shell string:

```bash
export RAG_EVAL_GRAPHRAG_COMMAND='["/absolute/path/to/graphrag-adapter"]'
export RAG_EVAL_LIGHTRAG_COMMAND='["/absolute/path/to/lightrag-adapter"]'
export RAG_EVAL_HIPPORAG_COMMAND='["/absolute/path/to/hipporag-adapter"]'
```

`doctor` must print JSON with `status`, the exact pinned `version`, and `detail`.
`build` receives `--dataset-dir`, `--index-dir`, embedding model/dimensions,
completion model/reasoning/execution, and top-k. The adapter must honor these
shared model settings while leaving every other indexing choice at the pinned
framework's official default. `retrieve` receives the same arguments plus
`--output`, and writes the common `RetrievalResult` JSONL contract. Missing commands, version mismatches,
unsupported dataset tracks, absent credentials, and resource failures remain
visible as `blocked`, `unsupported`, or `error` records.

Integration constraints are retained in the artifacts:

- OpenAI's third-generation embedding API uses the same input contract for
  documents and queries. All compatible frameworks therefore receive the same
  unprefixed text; LightRAG's asymmetric mode is disabled for this baseline.
- GraphRAG and HippoRAG keep their native OpenAI-compatible integration
  boundary. Their local completion proxy forwards embedding calls to the same
  OpenAI model and records exact API input-token usage beside each index.
- HippoRAG 2.0.0a4 pins the now-unavailable `openai==1.91.1`; its lockfile uses
  the nearest `1.91.0` and doctor reports the exception. Its official
  `index(docs)` path also has no native long-document chunker. If a raw document
  exceeds the embedding model's 8,192-token limit, the cell is `unsupported`; the adapter
  does not silently introduce a custom chunker.

## Commands

```bash
# Prepare the official FRAMES TSV and a reproducible Wikitext snapshot of every
# referenced Wikipedia page. The page cache is resumable and generated data
# stays gitignored.
bench/node_modules/.bin/tsx bench/src/rag-eval-v2/cli.ts prepare-frames

# Inspect every model, dataset, and framework prerequisite.
bench/node_modules/.bin/tsx bench/src/rag-eval-v2/cli.ts doctor

# Wiring smoke test; this is not a benchmark score.
bench/node_modules/.bin/tsx bench/src/rag-eval-v2/cli.ts smoke \
  --frameworks kontext-brain --limit 1 \
  --work-dir /tmp/kontext-rag-eval-v2-smoke

# Full run after all required resources are ready.
bench/node_modules/.bin/tsx bench/src/rag-eval-v2/cli.ts run \
  --work-dir /absolute/path/to/rag-eval-v2-run
```

### Embedding cost and resumability

The built-in vector baseline persists each completed 100-document OpenAI
embedding batch under its index directory. A rate-limit or transient failure can
therefore be resumed with the same `--work-dir` without re-embedding completed
batches. Never reuse a Gemini-era run directory: the frozen manifest digest,
model, and dimension checks intentionally reject mixed indexes.

The OpenAI documentation price represented by the 2026-08-14 baseline is
`$0.02` per million input tokens for `text-embedding-3-small`. The conservative
budgeting envelope is about 238.5 million tokens if all five framework cells
re-embed the complete raw corpus, or about `$4.77` before framework-generated
KG text and queries. The actual total differs because kontext-brain does not use
this embedding path and some framework cells may be unsupported. This is an
estimate, not a spending guarantee. Exact successful embedding-request usage is
written to `embedding-usage.json` for the built-in vector baseline and
`openai-embedding-usage.jsonl` for external adapters.

Set `OPENAI_API_KEY` in the process environment before any embedding-backed
framework is run. The CLI also loads the repository-root `.env.local`, which is
gitignored; this lets a local run use `OPENAI_API_KEY=...` without putting the
secret in tracked files or chat.

The answer, judge, and framework-native completion paths use the saved ChatGPT
login of the local Codex CLI, not API-key billing. Every `codex exec` child
process receives a scrubbed environment with `OPENAI_API_KEY`, `CODEX_API_KEY`,
and other provider API-key variables removed. Check the active method with
`codex login status`; the frozen baseline requires `Logged in using ChatGPT`.
Codex plugins, apps, MCP-install helpers, and browser features are disabled for
these calls because the evaluator forbids tools and external knowledge.

## Metrics and artifacts

Retrieval metrics include evidence recall@k and context precision. Answer metrics
include answer correctness, claim precision/recall/F1, strict faithfulness,
citation precision/recall/F1, and acceptable abstention where a dataset supplies
the required labels. Robustness tracks report permutation sensitivity. Resource
metrics report p95 retrieval/end-to-end latency, tokens, and cost when the
provider exposes it.

Each dataset/framework directory contains `retrieval.jsonl`, `answers.jsonl`,
`judgements.jsonl`, and `score.json`. Retrieval files cover the complete query
population; answer and judgement files contain only the frozen evaluation
sample. Scores report `retrievalQueries/retrievalCompleted` separately from the
sample-level `queries/completed` counts. Dataset directories also contain
`evaluation-sample.json`, a blind human-audit file, and a separate private
mapping. `run-report.json` is the run index. Smoke runs only verify wiring and
must never be cited as comparative evidence.

Any later limited tuning experiment must be registered in `TUNING_LOG.md` before
execution and must use a separate run directory and held-out split.
