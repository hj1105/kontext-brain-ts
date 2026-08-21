# Limited tuning ledger

## Baseline freeze

Status: frozen, not tuned.

The first comparison uses only official/default automatic configurations and the
shared settings in `manifest.ts`. No result from the test datasets or the 100-row
human audit may influence this baseline.

Embedding choice record (2026-08-14): the pre-score baseline was changed from
Gemini Embedding 2 to OpenAI `text-embedding-3-small` at its default 1,536
dimensions after the Gemini free quota proved too small for the corpus. The
choice was made before comparative scores existed, applies identically to every
compatible framework, and is a baseline infrastructure decision rather than a
tuning trial. Any later model or dimension comparison must be registered below
and use a separate development split and run directory.

Evaluation execution record (2026-08-14): all retrieval queries remain in the
baseline. Answer and judge evaluation uses a shared deterministic proportional
category-stratified sample of 200 queries per dataset (`seed=20260814`), frozen
before framework scores exist. Local Codex answer and judge execution use one
case per structured request with concurrency one. These values are
resource-control settings fixed equally across frameworks, not parameters
selected from scores.

Execution reliability record (2026-08-14): the judge process timeout was raised
from 10 to 30 minutes after three of four concurrent `gpt-5.6-sol` xhigh batches
hit the 10-minute process limit. Model, reasoning effort, prompts, batch size,
concurrency, seed, and retry count are unchanged. This change governs recovery
from incomplete processes only and was made before the affected wave produced a
score. Timed-out processes now receive `SIGTERM`, then `SIGKILL` after two
seconds so a child that ignores graceful shutdown cannot stall the run.
Provider API-key variables and Codex plugin, app, MCP-install, and browser
features are removed from evaluator child processes. This enforces the frozen
no-tools prompt boundary and prevents unrelated remote catalog initialization;
it does not change the model, benchmark evidence, judge rubric, or schema.

Structured batching amendment (2026-08-15): five-case `gpt-5.6-sol` xhigh judge
requests repeatedly produced no result before both 10- and 30-minute process
limits. After unrelated Codex plugin features were disabled, the same real
five-case request still timed out at 10 minutes, while its first case completed
alone in 21.5 seconds with the same model, prompt template, evidence, rubric,
and output schema. Five-case `gpt-5.6-terra` answer requests then showed the
same failure, while the identical first answer case completed alone in 5.1
seconds. Answer and judge batch sizes are therefore one. Four concurrent
single-case answer requests then reproduced the transport failure: one made
progress while three hit the three-minute limit, whereas the same request alone
completed. Codex execution concurrency is therefore one. The incomplete mixed-batch diagnostic run
at `bench/data/rag-eval-v2/runs/openai-small-defaults-2026-08-14` is preserved
but cannot be reported as the comparison baseline. The subsequent incomplete
judge-single and single-case concurrent runs are also diagnostic-only. The
consistent replacement run uses
`openai-small-defaults-single-case-serial-2026-08-15`.

Embedding transport amendment (2026-08-17): during the frozen Medical vector
baseline, the OpenAI embedding endpoint intermittently raised `fetch failed`
and once returned HTTP 200 with an empty embedding array. The first 1,200
document embeddings had already been committed in 100-document checkpoint
batches. The client now applies the existing retry budget to network exceptions
and incomplete successful responses inside the current checkpoint batch before
the retrieval-stage retry is used. The model, 1,536 dimensions, 100-document
batch size, corpus, query set, and retry budget are unchanged; the same run
directory resumes from the committed checkpoints. This is failure recovery,
not retrieval or model tuning.

LightRAG retrieval scheduling amendment (2026-08-18): the first Medical
LightRAG retrieval attempt processed 103 of 2,062 queries in about 1 hour 45
minutes because the adapter awaited each independent query serially. LightRAG 1.5.6
defines `DEFAULT_MAX_ASYNC = 4`, and its native indexing had already completed
more than an hour of four-way Codex execution without a timeout or failed
chunk. The retrieval adapter now limits whole-query concurrency to the
instantiated framework's `rag.llm_model_max_async` value and uses
`asyncio.gather`, which preserves the frozen input order in the output file.
The incomplete attempt wrote no retrieval output and is restarted against the
same completed index. Model, prompts, native `mix` mode, reranking setting,
top-k, corpus, query order, and evaluation sample are unchanged. This restores
the framework's official default resource scheduling; it is not score-driven
tuning.

Dataset scheduling amendment (2026-08-19): the baseline runner now supports a
retrieval-only stage so independent dataset indexes and retrievals can be
materialized in parallel. Answer generation and judging remain in the original
single-process, single-case, concurrency-one path and consume the frozen
retrieval artifacts afterward. Retrieval-only reports use dataset-qualified
filenames so concurrent workers cannot overwrite the final report. LightRAG
also commits one adapter result checkpoint per query and restores checkpoints
in frozen input order after an interrupted worker. These are execution and
failure-recovery controls only: framework defaults, corpus, query order, top-k,
models, prompts, answer sample, judge rubric, and metrics are unchanged.

Aggressive scheduling amendment (2026-08-19): at the operator's explicit
request to minimize wall-clock time, the independent FRAMES Kontext,
Microsoft GraphRAG, LightRAG, and HippoRAG retrieval workers are launched
concurrently instead of waiting for each framework to finish. Novel retrieval
workers join once the short Medical answer/judge tail releases capacity. Each
framework retains its own official internal concurrency and frozen defaults;
this changes only cross-dataset/cross-framework scheduling. Checkpoints remain
authoritative after memory pressure, transport failure, or worker restart.

Dataset scope amendment (2026-08-19): the operator removed FRAMES from the
active comparison after measured corpus size and Medical throughput projected
roughly 9--12 days for each LLM-indexed FRAMES framework. Partially generated
FRAMES artifacts are preserved but are not scored or included in the final
comparison. The active baseline scope is GraphRAG-Bench Medical and Novel,
covering ten dataset/framework cells. This is an operator-directed runtime
scope decision, not a result-driven tuning choice; no completed score informed
the removal.

## Rules for later experiments

A limited tuning experiment must be recorded here before it starts. It must:

1. state one hypothesis and one bounded parameter family;
2. give every compatible framework the same trial budget;
3. use a versioned development split separate from the final test split;
4. keep answer and judge models frozen;
5. use a new output directory and preserve the baseline artifacts;
6. report all attempted settings, including failures; and
7. never replace baseline results with tuned results in the same table.

Copy this template for each experiment:

```text
Experiment ID:
Registered at (UTC):
Owner:
Hypothesis:
Dataset and development split digest:
Eligible frameworks:
Parameter family and allowed values:
Maximum trials per framework:
Selection metric:
Held-out test run directory:
Outcome:
```

## Exploratory experiment: Medical Kontext retrieval v2

```text
Experiment ID: medical-kontext-bidirectional-kg-v2-2026-08-19
Registered at (UTC): 2026-08-19T08:00:00Z (formalized after the exploratory
  development trials; this is not a preregistered result)
Owner: benchmark operator / Codex
Hypothesis: high graph fanout fills maxCandidates before lower-ranked direct
  seeds are evaluated; preserving direct-seed evidence and adding a small
  lexical/query-aware seed set will recover recall without expanding top-k.
Dataset and development split digest:
  GraphRAG-Bench Medical; SHA-256(queryId) bucket, 1,640 development / 422
  holdout; 76ea2431c21de356010e2da02a416112fc4542b8e3d0dcab8d594a0ad1051af4
Eligible frameworks: kontext-brain only (product-improvement experiment, not a
  symmetric framework-comparison tuning budget)
Parameter family and allowed values: vector seeds {5,8,10,12,15,20}; lexical
  seeds {0,3,5,7,10}; common graph fanout {5,8,10,12}; query-aware {false,true}
Maximum trials per framework: exploratory; 21 recorded observations
Selection metric: development evidence recall@10, with context precision as a
  guardrail
Held-out test run directory:
  openai-small-kontext-bidirectional-kg-v2-2026-08-19
Outcome: selected vector=10, lexical=5, query-aware=true, graph fanout=10.
  All-query recall@10=0.70223; holdout recall@10=0.67773; holdout context
  precision=0.36019. The strict 0.70 holdout gate remains red. The frozen
  200-query answer score is correctness=0.83627, claim-F1=0.78436,
  faithfulness=0.87884, and citation-F1=0.87890.
```

Development recall@10 observations, including superseded trials:

| Retriever state | Vector seeds | Lexical seeds | Common graph fanout | Query-aware | Dev recall@10 |
|---|---:|---:|---:|:---:|---:|
| Before core fix, baseline | 5 | 0 | broad/default | false | 0.4933 |
| Before core fix, lean | 10 | 5 | lean | true | 0.6854 |
| Before core fix, tight | 20 | 10 | tight | true | 0.6963 |
| After core fix, baseline | 5 | 0 | broad/default | false | 0.5811 |
| After core fix | 10 | 0 | broad/default | false | 0.5896 |
| After core fix | 10 | 0 | broad/default | true | 0.6280 |
| After core fix, lean | 10 | 0 | lean | true | 0.6445 |
| After core fix, lean | 10 | 5 | lean | true | 0.7055 |
| After core fix, lean | 20 | 5 | lean | true | 0.7012 |
| After core fix, tight | 20 | 10 | tight | true | 0.6963 |
| After core fix | 8 | 3 | 8 | true | 0.6835 |
| After core fix | 8 | 5 | 8 | true | 0.7012 |
| After core fix | 8 | 7 | 8 | true | 0.6982 |
| After core fix | 10 | 5 | 5 | true | 0.6963 |
| After core fix | 10 | 3 | 8 | true | 0.6921 |
| After core fix | 10 | 5 | 8 | true | 0.7012 |
| After core fix | 10 | 7 | 8 | true | 0.6963 |
| After core fix | 10 | 5 | 12 | true | 0.7037 |
| After core fix | 12 | 5 | 8 | true | 0.7012 |
| After core fix | 12 | 5 | 10 | true | 0.7037 |
| After core fix | 15 | 5 | 8 | true | 0.7012 |

The exploratory table has more observations than the initially intended
limited budget and was documented post hoc. Consequently it is useful for
engineering diagnosis but must not be presented as a confirmatory benchmark.
The fixed holdout and isolated output directory are retained to make that
boundary auditable.

## Source-native hydration and LLM rerank experiment

```text
Experiment ID: kontext-source-hydrated-llm-stack-v5-2026-08-20
Registered at (UTC): 2026-08-20T07:09:00Z, after a two-query execution smoke
  test and before any complete v5 retrieval or answer score existed
Owner: benchmark operator / Codex
Hypothesis: retrieval chunks should remain ontology/index anchors, while answer
  evidence should be restored from bounded source-native windows; over-retrieval
  followed by a gold-blind LLM reranker should recover candidate recall without
  passing every graph-adjacent distractor to the answer model.
Dataset and development split digest: no dataset-specific development split or
  score-driven selection. The same frozen policy is executed independently on
  all 2,062 Medical and 2,010 Novel queries as a cross-domain guardrail.
Eligible frameworks: kontext-brain only (product architecture experiment)
Parameter family and allowed values: fixed only -- 20 fused candidates, local
  Codex answer model rerank to 10 anchors, 5,000 source characters per anchor,
  overlapping windows merged, 36,000 total source characters. The reranker sees
  question and candidate literal text only; dataset ID, reference answer, gold
  evidence, and judge output are excluded.
Maximum trials per framework: one fixed v5 policy. The preceding hydration-only
  v4 is retained as an ablation, not used to fit a v5 value.
Selection metric: report Medical and Novel evidence recall/context usage
  separately; then run the already frozen Medical 200-query answer/judge sample.
Held-out test run directory:
  openai-small-kontext-v5-source-hydrated-llm-stack-2026-08-20
Outcome: Medical retrieval completed for all 2,062 queries with evidence recall
  0.78952 and raw context precision 0.69422. This improved precision over v4
  (0.63349) but regressed recall from v4 (0.82832), so v5 was rejected before
  answer/judge scoring. Novel stopped after 158 checkpointed queries to avoid
  spending further compute on a rejected policy; those checkpoints are retained.
```

The v4 ablation used 10 fused anchors with the same 5,000/36,000 source policy.
It raised Medical evidence recall from v3 0.71629 to 0.82832, but lowered frozen
answer correctness from 0.85197 to 0.83807 because expanded source windows also
introduced distractors. Novel recall was 0.45771. Those results motivated the
pre-answer, gold-blind reranking seam; they did not select a dataset-specific
weight, keyword, window size, or ontology rule.

## Recall-safe LLM rerank experiment

```text
Experiment ID: kontext-source-hydrated-llm-recall-safe-v6-2026-08-20
Registered at (UTC): 2026-08-20T07:30:00Z, before any v6 retrieval existed
Owner: benchmark operator / Codex
Hypothesis: an LLM reranker should order the full over-retrieved candidate set,
  while a bounded source-context budget decides how much evidence survives.
  Truncating ranked anchors to ten before hydration is an avoidable recall loss.
Dataset and development split digest: no dataset-specific development split or
  query rules. Execute the same policy on all 2,062 Medical and 2,010 Novel
  queries; Novel is the cross-domain generalization guardrail.
Eligible frameworks: kontext-brain only (product architecture experiment)
Parameter family and allowed values: fixed only -- rank all 20 fused candidates
  with the local Codex answer model, restore 5,000 source characters around each
  ranked anchor, merge overlapping windows, stop at a 50,000-character global
  budget (10 non-overlapping source windows at the declared 5,000-character
  unit). Dataset ID, reference answer, gold evidence, and judge output remain
  unavailable to the reranker.
Maximum trials per framework: one v6 policy. No Medical keyword, question ID,
  ontology rule, score threshold, or per-domain parameter is permitted.
Selection metric: require Medical recall recovery and report Novel independently;
  only then run the frozen Medical 200-query answer/judge sample.
Held-out test run directory:
  openai-small-kontext-v6-source-hydrated-llm-recall-safe-2026-08-20
Outcome: completed. Medical retrieval recall 0.88749, raw context precision
  0.51999; frozen 200-query answer correctness 0.87549, strict faithfulness
  0.91962, citation F1 0.93281. Novel retrieval recall 0.50846 and raw context
  precision 0.24600. The policy improved both domains over v4 and exceeded the
  vector baseline on Medical correctness, but did not exceed LightRAG on Medical
  recall/correctness or MS GraphRAG/vector retrieval recall on Novel.
```

## Declared candidate-k contract experiment

```text
Experiment ID: kontext-source-hydrated-llm-candidate-safe-v7-2026-08-20
Registered at (UTC): 2026-08-20T08:15:00Z, before any v7 retrieval existed
Owner: benchmark operator / Codex
Hypothesis: the max-stack adapter incorrectly ignores the benchmark's declared
  --candidate-k=50 and truncates vector, BM25, graph, and fused candidates at an
  internal constant of 20. Honoring the shared candidate contract should improve
  cross-domain candidate recall without a dataset-specific rule.
Dataset and development split digest: no dataset-specific split or parameter.
  Run the identical declared candidate-k=50 policy on all 2,062 Medical and 2,010
  Novel queries; Novel remains the generalization guardrail.
Eligible frameworks: kontext-brain only (adapter contract correction)
Parameter family and allowed values: candidate count comes only from the existing
  CLI benchmark contract (50), not a result-driven search. The local Codex model
  ranks all 50; source windows remain 5,000 characters with the existing 50,000
  global budget. All v6 fusion weights, KG traversal settings, prompts, and source
  policy are frozen.
Maximum trials per framework: one v7 contract-corrected policy.
Selection metric: Medical and Novel evidence recall/context usage, followed by
  the same frozen Medical 200-query answer/judge sample if retrieval is viable.
Held-out test run directory:
  openai-small-kontext-v7-source-hydrated-llm-candidate-safe-2026-08-20
Outcome: completed. Medical retrieval recall 0.89040, raw context precision
  0.57597; frozen 200-query answer correctness 0.88035, strict faithfulness
  0.93195, citation F1 0.91789, and claim F1 0.85291. Novel retrieval recall
  0.51741 and raw context precision 0.28820. Honoring candidate-k improved
  precision and answer quality over v6 and produced a clear Medical quality win
  over Microsoft GraphRAG and the vector baseline. It remained slightly behind
  LightRAG on Medical quality, and Novel recall remained behind Microsoft
  GraphRAG/vector, so no universal-best claim is permitted.
```

## Coverage-aware LLM rerank experiment

```text
Experiment ID: kontext-coverage-aware-rerank-v10-2026-08-21
Registered at (UTC): 2026-08-21T08:40:00Z, before implementation and before any
  v10 retrieval exists
Owner: benchmark operator / Codex
Hypothesis: the v7 reranker optimizes each candidate's direct support in
  isolation and can rank redundant passages above complementary evidence needed
  by multi-hop questions. A source-only coverage instruction should make the
  leading ranked anchors cover distinct entities, constraints, events, and
  temporal or causal steps while retaining explicit answer-bearing passages.
Dataset and development split digest: no score-driven parameter selection. Run
  the one fixed policy on all 2,062 Medical queries; only if it improves the v7
  retrieval gate, run the identical policy on all 2,010 Novel queries and the
  frozen Medical 200-query answer/judge sample. Additional public datasets are
  an external generalization gate once their canonical adapters are prepared.
Eligible frameworks: kontext-brain only (product retrieval-policy experiment)
Parameter family and allowed values: one fixed prompt policy. Candidate-k=50,
  base GraphRAG-Bench KG, embedding model, vector/BM25/graph/context fusion,
  graph fanout and budgets, 5,000-character source windows, 50,000-character
  total context budget, local model, reasoning effort, and output contract all
  remain identical to v7. Dataset ID, reference answer, gold evidence, and judge
  output are unavailable to the reranker.
Maximum trials per framework: one v10 policy
Selection metric: Medical evidence recall@10 must exceed 0.890398 without lower
  raw context precision than 0.575969. Novel must then avoid a material recall
  or precision regression relative to v7 before answer/judge promotion.
Held-out test run directory:
  openai-small-kontext-v10-coverage-aware-rerank-2026-08-21
Outcome: completed, not promoted. Medical retrieval completed 2,062/2,062
  with 0 errors. Evidence recall@10 was 0.891368 (+0.000970 versus v7),
  while raw context precision was 0.575153 (-0.000816 versus v7). The fixed
  joint gate required both recall above 0.890398 and precision at least
  0.575969, so coverage-aware reranking alone failed the precision condition.
  It is retained as the single-query ablation and is not run on Novel or
  answer/judge.
```

## Multi-query × evidence-selection factorial

```text
Experiment ID: kontext-multi-query-factorial-v11-2026-08-21
Registered at (UTC): 2026-08-21T08:43:00Z, before any v11 retrieval or score
Owner: benchmark operator / Codex
Hypothesis: complementary search perspectives can recover bridge evidence that
  a single embedding/BM25 query misses, while coverage-aware selection can keep
  those complementary passages ahead of redundant passages. Compare the fixed
  2×2 cells without score-driven parameter search: existing v7 (single query,
  standard rerank), v10 (single query, coverage-aware rerank), v11a (multi-query,
  standard rerank), and v11b (multi-query, coverage-aware rerank).
Dataset and development split digest: run all 2,062 Medical retrieval queries in
  v11a and v11b. Promote only a predeclared winner that beats v7 recall 0.890398
  without reducing raw context precision below 0.575969. Then run the identical
  configuration on all 2,010 Novel queries, BEIR SciFact/NFCorpus retrieval-only
  gates, and the frozen Medical 200-query answer/judge sample.
Eligible frameworks: kontext-brain only (product retrieval-policy experiment)
Parameter family and allowed values: original query is always retained. One
  local GPT call may produce at most three distinct standalone search queries.
  Original and expanded vector lists receive equal-weight RRF; original and
  expanded BM25 lists receive equal-weight RRF. Candidate-k=50, outer fusion,
  graph traversal, 5,000-character source windows, 50,000-character context
  budget, models, reasoning effort, and output top-k=10 remain fixed from v7.
  v11a uses the v7 rerank instruction; v11b uses the exact v10 coverage-aware
  instruction. No dataset ID, corpus text, reference answer, gold evidence, or
  judge output is available to query expansion.
Maximum trials per framework: exactly two new Medical cells (v11a and v11b)
Selection metric: the same joint Medical recall/raw-context-precision gate above;
  ties are broken by lower retrieval latency, then checked on Novel and public
  datasets. No per-dataset branch or post-score parameter adjustment is allowed.
Held-out test run directories:
  openai-small-kontext-v11a-multi-query-standard-2026-08-21
  openai-small-kontext-v11b-multi-query-coverage-2026-08-21
Outcome: pending
```

## Query-plan-aware coverage selection

```text
Experiment ID: kontext-query-plan-aware-coverage-v12-2026-08-21
Registered at (UTC): 2026-08-21T09:11:00Z, before any v11 score and before any
  v12 retrieval exists
Owner: benchmark operator / Codex
Hypothesis: v11b generates complementary search perspectives but its final
  selector sees only the original question, forcing it to infer the evidence
  decomposition again. Supplying those same question-derived perspectives as
  a non-factual coverage plan should improve complementary evidence selection
  without exposing corpus, answer, or evaluation metadata.
Dataset and development split digest: reuse the exact 2,062 Medical expansion
  checkpoints, 5,893 expanded query strings, embeddings, candidate-k=50, and
  retrieval/fusion policy from v11b. Only the coverage reranker context differs.
  Promote only if the fixed Medical joint gate is passed, then apply unchanged
  to Novel and BEIR SciFact/NFCorpus.
Eligible frameworks: kontext-brain only
Parameter family and allowed values: one fixed boolean, queryPlanAware=true.
  The original question remains the actual answer request. Query-derived needs
  are explicitly labeled as search cues, not facts or answers. Candidate texts
  remain untrusted. All v11b model, reasoning, graph, hydration, budget, and
  fusion settings remain fixed. No dataset ID, corpus metadata, reference
  answer, gold evidence, or judge output is available.
Maximum trials per framework: one v12 policy
Selection metric: Medical evidence recall@10 > 0.890398 and raw context
  precision >= 0.575969; then no material Novel/BEIR regression.
Held-out test run directory:
  openai-small-kontext-v12-multi-query-plan-aware-2026-08-21
Outcome: pending
```
