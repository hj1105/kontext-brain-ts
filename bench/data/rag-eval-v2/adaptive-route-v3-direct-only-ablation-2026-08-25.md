# adaptive-route-v3 direct-only ablation

Date: 2026-08-25
Branch: `codex/adaptive-route-scoring`
Candidate: graph-enabled `adaptive-route-v3` version 2
Control: identical hybrid retrieval with graph traversal disabled
Status: **graph advantage not demonstrated; production activation rejected**

## Question

Does the adaptive N-layer graph traversal improve retrieval over the direct hybrid candidate pool,
or did the earlier gains come from better direct vector/lexical retrieval and graph suppression?

## Isolation protocol

- All 2,062 GraphRAG-Bench Medical and all 2,010 GraphRAG-Bench Novel queries; top-k 10.
- Same corpus artifacts, cached document/query embeddings, vector seeds 10, lexical seeds 5, seed
  fusion, evidence scorer, scoring profile, and feature schema.
- The graph-enabled candidate used the normal traversal budget. The direct-only control used the
  same retriever with `maxHops: 0`, so no neighbor expansion could contribute evidence.
- The two runs have distinct framework versions and configuration digests, preventing result-cache
  reuse across treatments.
- `candidatePoolDigest` and scoring-profile digest match between treatments on both datasets.
- The direct-only run was executed with `OPENAI_API_KEY` unset. All 4,072 queries completed from
  the frozen embedding cache, and any cache miss would have failed the run.
- Paired deltas below are graph-enabled minus direct-only. Confidence intervals use 10,000
  query-level bootstrap samples.

This isolates the incremental effect of graph traversal within the current implementation. It is
not a comparison against every possible direct retriever or every possible graph architecture.

## Headline result

| Dataset | Direct recall | Graph recall | Graph delta (95% CI) | Direct precision | Graph precision | Graph delta (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Medical | **0.71532** | 0.71435 | -0.00097 [-0.00679, 0.00485] | **0.37410** | 0.36508 | **-0.00902 [-0.01091, -0.00713]** |
| Novel | **0.37065** | 0.36965 | -0.00100 [-0.00398, 0.00199] | **0.18483** | 0.18308 | **-0.00174 [-0.00269, -0.00085]** |

Graph traversal did not produce a statistically distinguishable recall improvement on either
dataset. It produced a statistically supported precision decrease on both datasets.

## Paired behavior

| Dataset | Graph recall wins | Ties | Graph recall losses | Mean top-10 overlap | Graph-selected path share |
|---|---:|---:|---:|---:|---:|
| Medical | 18 | 2,024 | 20 | 93.02% | 8.14% |
| Novel | 4 | 2,000 | 6 | 98.24% | 1.89% |

The graph changed relatively few recall outcomes and lost slightly more often than it won. No
query category had a recall delta whose 95% interval excluded zero. Medical Complex Reasoning and
Novel Complex Reasoning had small positive point estimates, but both also lost precision and their
recall intervals crossed zero.

## Regression-holdout result

| Dataset | Queries | Graph recall delta (95% CI) | Graph precision delta (95% CI) |
|---|---:|---:|---:|
| Medical | 422 | 0.00000 [-0.01422, 0.01422] | **-0.00735 [-0.01161, -0.00332]** |
| Novel | 412 | +0.00243 [-0.00485, 0.00971] | **-0.00218 [-0.00388, -0.00049]** |

The same pattern remains on the previously inspected deterministic holdout: recall is
inconclusive, while precision is lower with graph traversal.

## Why precision fell

The adaptive policy correctly suppressed most graph routes, but the graph evidence that remained
was still weaker than the direct alternatives it displaced:

| Dataset | Selected graph evidence | Entity-path precision | Resource-path precision | Direct-only precision |
|---|---:|---:|---:|---:|
| Medical | 1,678 | 0.22965 | 0.00000 | 0.37410 |
| Novel | 380 | 0.15746 | 0.27778 | 0.18483 |

The earlier entity-path improvement from 0.11891 to 0.22965 was therefore real but insufficient.
It means the filter removed many bad entity paths; it does not mean the surviving graph paths beat
the direct candidates at the top-10 decision boundary.

## Latency

| Dataset | Direct-only p95 | Graph-enabled p95 | Graph/direct ratio |
|---|---:|---:|---:|
| Medical | **2.26 ms** | 17.30 ms | 7.65x |
| Novel | **4.37 ms** | 10.21 ms | 2.34x |

These are in-process benchmark timings rather than a production load test, but the direction is
unambiguous: the graph path adds work while not improving the measured retrieval quality.

## Decision

The current evidence rejects the claim that this graph-enabled configuration is better than its
direct hybrid control. The defensible claim is narrower: query-local gating is effective at
suppressing weak graph traversal compared with earlier graph policies.

Do not activate graph traversal by default. A next graph candidate should only run when it can
demonstrate incremental value over the direct score at the top-k boundary—for example, by using
graph evidence as a fill-only fallback or requiring a calibrated margin over the direct candidate
that would be displaced. Any redesign must pass this same paired direct-only test and an
answer-level faithfulness/citation evaluation.

## Artifacts

The ignored direct-only run directory is
`bench/data/rag-eval-v2/runs/adaptive-route-v3-direct-only-ablation-2026-08-25`.
It contains both retrieval files, standard scores, and:

- `paired-graph-vs-direct-medical.json`
- `paired-graph-vs-direct-novel.json`

The graph-enabled treatment is
`bench/data/rag-eval-v2/runs/adaptive-route-v3-evidence-v2-reeval-2026-08-24`.
