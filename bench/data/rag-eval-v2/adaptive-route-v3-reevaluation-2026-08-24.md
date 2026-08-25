# adaptive-route-v3 query-local route evaluation

Date: 2026-08-24
Branch: `codex/adaptive-route-scoring`
Candidate: `adaptive-route-v3` version 2
Status: **prior-policy quality gate passed; direct-only ablation failed; production activation rejected**

## What changed

The candidate keeps the versioned `calibrated-v2` base scorer but removes a globally fixed route
penalty from the most failure-prone decision. For each question and neighbor list it derives route
authority from observed fanout selectivity, reciprocal rank, and normalized query evidence. It
does not branch on dataset or organization identity.

Version 1 used only fanout and rank. It improved recall but admitted weak entity paths because the
best item in a weak route was still treated as strong. Version 2 also requires available absolute
query evidence. Missing native scores remain neutral instead of receiving an invented value.

## Protocol

- Same GraphRAG-Bench Medical and Novel corpus/query assets.
- All 2,062 Medical and all 2,010 Novel retrieval queries; top-k 10.
- Same `text-embedding-3-small`, 1,536 dimensions, vector seeds 10, lexical seeds 5.
- Same query-aware neighbor cap 10, visited budget 200, candidate budget 500.
- Existing document/query embedding batches reused. Novel was rerun with an offline embedding
  client that fails on any cache miss; it completed 2,010/2,010 with zero external requests.
- Retrieval checkpoints and scores were recomputed under the version 2 policy digest.
- Paired deltas use 10,000 bootstrap samples.

The previously inspected deterministic 20% split is reported as a regression holdout, not as a new
untouched confirmatory set.

## Headline results

| Dataset / baseline | Baseline recall | Candidate recall | Delta (95% CI) | Baseline precision | Candidate precision | Delta (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Medical previous bidirectional | 0.70223 | **0.71435** | +0.01212 [0.00000, 0.02376] | 0.35470 | **0.36508** | +0.01038 [0.00660, 0.01421] |
| Medical calibrated-v2 | 0.65567 | **0.71435** | +0.05868 [0.04316, 0.07420] | 0.33118 | **0.36508** | +0.03390 [0.02750, 0.04074] |
| Novel calibrated-v2 | 0.32587 | **0.36965** | +0.04378 [0.02886, 0.05871] | 0.16512 | **0.18308** | +0.01796 [0.01249, 0.02363] |

Against adaptive version 1, version 2 improved Medical recall by 0.02376 and precision by 0.03919;
Novel improved by 0.02637 and 0.02657. All four paired intervals exclude zero.

## Regression-holdout results

| Comparison | Queries | Recall delta (95% CI) | Precision delta (95% CI) |
|---|---:|---:|---:|
| Medical vs previous bidirectional | 422 | +0.04028 [0.01185, 0.06872] | +0.00498 [-0.00308, 0.01351] |
| Medical vs calibrated-v2 | 422 | +0.06872 [0.03318, 0.10427] | +0.02204 [0.00782, 0.03649] |
| Novel vs calibrated-v2 | 412 | +0.04369 [0.00971, 0.07767] | +0.01917 [0.00752, 0.03155] |

The Medical precision improvement over the previous policy is positive overall but inconclusive on
the smaller regression holdout. Recall remains positive there.

## Why version 2 worked

| Dataset | Adaptive v1 entity paths | v1 entity precision | Version 2 entity paths | v2 entity precision | v2 direct share |
|---|---:|---:|---:|---:|---:|
| Medical | 5,601 | 0.11891 | 1,511 | 0.22965 | 91.86% |
| Novel | 5,220 | 0.05843 | 362 | 0.15746 | 98.11% |

Version 1 confused relative rank with evidence strength. Version 2 suppresses a route when its
best candidate has weak absolute question alignment. This more than doubled entity-path precision
on both datasets while preserving a smaller set of useful graph results.

The stop condition also changed: 1,886/2,062 Medical queries and every Novel query exhausted the
eligible frontier instead of consuming the 200-node visit budget. The scorer is therefore pruning
weak graph work rather than merely reordering an equally broad traversal.

## Latency and deployment decision

| Run | Medical p95 | Novel p95 |
|---|---:|---:|
| calibrated-v2 | 11.05 ms | 16.88 ms |
| adaptive version 1 | 20.13 ms | 19.02 ms |
| adaptive version 2 | 17.30 ms | **10.21 ms** |
| previous Medical bidirectional | **10.28 ms** | n/a |

Version 2 recovered much of version 1's overhead and is faster on Novel, but Medical remains about
68% slower than the previous policy in these runs. The rollout runbook permits at most a 10% p95
increase, so the candidate must not be activated.

## What this experiment proves—and does not prove

It supports the claim that query-local route observations generalize better than one fixed route
weight set: quality moved in the same direction on Medical and Novel without dataset-specific
branches or new training data.

The follow-up direct-only ablation established that N-layer traversal does **not** beat the matched
direct hybrid control in this configuration. Graph-minus-direct recall was inconclusive on both
datasets, while precision decreased by 0.00902 on Medical and 0.00174 on Novel; both precision
intervals excluded zero. See
`adaptive-route-v3-direct-only-ablation-2026-08-25.md` for the paired analysis.

Production PostgreSQL routes also do not yet expose normalized query evidence for every
resource/entity-to-chunk edge. Those missing observations are neutral, so a production shadow run
is required to measure the real adaptive effect rather than extrapolating from the benchmark
adapter.

## Decision and next gates

Keep the implementation on the experiment branch as a route-suppression result, not as the default
graph profile. A redesigned graph candidate must pass all of these before activation:

1. a positive paired increment over the same direct-only hybrid control;
2. isolated latency profiling and removal of the Medical p95 regression;
3. production shadow traces with real PostgreSQL fanout/query observations;
4. a newly frozen confirmatory set, since the current regression holdout has been inspected;
5. answer-level faithfulness and citation evaluation, not retrieval metrics alone.

## Artifacts

The ignored raw run directory is
`bench/data/rag-eval-v2/runs/adaptive-route-v3-evidence-v2-reeval-2026-08-24`.
It contains both `retrieval.jsonl` files, standard retrieval scores, versioned score breakdowns, and
paired comparisons against calibrated-v2, adaptive version 1, and the previous Medical policy.
