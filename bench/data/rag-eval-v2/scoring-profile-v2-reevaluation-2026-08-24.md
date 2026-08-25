# calibrated-v2 scoring profile re-evaluation

Date: 2026-08-24
Status: exploratory regression evaluation; **do not promote**

## Scope

The observation-based `calibrated-v2` traversal scorer was rerun against the
existing GraphRAG-Bench Medical and Novel assets. The retrieval configuration
was held at top-k 10, vector seeds 10, lexical seeds 5, query-aware fanout 10,
200 visited nodes, and 500 candidates. Existing OpenAI document/query embedding
checkpoints were reused; retrieval checkpoints and scores were recomputed in a
new directory.

The causal comparison is the 2,062-query Medical `bidirectional-kg-v2` run. The
Novel run is a cross-domain standalone result because there is no prior Novel
run using exactly the same bidirectional scorer configuration. Prior hydrated
stack results are not treated as a scorer-only baseline.

## Headline result

| Medical metric | Existing v2 | calibrated-v2 | Delta | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Recall@10, all 2,062 | 0.70223 | 0.65567 | -0.04656 | [-0.06062, -0.03298] |
| Raw context precision, all | 0.35470 | 0.33118 | -0.02352 | [-0.02939, -0.01799] |
| Recall@10, 422-query holdout | 0.67773 | 0.64929 | -0.02844 | [-0.05924, 0.00474] |
| Precision, holdout | 0.36019 | 0.34313 | -0.01706 | [-0.02938, -0.00474] |

Both all-query metrics regressed, and their paired intervals exclude zero.
Across individual questions, recall improved on 59, tied on 1,848, and regressed
on 155. Mean top-10 evidence overlap was 0.65315. Retrieval completed without
errors on all 2,062 Medical and all 2,010 Novel questions.

Medical p95 retrieval latency changed from 10.28 ms to 11.05 ms. This small
increase is secondary to the quality regression and should not drive the
decision.

## Category breakdown

| Medical category | Queries | Existing recall | New recall | Delta |
|---|---:|---:|---:|---:|
| Complex Reasoning | 509 | 0.64637 | 0.57367 | -0.07269 |
| Contextual Summarize | 289 | 0.52595 | 0.47059 | -0.05536 |
| Fact Retrieval | 1,098 | 0.88069 | 0.84153 | -0.03916 |
| Creative Generation | 166 | 0.00000 | 0.00000 | 0.00000 |

The regression is not isolated to one question type. Complex reasoning is the
largest loss, while fact retrieval also falls materially.

## Cross-domain result

| Novel metric | All 2,010 | 412-query holdout |
|---|---:|---:|
| Recall@10 | 0.32587 | 0.33252 |
| Raw context precision | 0.16512 | 0.15461 |

The historical v4 Novel recall of 0.45771 used source hydration and a different
retrieval stack, so its difference from this run must not be attributed solely
to the scoring profile.

## Failure analysis

The regression is primarily an observation-contract and ranking failure, not
evidence that versioned profiles are the wrong architecture.

| Selected Medical top-10 path | Existing count | Existing share | New count | New share |
|---|---:|---:|---:|---:|
| Direct chunk seed | 14,122 | 68.49% | 4,551 | 22.07% |
| Entity-seed path | 6,475 | 31.40% | 2,090 | 10.14% |
| Chunk → resource → chunk | 23 | 0.11% | 13,979 | 67.79% |

1. The first entity provider rank receives a seed score of 1.0 even when the
   underlying entity match is broad, because provider rank is treated as
   comparable across node kinds. The profile has no seed-provider or seed-node
   calibration feature.
2. `resource → chunk` and `entity → chunk` candidates are query-sorted by the
   adapter, but their rank and candidate count are not included in edge
   observations. They consequently enter the priority queue with identical
   scores, where list order is not a durable ranking signal.
3. Every selected traversed edge was missing query and support observations.
   Missing values are correctly reported and neutrally scored, but neutrality
   lets broad deterministic resource expansion compete too strongly with direct
   retrieval evidence.
4. Every selected evidence item was missing confidence and freshness. Those
   profile dimensions are therefore observable in the trace but inactive in
   this dataset.
5. Removing the old triangular hop penalty was mathematically correct, but it
   exposed how much the old accidental extra distance penalty had suppressed
   broad resource fanout. A single global hop factor cannot separately calibrate
   broad resource grounding and meaningful graph traversal.

## Decision

Reject `calibrated-v2` for activation. Keep the existing policy active until a
new candidate passes paired retrieval gates. The current branch should not make
`calibrated-v2` the production fallback without another change.

Before tuning numeric weights, the next candidate must add and persist:

- query rank/candidate count on every query-sorted neighbor list;
- seed provider and seed node kind as explicit scoring observations;
- edge operation/specificity or source-fanout observations, so broad resource
  grounding can be distinguished from graph evidence traversal;
- benchmark provenance confidence/freshness only where honestly available.

After those contract changes, candidate profiles should be fitted on a new
development/validation split, evaluated against the legacy policy in shadow,
and checked on a newly frozen test split. The already inspected 422-query
Medical holdout is now an engineering regression set, not an untouched
confirmatory holdout.

## Artifacts

- `retrieval.jsonl` under each dataset/framework directory contains all newly
  computed retrievals and per-hit score breakdowns.
- `retrieval-score.json` contains the standard harness metrics.
- `runs/scoring-profile-v2-reeval-2026-08-24/paired-analysis.json` contains the
  paired split/category/path analysis and 10,000-sample bootstrap intervals.
- `run-manifest.json` and the retrieval run report record the exact benchmark
  identity.
