# Source-hydrated direct-only ablation

Date: 2026-08-25
Branch: `codex/adaptive-route-scoring`
Treatment: source-hydrated hybrid retrieval with graph traversal (`maxHops: 8`)
Control: the same source-hydrated stack with graph traversal disabled (`maxHops: 0`)
Status: **source hydration promoted as the quality candidate; graph traversal remains opt-in**

## Question

The raw chunk-level direct-only ablation showed that graph traversal reduced precision without a
recall advantage. Does graph traversal add value after the final evidence is fused and hydrated
into contiguous source windows?

## Isolation protocol

- All 2,062 GraphRAG-Bench Medical and all 2,010 GraphRAG-Bench Novel queries; top-k 10.
- Same corpus artifacts, frozen OpenAI document/query embeddings, vector seeds 10, lexical seeds 5,
  candidate count 20, BM25/context reranking, weighted reciprocal-rank fusion, source hydration,
  and output limit.
- Both treatments used 5,000-character source windows and a 36,000-character context cap.
- The normalized runtime configurations are identical after removing the treatment identity and
  `graphBudget.maxHops`. The treatment uses 8 hops and the control uses 0.
- The control ran with `OPENAI_API_KEY` unset and completed 4,072/4,072 queries from the same frozen
  embedding checkpoints. A missing checkpoint would have failed the run.
- Paired deltas are graph-enabled minus direct-only. Confidence intervals use 10,000 query-level
  bootstrap samples.

This is the matched experiment needed to isolate graph traversal inside the source-hydrated stack.

## Four-way score table

| Dataset | Stack | Graph hops | Recall@10 | Context precision | p95 retrieval | Avg input tokens |
|---|---|---:|---:|---:|---:|---:|
| Medical | Raw direct hybrid | 0 | 0.71532 | 0.37410 | 2.26 ms | 3,003 |
| Medical | Raw graph hybrid | 8 | 0.71435 | 0.36508 | 17.30 ms | 3,070 |
| Medical | **Source-hydrated direct** | **0** | 0.80892 | **0.65641** | **4.18 ms** | 6,619 |
| Medical | Source-hydrated graph | 8 | **0.81474** | 0.65038 | 24.41 ms | 6,660 |
| Novel | Raw direct hybrid | 0 | 0.37065 | 0.18483 | 4.37 ms | 3,030 |
| Novel | Raw graph hybrid | 8 | 0.36965 | 0.18308 | 10.21 ms | 3,068 |
| Novel | **Source-hydrated direct** | **0** | 0.43980 | **0.35696** | **12.40 ms** | 7,085 |
| Novel | Source-hydrated graph | 8 | **0.44478** | 0.35420 | 43.09 ms | 7,117 |

Compared with raw direct hybrid, source-hydrated direct improves Medical recall by 0.09360 and
precision by 0.28230. It improves Novel recall by 0.06915 and precision by 0.17214. This is the
largest robust quality improvement found in this iteration.

The raw and hydrated rows package evidence differently: raw rows return chunks, while hydrated
rows return contiguous source windows. Their score difference therefore measures the behavior of
the end-to-end evidence-selection product, not a pure ranking-only change. The matched hydrated
graph/direct comparison below uses the same evidence unit and is the causal graph ablation.

## Matched graph effect

| Dataset | Direct recall | Graph recall | Graph delta (95% CI) | Direct precision | Graph precision | Graph delta (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 0.80892 | **0.81474** | **+0.00582 [0.00048, 0.01115]** | **0.65641** | 0.65038 | **-0.00602 [-0.00934, -0.00269]** |
| Novel | 0.43980 | **0.44478** | **+0.00498 [0.00100, 0.00896]** | **0.35696** | 0.35420 | **-0.00277 [-0.00508, -0.00048]** |

Graph traversal now has a statistically supported aggregate recall benefit, but it also has a
statistically supported precision cost. It is a tradeoff, not a strict quality win.

Only a small query minority changes recall:

| Dataset | Graph recall wins | Ties | Graph recall losses | Mean top-10 window overlap |
|---|---:|---:|---:|---:|
| Medical | 22 | 2,030 | 10 | 88.56% |
| Novel | 14 | 1,992 | 4 | 93.94% |

The source hydrator replaces underlying path metadata with source-window metadata, so path-share
and direct-evidence ratios cannot be interpreted for these rows. The graph effect is instead
measured from the treatment toggle and paired outcomes.

## Regression holdout

| Dataset | Queries | Graph recall delta (95% CI) | Graph precision delta (95% CI) |
|---|---:|---:|---:|
| Medical | 422 | +0.00474 [0.00000, 0.01185] | **-0.01296 [-0.02122, -0.00512]** |
| Novel | 412 | 0.00000 [-0.00971, 0.00971] | +0.00034 [-0.00464, 0.00542] |

The aggregate graph recall gain does not generalize as a strict improvement on both holdouts.
Medical retains a positive point estimate but its interval touches zero and precision falls. Novel
shows no detectable holdout difference. This is insufficient for a graph-on default.

## Latency and context cost

Source hydration roughly doubles the packaged context tokens relative to raw chunks. Within the
matched hydrated pair, graph traversal adds about 20 ms to Medical p95 and 31 ms to Novel p95 in
these in-process runs. These are directional benchmark timings, not a production load test.

## Decision

Promote **source-hydrated direct** as the current default quality candidate. It has the best
precision on both datasets, substantially higher recall than raw direct retrieval, and materially
lower latency than source-hydrated graph traversal.

Keep graph traversal available only as an explicit recall-first policy. Do not describe it as
generally better than direct hybrid retrieval: it buys about 0.5 percentage points of aggregate
recall by giving back 0.3–0.6 points of precision, with no strict two-dataset holdout win.

A future adaptive graph gate must identify the small set of graph-win queries from runtime-only
signals and pass the same untouched holdout. The graph-admission sweeps in this iteration found no
such validation-stable policy, so per-dataset thresholds are not justified.

## Artifacts

The ignored treatment run directory is
`bench/data/rag-eval-v2/runs/adaptive-source-hydrated-v4-2026-08-25` and contains:

- `paired-hydrated-graph-vs-direct-medical.json`
- `paired-hydrated-graph-vs-direct-novel.json`

The ignored control run directory is
`bench/data/rag-eval-v2/runs/adaptive-source-hydrated-direct-only-2026-08-25`.
