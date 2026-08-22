# Kontext Brain RAG final comparison (2026-08-22)

## Decision

Promote **v13 anchored evidence answer stack**. It is the only tested
configuration that improves the frozen Medical end-to-end metrics while also
improving Novel and preserving BEIR retrieval recall without dataset-specific
branches. v14a/v14b are retained as useful precision-oriented ablations, not as
the product default.

The retrieval and answer paths never receive dataset names, reference answers,
gold evidence, or judge outputs. All four datasets use the same fixed v13
weights and policies: original-query weight 2, each expansion weight 1, RRF
k=10, at most three expansions, candidate-k 50, source hydration capped at
5,000 characters per window and 50,000 characters total, plus the
supported-evidence-needs answer contract.

## Medical end-to-end comparison

All answer metrics use the same frozen 200-query sample. Context precision is
raw evidence-unit precision and is not directly comparable with LightRAG's
single bundled-context packaging.

| System | Retrieval | Recall@10 | Raw context precision | Correctness | Strict faithfulness | Claim F1 | Citation F1 | Retrieval p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v7 | 2,062/2,062 | 0.890398 | 0.575969 | 0.880350 | 0.931951 | 0.852906 | 0.917894 | 17.62 s | 123.79 s |
| v12 | 2,062/2,062 | 0.891368 | 0.583434 | 0.865488 | 0.912990 | 0.843040 | 0.929133 | 76.58 s | 166.60 s |
| **v13** | **2,062/2,062** | **0.892338** | **0.580222** | **0.946127** | **0.953411** | **0.855016** | **0.954141** | **34.94 s** | **104.24 s** |
| LightRAG | 2,062/2,062 | 0.932590 | 0.999030* | 0.893900 | 0.941684 | 0.857511 | 0.947662 | 377.92 s | 445.01 s |

`*` LightRAG packages a very large context as one evidence item, so its context
precision is package-sensitive and not an apples-to-apples raw precision score.

v13 versus LightRAG: correctness +0.052227, strict faithfulness +0.011728,
citation F1 +0.006480, claim F1 -0.002495, and recall -0.040252. v13 retrieval
p95 is about 10.8x faster and end-to-end p95 about 4.3x faster in these runs.

v13 versus v7: recall +0.001940, raw context precision +0.004253,
correctness +0.065777, strict faithfulness +0.021461, claim F1 +0.002110,
and citation F1 +0.036247.

## Generalization guardrails

| Dataset | System | Completed | Recall@10 | Raw context precision |
|---|---|---:|---:|---:|
| Novel | v7 | 2,010/2,010 | 0.517413 | 0.288200 |
| Novel | v12 | 2,010/2,010 | 0.522886 | 0.287472 |
| Novel | **v13** | **2,010/2,010** | **0.525871** | **0.289687** |
| SciFact | vector baseline | 300/300 | 0.822444 | 0.163667 |
| SciFact | v12 | 300/300 | 0.960000 | 0.034727 |
| SciFact | **v13** | **300/300** | **0.960000** | **0.034686** |
| NFCorpus | vector baseline | 323/323 | 0.162923 | 0.308978 |
| NFCorpus | v12 | 323/323 | 0.278269 | 0.175353 |
| NFCorpus | **v13** | **323/323** | **0.278486** | **0.174640** |

Novel improves over v12 on both metrics. SciFact preserves v12's 0.96 recall,
and NFCorpus slightly improves recall. This is the expected pattern for a
general retrieval policy rather than a Medical-only fit.

## v14 deterministic ablations

v14a uses anchored soft RRF without an LLM reranker. v14b additionally forces
five original-query candidates and one unique candidate from every available
expansion into the top-10 window. Both read frozen expansion/embedding caches
and record zero new Codex calls, embedding calls, and input tokens.

| Dataset | v14a recall / precision | v14b recall / precision |
|---|---:|---:|
| Medical | 0.833657 / 0.654693 | 0.831232 / 0.658388 |
| Novel | 0.460697 / 0.352307 | 0.452736 / 0.352059 |
| SciFact | 0.883000 / 0.100333 | 0.883667 / 0.100667 |
| NFCorpus | 0.187594 / 0.288545 | 0.190268 / 0.288235 |

The effect is consistent across domains: precision rises, but recall falls too
far. Neither v14 policy is promoted.

## Auditable token and embedding cost

For Medical v13 answer plus judge calls, the recorded totals are 9,029,688
input tokens and 268,838 output tokens across 200 queries. This is 45,148 input
tokens and 1,344 output tokens per judged query. Local GPT CLI usage is not
priced by these artifacts, so it must not be represented as a dollar cost.

OpenAI `text-embedding-3-small` is priced in the run manifests at $0.02 per
million input tokens:

| Dataset | Cached/indexed embedding tokens | Auditable cost |
|---|---:|---:|
| Medical | 407,518 | $0.008150 |
| Novel | 891,981 | $0.017840 |
| SciFact | 1,680,151 | $0.033603 |
| NFCorpus | 1,253,777 | $0.025076 |
| **All four** | **4,233,427** | **$0.084669** |

Illustrative monthly floors, excluding any separately priced generation or
judge service: rebuilding all four indexes daily is about $2.54/month;
rebuilding Medical and Novel daily is about $0.78/month. Actual production cost
depends on corpus churn and query volume. v14a/v14b incurred zero new embedding
or expansion/reranker tokens because all cache access was read-only.

## Artifact integrity

- v13 Medical: `openai-small-kontext-v13-anchored-evidence-answer-medical-2026-08-22`
- v13 Novel: `openai-small-kontext-v13-anchored-evidence-answer-novel-2026-08-22`
- v13 SciFact: `openai-small-kontext-v13-anchored-evidence-answer-scifact-2026-08-22`
- v13 NFCorpus: `openai-small-kontext-v13-anchored-evidence-answer-nfcorpus-2026-08-22`
- v14 final paths follow the registered `v14a-cache-only-soft-coverage` and
  `v14b-cache-only-quota-coverage` names in `TUNING_LOG.md`.

Every final retrieval cell has zero final errors. Medical v13 has 200/200
answers and 200/200 judgements. Aborted wrong-mode startup directories and the
initial strict-cache NFCorpus failures were renamed with `aborted-` or
`failed-` prefixes and are not inputs to this report.
