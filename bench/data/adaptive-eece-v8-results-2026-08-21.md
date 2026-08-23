# Adaptive EECE v8 and current-head v9 evaluation — 2026-08-21

## Scope and guardrails

- Historical v8 implementation: `996b8bb` on `codex/adaptive-knowledge-graph` (PR #3).
- Current-head v9 implementation: `edb4ab0`, including independent explicit-Claim verification, Resource-wide identity resolution, prior-active identity reuse, and all-or-nothing enrichment.
- Retrieval stack: v7 source hydration, declared candidate `k=50`, recall-safe local GPT rerank, plus the adaptive Entity–Event–Claim–Evidence graph.
- Datasets: GraphRAG-Bench Medical and Novel. FRAMES was out of scope.
- The builder read source chunks and the pre-existing KG only. It did not read questions, reference answers, gold evidence, or dataset names when selecting capabilities or extracting knowledge.
- Capability selection remained source-driven and domain-independent.
- Invalid citations and references never entered the graph. The evaluation artifact builder retried five times and then withheld the whole invalid window.
- Post-evaluation review removed tolerant failure handling from the product API. Current PR code fails the whole enrichment after three exhausted attempts, independently verifies explicit Claim support, resolves identity across all Resource windows, and reuses prior active IDs through a separate sync input without reviving stale Mentions. The v9 replay below uses these semantics with a fixed evaluation retry allowance of eight attempts.
- LLM work used the local Codex CLI. OpenAI API usage was limited to `text-embedding-3-small`.

## Fixed extraction policy

| Setting | Value |
| --- | ---: |
| Chunks per window | 6 |
| Overlap | 1 |
| Maximum window characters | 12,000 |
| Product default attempts | 3 |
| v8 evaluation attempts | 5 |
| v8 artifact failure handling | Withhold exhausted window |
| v9 evaluation attempts | 8, fixed for both datasets |
| v9 artifact failure handling | Reject the whole Resource |
| Current product failure handling | Reject whole enrichment |
| Extraction concurrency per Resource | 10 |
| Resource concurrency | 2 |
| Local model | `gpt-5.6-terra`, medium reasoning |

No retrieval weights, fanout, budgets, prompts, or thresholds were fitted to Medical. The evaluation used the existing v7 retrieval defaults.

## KG generation

| Dataset | Chunks | Resources | Windows | Withheld windows | Adaptive entities | Explicit Facts | Withheld Hypotheses | Final entities | Final Facts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Medical | 1,385 | 1 | 277 | 1 (0.36%) | 2,209 | 1,909 | 146 | 4,645 | 11,431 |
| Novel | 3,503 | 11 | 705 | 7 (0.99%) | 5,931 | 4,803 | 514 | 8,900 | 8,830 |

Local model usage was 1,010 calls / 13.36M input tokens for Medical and 2,472 calls / 33.00M input tokens for Novel. Provider API cost was `$0`.

## Retrieval results

| Dataset | Configuration | Completed | Recall@10 | Raw context precision |
| --- | --- | ---: | ---: | ---: |
| Medical | v7 | 2,062/2,062 | 0.890398 | 0.575969 |
| Medical | adaptive EECE v8 | 2,062/2,062 | 0.888943 | 0.573499 |
| Novel | v7 | 2,010/2,010 | 0.517413 | 0.288200 |
| Novel | adaptive EECE v8 | 2,010/2,010 | 0.520896 | 0.296935 |

Medical changed by `-0.001455` recall and `-0.002470` raw context precision. Novel changed by `+0.003483` recall and `+0.008735` raw context precision. This is a mixed result: no Medical gain, but a small out-of-domain Novel gain and no sign of Medical-specific fitting.

## Medical answer and judge results

All 200 fixed evaluation queries completed in ten answer shards and ten judge shards without errors.

| Configuration | Correctness | Strict faithfulness | Claim F1 | Citation F1 | Average input tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| v7 | 0.880350 | 0.931950 | 0.852906 | 0.917894 | 14,119.60 |
| adaptive EECE v8 | 0.874168 | 0.918871 | 0.848665 | 0.933871 | 14,117.78 |
| Delta | -0.006182 | -0.013079 | -0.004241 | +0.015977 | -1.82 |

Citation F1 improved, but correctness, faithfulness, and Claim F1 did not beat v7.

## Current-head v9 replay

The current PR head was replayed in a new immutable run path; v3, v6, v7, and v8 artifacts were not changed. The first fail-closed pass rejected one Medical and one Novel Resource after exhausting the five cached v8 attempts. The evaluation retry allowance was therefore fixed at eight for both datasets, with no dataset-specific branch, prompt, threshold, or retrieval change. Both datasets then completed without partial Resource output.

| Dataset | Chunks | Resources | Windows | Adaptive entities | Explicit Facts | Withheld Hypotheses | Final entities | Final Facts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Medical | 1,385 | 1 | 277 | 2,225 | 1,993 | 159 | 4,662 | 11,528 |
| Novel | 3,503 | 11 | 705 | 6,122 | 4,922 | 560 | 9,093 | 8,946 |

The cache-backed logical build represents 1,382 local model calls / 18.30M input tokens for Medical and 3,274 calls / 43.75M input tokens for Novel. This includes reused v8 extraction completions plus the new verifier and repair completions. Provider API cost remained `$0`.

### v9 retrieval

| Dataset | Configuration | Completed | Recall@10 | Raw context precision |
| --- | --- | ---: | ---: | ---: |
| Medical | v7 | 2,062/2,062 | 0.890398 | 0.575969 |
| Medical | adaptive EECE v8 | 2,062/2,062 | 0.888943 | 0.573499 |
| Medical | adaptive EECE v9 | 2,062/2,062 | 0.887488 | 0.573516 |
| Novel | v7 | 2,010/2,010 | 0.517413 | 0.288200 |
| Novel | adaptive EECE v8 | 2,010/2,010 | 0.520896 | 0.296935 |
| Novel | adaptive EECE v9 | 2,010/2,010 | 0.518408 | 0.297482 |

Against v7, v9 changed Medical recall by `-0.002910` and raw context precision by `-0.002453`. On the independent Novel guardrail it changed recall by `+0.000995` and raw context precision by `+0.009282`. This again shows no Medical-specific fitting benefit and a small precision generalization gain on Novel.

### v9 Medical answer and judge

All 200 fixed Medical queries completed in ten answer shards and ten judge shards without errors.

| Configuration | Correctness | Strict faithfulness | Claim F1 | Citation F1 | Average input tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| v7 | 0.880350 | 0.931950 | 0.852906 | 0.917894 | 14,119.60 |
| adaptive EECE v8 | 0.874168 | 0.918871 | 0.848665 | 0.933871 | 14,117.78 |
| adaptive EECE v9 | 0.867484 | 0.906579 | 0.855637 | 0.921618 | 14,118.45 |
| v9 delta vs v7 | -0.012866 | -0.025371 | +0.002731 | +0.003724 | -1.15 |

The stricter current-head graph improved Claim F1 and citation F1 over v7, but did not improve correctness or strict faithfulness. Compared with v8, Claim F1 increased while correctness, faithfulness, and citation F1 decreased.

### v9 framework comparison

| Framework | Recall@10 | Raw context precision | Correctness | Strict faithfulness | Claim F1 | Citation F1 | Average input tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive EECE v9 | 0.887488 | 0.573516 | 0.867484 | 0.906579 | 0.855637 | 0.921618 | 14,118.45 |
| vector RAG + reranker | 0.706596 | 0.381232 | 0.873801 | 0.895057 | 0.807975 | 0.899981 | 14,242.00 |
| Microsoft GraphRAG | 0.830262 | 0.997090* | 0.781650 | 0.873956 | 0.733552 | 0.851768 | 21,701.20 |
| LightRAG | 0.932590 | 0.999030* | 0.893900 | 0.941683 | 0.857511 | 0.947662 | 28,955.61 |

`*` Microsoft GraphRAG and LightRAG package a large bundled context as one evidence item, so their near-1.0 context precision is not measured in the same raw evidence units as Kontext and vector RAG.

V9 still beats Microsoft GraphRAG on recall and all answer-quality metrics and beats vector RAG on retrieval, faithfulness, Claim F1, and citation F1. It does not beat LightRAG on Medical quality. Its average answer/judge input is about half LightRAG's, but that efficiency is not a quality win.

## Medical framework comparison

| Framework | Recall@10 | Raw context precision | Correctness | Strict faithfulness | Claim F1 | Citation F1 | Average input tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive EECE v8 | 0.888943 | 0.573499 | 0.874168 | 0.918871 | 0.848665 | 0.933871 | 14,117.78 |
| vector RAG + reranker | 0.706596 | 0.381232 | 0.873801 | 0.895057 | 0.807975 | 0.899981 | 14,242.00 |
| Microsoft GraphRAG | 0.830262 | 0.997090* | 0.781650 | 0.873956 | 0.733552 | 0.851768 | 21,701.20 |
| LightRAG | 0.932590 | 0.999030* | 0.893900 | 0.941683 | 0.857511 | 0.947662 | 28,955.61 |

`*` Microsoft GraphRAG and LightRAG package a large bundled context as one evidence item, so their near-1.0 context precision is not measured in the same raw evidence units as Kontext and vector RAG.

Adaptive EECE v8 beats vector RAG and Microsoft GraphRAG on most quality metrics while using fewer tokens than Microsoft GraphRAG. It does not beat LightRAG on Medical quality. It uses about half LightRAG's answer/judge input tokens, but that efficiency does not erase the quality gap.

## Cost and latency notes

- Medical embeddings: 318,078 input tokens, approximately `$0.00636`.
- Novel embeddings: 816,778 input tokens, approximately `$0.01634`.
- Total OpenAI embedding cost: approximately `$0.02270`.
- V9 reused the identical `text-embedding-3-small` vectors, so incremental embedding API spend was `$0`; the logical token and cost totals remain the same `$0.02270` for comparison.
- KG extraction, rerank, answer, and judge used local Codex CLI and incurred no provider API charge in this run.
- Medical retrieval ran concurrently with Novel KG construction, and Medical answer/judge overlapped Novel retrieval. The recorded v8 latency values are therefore contention-affected and must not be used as an apples-to-apples latency claim against v7 or other frameworks.
- V9 ran KG construction before sequential Medical and Novel retrieval. Its Medical retrieval p95 was 17.85s and end-to-end p95 was 101.17s, but cross-framework latency still depends on local CLI scheduling and is not treated as a quality winner claim.

## Interpretation

The adaptive graph added many validated Facts, but v7 was already dominated by strong vector/BM25 candidates, source hydration, and an LLM reranker. Extra graph edges widened or duplicated graph paths without a source-only calibration step, producing no Medical retrieval benefit in either replay. The denser graph also introduces high-degree concept and literal hubs that can displace useful candidates before reranking.

The next general improvement should not add more extraction volume. It should select graph capabilities and edges by source-only structural signals, cap hub expansion, and treat generic `related_to` and literal-valued edges more conservatively. Any such change must be fixed before looking at benchmark answers and must be checked on both Medical and Novel.

## Decision

Keep PR #3's production enrichment seam, identity continuity, and validation fixes. Do not promote adaptive EECE v8 or v9 as the new benchmark winner. Keep v7 as the Medical quality baseline until a source-only graph calibration policy beats it on Medical without losing the Novel guardrail.

The v9 replay closes the current-head scoring gap: the safety-hardened implementation is valid and avoids a Novel regression relative to v7, but it does not improve the primary Medical result. Both v8 and v9 artifacts remain immutable evidence for their evaluated commits.
