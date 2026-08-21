# Adaptive EECE v8 evaluation — 2026-08-21

## Scope and guardrails

- Implementation: `996b8bb` on `codex/adaptive-knowledge-graph` (PR #3).
- Retrieval stack: v7 source hydration, declared candidate `k=50`, recall-safe local GPT rerank, plus the adaptive Entity–Event–Claim–Evidence graph.
- Datasets: GraphRAG-Bench Medical and Novel. FRAMES was out of scope.
- The builder read source chunks and the pre-existing KG only. It did not read questions, reference answers, gold evidence, or dataset names when selecting capabilities or extracting knowledge.
- Capability selection remained source-driven and domain-independent.
- Invalid citations and references never entered the graph. The evaluation-only policy retried five times and then withheld the whole invalid window. The product default remains three attempts and `throw`.
- LLM work used the local Codex CLI. OpenAI API usage was limited to `text-embedding-3-small`.

## Fixed extraction policy

| Setting | Value |
| --- | ---: |
| Chunks per window | 6 |
| Overlap | 1 |
| Maximum window characters | 12,000 |
| Product default attempts | 3 |
| Evaluation attempts | 5 |
| Evaluation validation policy | `empty-window` |
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
- KG extraction, rerank, answer, and judge used local Codex CLI and incurred no provider API charge in this run.
- Medical retrieval ran concurrently with Novel KG construction, and Medical answer/judge overlapped Novel retrieval. The recorded v8 latency values are therefore contention-affected and must not be used as an apples-to-apples latency claim against v7 or other frameworks.

## Interpretation

The adaptive graph added many validated Facts, but v7 was already dominated by strong vector/BM25 candidates, source hydration, and an LLM reranker. Extra graph edges widened or duplicated graph paths without a source-only calibration step, producing almost no Medical retrieval benefit. The denser graph also introduces high-degree concept and literal hubs that can displace useful candidates before reranking.

The next general improvement should not add more extraction volume. It should select graph capabilities and edges by source-only structural signals, cap hub expansion, and treat generic `related_to` and literal-valued edges more conservatively. Any such change must be fixed before looking at benchmark answers and must be checked on both Medical and Novel.

## Decision

Keep PR #3's production enrichment seam and validation fixes. Do not promote adaptive EECE v8 as the new benchmark winner. Keep v7 as the Medical quality baseline until a source-only graph calibration policy beats it on Medical without losing the Novel guardrail.
