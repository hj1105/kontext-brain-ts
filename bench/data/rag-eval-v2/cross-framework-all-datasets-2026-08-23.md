# Cross-framework RAG evaluation (2026-08-23)

## Decision

The no-configuration evaluation default remains **Kontext v13 anchored evidence
answer stack**. The v15 corpus-completeness correction is validated as a
candidate for artifact-backed corpora whose precomputed KG omitted original
resources: it fixes the identified Novel coverage failure while preserving the
same dataset-independent ranking and answer policies. It is not silently
promoted in this release because Novel was the registered development signal
and SciFact recall moved from 0.9600 to 0.9533. No per-dataset branch or
score-driven parameter change was introduced.

The strongest result is not a precision/recall trade: on Medical, v15 keeps
recall within 0.001 of v13 while improving correctness, faithfulness, claim F1,
and citation F1. On Novel, it raises both recall and correctness substantially.

## Frozen comparison boundary

- Every framework receives the same raw corpus and questions and builds its
  native automatic index.
- Embeddings use OpenAI `text-embedding-3-small`, 1,536 dimensions. Answer and
  judge calls use the local Codex CLI with the frozen models and do not use the
  supplied OpenAI API key.
- Retrieval is scored on every query. Medical and Novel answer metrics use the
  same deterministic 200-query sample per dataset.
- The final adapter chooses an optional precomputed KG by Resource identity and
  normalized source-text coverage. Dataset names, reference answers, gold
  evidence, categories, and judge outputs are unavailable to query expansion,
  graph traversal, reranking, and answer-policy decisions.
- SciFact and NFCorpus are BEIR retrieval-only datasets; they have qrels but no
  generation reference answer compatible with the shared answer judge.

## GraphRAG-Bench Medical

All rows completed 2,062/2,062 retrievals. Answer-bearing rows completed the
same 200/200 answer and judgement sample with zero errors.

| System | Recall@10 | nDCG@10 | Raw/package-sensitive CP | Correctness | Strict faith | Claim F1 | Citation F1 | Warm retrieval p95 | Query-to-answer p95 | Eval E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Kontext v15** | **0.891368** | 0.968907 | **0.580861** | **0.949917** | **0.961385** | **0.861159** | **0.958348** | 7.77 s | 20.15 s | 97.65 s |
| Kontext v13 | 0.892338 | 0.970428 | 0.580222 | 0.946127 | 0.953411 | 0.855016 | 0.954141 | 6.83 s | 18.60 s | 108.08 s† |
| LightRAG 1.5.6 | 0.932590 | 0.999030* | 0.999030* | 0.893900 | 0.941683 | 0.857511 | 0.947662 | 6.00 s | 16.22 s | 123.02 s |
| Microsoft GraphRAG 3.1.1 | 0.830262 | 0.997090* | 0.997090* | 0.781650 | 0.873956 | 0.733552 | 0.851768 | 0.28 s | 12.83 s | 123.08 s† |
| Vector + BM25-RRF | 0.706596 | 0.743378 | 0.381232 | 0.873801 | 0.895057 | 0.807975 | 0.899981 | not re-measured | not re-measured | not re-measured |

`*` LightRAG and Microsoft GraphRAG package a large native context as one
evidence record. Their near-one nDCG/context precision values are
package-sensitive and are not comparable with Kontext's separately scored
evidence windows.

Relative to LightRAG, v15 has -0.041222 recall but +0.056017 correctness,
+0.019702 strict faithfulness, +0.003649 claim F1, and +0.010686 citation F1.
It is 1.3x slower on warm retrieval p95 and 1.2x slower on query-to-answer p95,
and 1.3x faster on the judge-inclusive Eval E2E p95. Relative to Microsoft
GraphRAG, v15 improves recall by 0.061106 and correctness by 0.168267 while
being 27.8x slower on warm retrieval p95 and 1.6x slower on query-to-answer p95.

An earlier revision of this report claimed v15's retrieval p95 was 11.1x lower
than LightRAG's. That comparison came from queue-contended timings and its
direction was wrong; under the clean protocol the two systems sit within 1.3x of
each other. Kontext does not win on speed. Both make exactly one LLM round trip
during retrieval — Kontext to rerank evidence, LightRAG to extract query
keywords — and Kontext's is the larger prompt, which accounts for the remaining
gap. Microsoft GraphRAG makes no retrieval LLM call at all, which is why it is
an order of magnitude faster and also why it trails both on correctness. The
accurate reading is that retrieval latency here tracks the number and size of
LLM round trips, and that Kontext spends one to buy grounding accuracy.

HippoRAG 2 is not assigned a score: its pinned native index path has no
long-document chunker and the Medical source contains inputs beyond the
embedding model's 8,192-token limit. The harness reports this as unsupported
rather than silently adding a custom chunker.

## GraphRAG-Bench Novel

All rows completed 2,010/2,010 retrievals and the same 200/200 answer and
judgement sample with zero errors.

| System | Recall@10 | nDCG@10 | Raw/package-sensitive CP | Correctness | Strict faith | Claim F1 | Citation F1 | Warm retrieval p95 | Query-to-answer p95 | Eval E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Kontext v15** | **0.820896** | 0.934913 | **0.449236** | **0.856572** | **0.929011** | **0.823365** | 0.936913 | 9.26 s | 20.15 s | 103.18 s |
| Kontext v13 | 0.525871 | 0.666185 | 0.289687 | 0.465432 | 0.792182 | 0.518112 | 0.552107 | 10.13 s | 18.87 s | 83.62 s† |
| LightRAG 1.5.6 | 0.856716 | 0.994527* | 0.994527* | 0.849765 | 0.927176 | 0.820114 | **0.940739** | 6.66 s | 17.88 s | 117.30 s |
| Microsoft GraphRAG 3.1.1 | 0.771642 | 0.981592* | 0.981592* | 0.766762 | 0.865095 | 0.743363 | 0.876322 | 0.40 s | 10.97 s | 140.49 s† |

v15 versus v13 improves recall by 0.295025 and correctness by 0.391141. It
also edges LightRAG on correctness (+0.006807), strict faithfulness
(+0.001835), and claim F1 (+0.003251), while trailing recall by 0.035821 and
citation F1 by 0.003826. It exceeds Microsoft GraphRAG on every reported
quality metric except the incomparable packaged context precision.

`†` marks a judge stage that did not complete all 200 queries, which
disqualifies only that row's Eval E2E figure. Retrieval and query-to-answer
latency are unaffected, because the judge never enters them.

The three latency columns come from the 2026-08-24/27 clean latency campaign
(protocol `clean-latency-v1.1`, artifacts under
`runs/clean-latency-2026-08-24/`), which replaced the earlier queue-contended
timings rather than adjusting them. Warm retrieval p95 excludes index
construction; query-to-answer p95 adds the answer call and is the user-facing
figure; Eval E2E p95 adds the judge and describes the evaluation pipeline only.
Each row reuses its finished warm index with no build and no new embeddings,
draws the same deterministic 200-query sample, runs alone on the host, and holds
retrieval, answer, and judge at concurrency 1 with batch size 1 and no retries.
A row is accepted only when retrieval and answer show no latency over 600 s, no
queue or usage-limit error, and no inter-completion gap in the 10–20 minute
throttle-wave band; rejected runs are preserved under `invalid-*`, `aborted-*`,
and `pre-expansion-latency-fix/`. The Vector + BM25-RRF baseline was outside the
campaign's eight-row matrix and is reported as not re-measured instead of
carrying its contaminated figure forward. Quality metrics are unchanged.

### Answer metric components

`Completeness` equals the aggregate claim recall in the preserved judge. The
archived score files expose citation F1; their immutable judgement records also
retain the separate citation precision/recall and per-claim support decisions
needed by the newer score contract.

| Dataset | System | Correctness 95% CI | Correct-claim precision | Claim recall | Claim support precision | Citation P / R |
|---|---|---:|---:|---:|---:|---:|
| Medical | Kontext v15 | [0.9222, 0.9746] | 0.983163 | 0.808113 | **0.961519** | **0.963485 / 0.965000** |
| Medical | Kontext v13 | [0.9179, 0.9718] | 0.981152 | 0.811252 | 0.951253 | 0.946661 / 0.964650 |
| Medical | LightRAG | [0.8559, 0.9294] | 0.965388 | **0.836200** | 0.940405 | 0.937669 / **0.971850** |
| Medical | Microsoft GraphRAG | [0.7314, 0.8306] | 0.944506 | 0.699735 | 0.845174 | 0.863309 / 0.872681 |
| Medical | Vector + BM25-RRF | [0.8301, 0.9118] | 0.956950 | 0.777640 | 0.866304 | 0.914637 / 0.908593 |
| Novel | Kontext v15 | [0.8129, 0.8999] | 0.928282 | 0.811234 | 0.925203 | 0.929161 / 0.952285 |
| Novel | Kontext v13 | [0.3972, 0.5322] | 0.865020 | 0.443700 | 0.699927 | 0.707732 / 0.559600 |
| Novel | LightRAG | [0.8046, 0.8921] | 0.915942 | **0.828250** | **0.925980** | **0.932701 / 0.958050** |
| Novel | Microsoft GraphRAG | [0.7122, 0.8203] | 0.908052 | 0.725950 | 0.854135 | 0.889195 / 0.889688 |

## Public BEIR retrieval guardrails

| Dataset | System | Completed | Recall@10 | nDCG@10 | Raw/package-sensitive CP | Retrieval p95 |
|---|---|---:|---:|---:|---:|---:|
| SciFact | **Kontext v15** | 300/300 | **0.953333** | **0.824729** | 0.034480 | 37.53 s |
| SciFact | Kontext v13 | 300/300 | 0.960000 | 0.824401 | 0.034686 | 38.71 s |
| SciFact | Vector + BM25-RRF | 300/300 | 0.822444 | 0.706630 | 0.163667 | 0.017 s |
| SciFact | LightRAG 1.5.6 | 300/300 | 0.149667 | 0.166667* | 0.166667* | 714.31 s |
| SciFact | Microsoft GraphRAG 3.1.1 | indexing | — | — | — | — |
| NFCorpus | **Kontext v15** | 323/323 | **0.279016** | 0.366554 | 0.174420 | 57.88 s |
| NFCorpus | Kontext v13 | 323/323 | 0.278486 | 0.368195 | 0.174640 | 57.26 s |
| NFCorpus | Vector + BM25-RRF | 323/323 | 0.162923 | 0.331071 | 0.308978 | 0.015 s |
| NFCorpus | LightRAG 1.5.6 | 323/323 | 0.004903 | 0.080495* | 0.080495* | 414.36 s |
| NFCorpus | Microsoft GraphRAG 3.1.1 | 323/323 | 0.146408 | 0.690402* | 0.690402* | 0.425 s |

LightRAG's official-corpus IDs are recovered through the adapter's persisted
`sourceIds` provenance contract. The raw pre-contract SciFact output is kept as
an audit artifact; it is not a second trial. The first Microsoft NFCorpus
retrieval process started 13 minutes before the same provenance contract was
committed, so its source-less output is likewise preserved as
`retrieval.pre-provenance-2026-08-24.jsonl`. The completed native index was
reused without rebuilding; a fresh retrieval process produced 323/323 rows
with 4,644 official source IDs before the score above was generated. The
Microsoft SciFact row will be filled only after its native graph index and
retrieval complete.

## Answer/judge tokens and API-equivalent cost

These are recorded local-Codex token counts for the shared answer and judge
stages. The final column applies the 2026-08-24 official uncached API list
prices to each stage: GPT-5.6 Terra answer tokens at $2 input/$12 output and
GPT-5.6 Sol judge tokens at $4 input/$20 output per million tokens. It is an
API-equivalent estimate, not an API invoice or a per-query Codex subscription
charge. Cache-token telemetry was unavailable, so all input is conservatively
priced as uncached.

| Dataset | System | Input tokens / 200 queries | Output tokens / 200 queries | Input/query | Output/query | API-equivalent $/query |
|---|---|---:|---:|---:|---:|---:|
| Medical | Kontext v15 | 9,029,975 | 255,371 | 45,150 | 1,277 | $0.1593 |
| Medical | Kontext v13 | 9,029,688 | 268,838 | 45,148 | 1,344 | $0.1606 |
| Medical | LightRAG | 11,582,245 | 330,228 | 57,911 | 1,651 | $0.2057 |
| Medical | Microsoft GraphRAG | 8,680,479 | 388,382 | 43,402 | 1,942 | $0.1678 |
| Novel | Kontext v15 | 9,538,002 | 274,849 | 47,690 | 1,374 | $0.1688 |
| Novel | Kontext v13 | 9,501,680 | 217,529 | 47,508 | 1,088 | $0.1629 |
| Novel | LightRAG | 11,741,076 | 270,193 | 58,705 | 1,351 | $0.2022 |
| Novel | Microsoft GraphRAG | 8,698,488 | 330,527 | 43,492 | 1,653 | $0.1624 |

## Metric-contract coverage

The project now treats the following as the official layered scorecard:
Evidence Recall@K, nDCG@K, raw Context Precision, Claim Recall, Strict
Faithfulness / Claim Support Precision, Citation Precision / Recall / F1,
Answerable/Unanswerable Joint Accuracy, Robustness Drop, and Clarity /
Conciseness / Fluency. There is no aggregate overall score.

This comparison predates judge-policy-v3. The nDCG, claim support, and separate
citation precision/recall values above were derived read-only from its preserved
retrieval and judgement records without another model call. Joint accuracy is unavailable because these four
scored datasets do not contain a reviewed mixed answerable/unanswerable track;
Robustness Drop is unavailable because no paired perturbation set was run; and
writing-quality scores require the new judge contract. Those cells remain
`null` rather than being inferred or backfilled by a different judge. Existing
v7/v12/v13/v14/v15 score artifacts are not overwritten.

## Embedding cost

The manifest price is $0.02 per million input tokens. “v15 newly incurred” is
the marginal cost of the v15 run, not the cost of constructing a fresh index
without its validated v13 source cache.

| Dataset/system | Embedding input tokens represented | Auditable cost | Note |
|---|---:|---:|---|
| Medical / Kontext v13 base | 407,518 | $0.008150 | Original reusable index |
| Medical / v15 newly incurred | **0** | **$0** | 9,340 vectors reused; 0 new |
| Medical / LightRAG | 770,450 | $0.015409 | Native-index usage log |
| Medical / Microsoft GraphRAG | 681,136 | $0.013623 | Native-index usage log |
| Novel / Kontext v13 base | 891,981 | $0.017840 | Original reusable index |
| Novel / v15 actual run | 1,470,715 | $0.029414 | Pre-fix whole-batch invalidation; preserved, not rerun |
| Novel / LightRAG | 2,468,525 | $0.049371 | Native-index usage log |
| Novel / Microsoft GraphRAG | 4,416,284 | $0.088326 | Native-index usage log |
| SciFact / Kontext v13 base | 1,680,151 | $0.033603 | Original reusable index |
| SciFact / v15 newly incurred | **0** | **$0** | 6,364 vectors reused; 0 new |
| SciFact / Vector baseline | 1,693,048 | $0.033861 | Native vector index |
| SciFact / LightRAG | 815,880 | $0.016318 | Native-index usage log |
| NFCorpus / Kontext v13 base | 1,253,777 | $0.025076 | Original reusable index |
| NFCorpus / v15 newly incurred | **0** | **$0** | 4,645 vectors reused; 0 new |
| NFCorpus / Vector baseline | 1,283,605 | $0.025672 | Native vector index |
| NFCorpus / LightRAG | 261,418 | $0.005228 | Native-index usage log |
| NFCorpus / Microsoft GraphRAG | 5,509,618 | $0.110192 | Native KG index plus two query retrieval passes; provenance replay added only 2,781 tokens/$0.0000556 and did not rebuild the index |

The embedding table above reports the separately auditable OpenAI embedding API
charge. The answer/judge table reports the stage-aware API-equivalent LLM cost;
Kontext's local-CLI query-expansion and reranker calls were not fully metered and
are excluded rather than estimated.

The original v15 Novel run revealed a cache transport defect: adding nine
missing resources changed a whole-corpus digest and invalidated otherwise
unchanged batches. The fix uses content-addressed read-through validation over
task, model, dimensions, ID, title, and text. Medical, SciFact, and NFCorpus
then proved zero newly incurred embedding tokens. Rebuilding a different
framework's native index is still required because its chunks, entities,
summaries, and vector-store format are not interchangeable with Kontext's.

## v13 retrieval and answer logic

1. Keep the original question and generate at most three question-only
   retrieval perspectives.
2. Fuse vector and BM25 perspective rankings with weighted RRF (original
   question 2, each expansion 1, RRF k=10).
3. Fuse vector, graph, BM25, and context candidates into a 50-candidate pool and
   apply the coverage-aware local-LLM reranker.
4. Hydrate selected sources into 5,000-character windows under a 50,000-character
   total context budget.
5. Derive supported evidence needs from only the question and retrieved
   evidence; emit at most one atomic claim and one best citation per supported
   need, omit unsupported needs, deduplicate, and cap at eight claims.

v15 changes only corpus completeness: if an artifact-backed corpus resource is
not represented by exact source identity or a normalized text probe, it adds
canonical 5,000-character/400-character-overlap fallback windows before the
unchanged v13 retrieval policy runs.

## Dataset coverage and exclusions

- **Scored here:** GraphRAG-Bench Medical, GraphRAG-Bench Novel, BEIR SciFact,
  and BEIR NFCorpus.
- **FRAMES:** excluded from this run by the operator. Its provided relevant
  Wikipedia URLs cannot form a fair retrieval corpus without a fixed background
  Wikipedia snapshot.
- **GaRAGe:** blocked because the paper's complete grounding corpus is not
  publicly redistributable; no similarly named substitute is used.
- **UAEval-style and Stable-RAG:** remain blocked until reviewed, versioned
  corpus/query assets are committed. Synthetic results are not fabricated.
- **CRAG:** kept in the dynamic query-scoped API track, not mixed with the
  static-KB table.
- **TREC RAG / RAGTIME:** require a separately provisioned large-corpus or live
  multilingual workflow.
- **HotpotQA fullwiki, 2WikiMultiHopQA, MuSiQue, and KILT/NQ:** researched as
  future gates; their full common corpora were not provisioned in this run, so
  they are listed as planned rather than assigned scores.

## Reproducibility and limitations

- The exact policies and trial limits were registered before execution in
  `bench/src/rag-eval-v2/TUNING_LOG.md`.
- The scored Medical/Novel runs originally used the dataset ID only as a file
  locator for their static KG artifact. A post-run architecture audit replaced
  that locator with corpus-evidence selection. A read-only full-data replay
  selected the same `gb-medical` and `gb-novel` chunk/graph pairs, so ranking
  inputs and scores are unchanged; equal-coverage ambiguity now fails closed.
- v15 paths are new and do not overwrite v3, v6, v7, v12, v13, or v14.
- Medical v15 contains 2,062 unique successful retrievals, 200 unique successful
  answers, and 200 unique successful judgements. Its sample digest
  `605f1e0f271019a881b2411664f3237526a1503591a410169c593c4d85d44d7d`
  exactly matches v13.
- Answer/judge models are stochastic. Confidence intervals are reported for
  correctness; small paired differences should not be overstated.
- Warm retrieval p95 excludes index construction. Index wall-clock time is not
  represented as a comparable score because framework-native builds expose
  different stage boundaries. The frozen conditions, per-stage validity, and
  environment provenance for every latency figure are recorded per row in
  `runs/clean-latency-2026-08-24/*/clean-latency-report.json`.
- The current Novel v15 result is a development-set repair signal, not a clean
  held-out estimate. Medical and BEIR are the unchanged regression and public
  generalization gates.

## Primary artifact directories

- `openai-small-kontext-v15-corpus-complete-medical-2026-08-23`
- `openai-small-kontext-v15-corpus-complete-novel-2026-08-23`
- `openai-small-kontext-v15-corpus-complete-scifact-2026-08-23`
- `openai-small-kontext-v15-corpus-complete-nfcorpus-2026-08-23`
- `openai-small-cross-framework-lightrag-{novel,scifact,nfcorpus}-2026-08-23`
- `openai-small-cross-framework-microsoft-graphrag-{novel,scifact,nfcorpus}-2026-08-23`
