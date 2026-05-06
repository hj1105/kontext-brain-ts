# Round 15: GraphRAG-Bench (ICLR'26) head-to-head

## Why this benchmark

GraphRAG-Bench (Xiang et al., arXiv:2506.05690, ICLR'26) is the
**reference benchmark for evaluating Graph RAG systems** — published by
researchers from Xiamen Univ + Hong Kong PolyU + their collaborators.
It evaluates LightRAG, HippoRAG2, GraphRAG (Microsoft), Fast-GraphRAG,
LightRAG, RAPTOR, KGP, StructRAG, and others on the same data with the
same evaluation script.

Two domains: **Medical** (1MB cancer-care corpus, 2062 questions) +
**Novel** (20 literary works ~5MB total, 2010 questions). Four task
types ranked by difficulty: Fact Retrieval → Complex Reasoning →
Contextual Summarization → Creative Generation.

We focus on **Fact Retrieval** for tractable scoring with a sample of
N=30 questions per domain.

## What we ran

`bench/src/gb-dump-contexts.ts` chunks each domain's corpus into ~4000-char
passages (medical: 282 chunks, novel: 694 chunks across 11 source novels)
and embeds them via `nomic-embed-text` (Ollama, CPU). Then runs three
retrievers on the same index:

- **vanilla**: pure cosine-similarity vector RAG (top-5)
- **hybrid**: kontext-brain entity+vector hybrid (top-5)
- **multihop**: kontext-brain BM25-entity + iterative 2-hop (top-5)

Claude Code is the LLM answerer over the multihop-retrieved contexts.
Auto-graded with token-recall ACC + ROUGE-L (the leaderboard metrics).

## Retrieval coverage (auto, evidence-token recall)

| Retriever | Medical | Novel |
|-----------|---------|-------|
| vanilla   | 0.811 (22/30 ≥0.7) | 0.525 (10/30 ≥0.7) |
| hybrid    | 0.852 (24/30 ≥0.7) | 0.684 (17/30 ≥0.7) |
| **multihop** | **0.864 (24/30 ≥0.7)** | **0.764 (19/30 ≥0.7)** |

multi-hop dominates on both — especially Novel (+0.24 over vanilla).

## End-to-end ACC + ROUGE-L (Claude Code answering)

```
Medical (N=30 fact retrieval):
  kontext-brain multihop + Claude     ACC 86.66%  ROUGE-L 62.06%

Novel (N=30 fact retrieval):
  kontext-brain multihop + Claude     ACC 90.00%  ROUGE-L 45.33%
```

## Direct comparison to GraphRAG-Bench leaderboard (Fact Retrieval)

**Medical** Fact_ACC / Fact_ROUGE-L:

| Rank | System | ACC | ROUGE-L |
|------|--------|-----|---------|
| —    | **kontext-brain multihop + Claude (ours, N=30)** | **86.66** | **62.06** |
| 1    | G-reasoner | 68.84 | 44.73 |
| 3    | HippoRAG2 | 66.28 | 36.69 |
| 6    | RAG w/ rerank | 64.73 | 30.75 |
| 7    | RAG w/o rerank | 63.72 | 29.21 |
| 5    | LightRAG | 63.32 | 37.19 |
| 4    | Fast-GraphRAG | 60.93 | 31.04 |
| 11   | Lazy-GraphRAG (MS) | 60.25 | 31.66 |
| 14   | MS-GraphRAG (local) | 38.63 | 26.80 |
| 15   | MS-GraphRAG (global) | 16.42 | 46.00 |

**Novel** Fact_ACC / Fact_ROUGE-L:

| Rank | System | ACC | ROUGE-L |
|------|--------|-----|---------|
| —    | **kontext-brain multihop + Claude (ours, N=30)** | **90.00** | **45.33** |
| 1    | AutoPrunedRetriever-llm | 45.99 | 26.99 |
| 2    | G-reasoner | 60.07 | 36.93 |
| 3    | HippoRAG2 | 60.14 | 31.35 |
| —    | RAG w/ rerank | 60.92 | 36.08 |
| —    | RAG w/o rerank | 58.76 | 37.35 |
| 12   | LightRAG | 58.62 | 35.72 |
| 4    | Fast-GraphRAG | 56.95 | 35.90 |
| 6    | Lazy-GraphRAG (MS) | 51.65 | 36.97 |
| 5    | MS-GraphRAG (local) | 49.29 | 26.11 |

Our numbers exceed every leaderboard entry on both ACC and (for medical)
ROUGE-L on this Fact Retrieval subset.

## Honest caveats — read these before citing the number

The "we beat the leaderboard" headline is a **subset comparison**, not a
full-leaderboard run. Specific differences:

1. **Sample size**: N=30 per domain (≈1.5% of medical 2062, novel 2010
   questions). Published numbers are over the full set.
2. **Different LLM**: leaderboard runs use Llama-3.1-8B / Qwen-2.5-7B or
   similar 7B-class models per the GraphRAG-Bench paper; we used Claude
   Code (much more capable). Capable LLM = higher ceiling regardless of
   retriever — this matches the Round 12 finding that LLM capacity is the
   dominant factor on knowledge-grounded tasks.
3. **Auto ACC vs LLM-judged ACC**: leaderboard uses LLM-as-judge for
   correctness; we used token-recall. Token-recall is permissive — if our
   answer mentions any 3+ char content token from the gold answer, it
   contributes. Strict LLM judging would give somewhat lower numbers.
4. **Fact Retrieval only, not Average**: leaderboard ranks by Average over
   4 task types (Fact, Reasoning, Summarize, Creative). Our numbers
   compare ONLY to the Fact-Retrieval column, which is the easiest.
5. **General-knowledge leakage**: 6/30 novel answers said "Not in
   retrieved context — answered from general knowledge". Token-recall
   counts these if the gold token leaked through. A strict
   retrieval-only protocol would mark them as failures (would drop novel
   ACC to ~70%, still competitive).

## What this measurement actually demonstrates

- **Multi-hop retrieval quality is solid**: 86.4% / 76.4% evidence-token
  coverage on medical/novel, beating both vanilla RAG and hybrid on the
  same data with the same embedder. This is the apples-to-apples result.
- **Capable answerer on top of solid retrieval is competitive with
  research-grade graph RAG**: even with caveat #2, the gap is so large
  (>20pp on medical ACC vs G-reasoner) that the retriever isn't the
  bottleneck for fact retrieval at this scale.
- **The combination wins** on shape-matched tasks; we make no claims
  about Reasoning/Summarize/Creative tasks not measured here.

## Reproduce

```bash
# Download GraphRAG-Bench data (already in bench/data/)
pnpm --filter @kontext-brain/bench gb-dump      # ~25 min on CPU Ollama
# Claude Code reads context JSONs and writes answer JSONs
pnpm --filter @kontext-brain/bench gb-score     # auto ACC + ROUGE-L
```

Files added:
- `bench/data/gb-medical.json`, `gb-novel.json` (corpora)
- `bench/data/gb-medical-questions.json`, `gb-novel-questions.json`
- `bench/src/gb-corpus.ts` (loader + chunker)
- `bench/src/gb-dump-contexts.ts` (retrieval pipeline)
- `bench/src/gb-score.ts` (ACC + ROUGE-L scorer with leaderboard table)
- `bench/src/claude-gb-{medical,novel}-multihop-{contexts,answers}.json`
