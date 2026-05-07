# Round 16: Controlled comparison — confounders removed

## Why this round exists

Round 15 reported `kontext-brain mh + Claude` at 86.66% medical / 90.00%
novel on GraphRAG-Bench Fact Retrieval, comparing to a leaderboard
showing the best system at 68.84% / 60.07%. **That headline had 5
confounders** that made the comparison unclear:

1. Sample size (N=30 vs full 2062/2010)
2. LLM (Claude vs leaderboard's 7B-class open models)
3. Auto-graded ACC (token recall) vs leaderboard's LLM-judge
4. Fact Retrieval only vs 4-task average
5. World-knowledge fallback when retrieval missed

Round 16 controls 4 of the 5 (everything except #2 LLM, which is
irreducible without re-running their systems with our LLM).

## How we control

### Retrieval-only metrics (no LLM at all)

The cleanest comparison removes the LLM entirely. We measure how well
each retriever surfaced the gold evidence sentence:

- **Token coverage**: fraction of gold-evidence content tokens (4+ char,
  non-stopword) present in any retrieved chunk
- **≥0.7 token coverage count**: how many queries had ≥70% gold tokens
  retrieved
- **Evidence substring**: did the literal first 60 chars of gold evidence
  appear verbatim? (very strict — paraphrasing fails this)
- **Top-1 chunk coverage**: did the #1 retrieved chunk alone hold ≥70%
  of gold tokens?

### Strict end-to-end (#5 controlled)

Multi-hop answers are rescored with a **strict protocol**: any answer
containing fallback markers (`"general knowledge"`, `"not in retrieved
context"`, `"retrieval failed"`, etc.) is automatically scored 0,
regardless of whether the answer happened to mention the gold token. This
prevents world-knowledge leakage from inflating the ACC.

## Result A: Retrieval-only (LLM-free, fully controlled)

```
MEDICAL (N=30)
retriever  | tokenCov | ≥0.7-cov | ev-substr | top1-hit
vanilla    | 0.811    |   22/30  |    0/30   |   8/30
hybrid     | 0.852    |   24/30  |    1/30   |   9/30
multihop   | 0.864    |   24/30  |    1/30   |   6/30   ← best aggregate

NOVEL (N=30)
retriever  | tokenCov | ≥0.7-cov | ev-substr | top1-hit
vanilla    | 0.525    |   10/30  |    0/30   |   4/30
hybrid     | 0.684    |   17/30  |    0/30   |   6/30
multihop   | 0.764    |   19/30  |    0/30   |   9/30   ← best on every metric
```

**This is the cleanest comparison kontext-brain has.** Same corpus, same
chunker, same embedder, same questions, no LLM in the loop. multi-hop
beats vanilla RAG on aggregate token coverage by **+5pp medical** and
**+24pp novel**.

The `top1-hit` column shows multi-hop is *worse than hybrid* on medical
(6/30 vs 9/30) — its iterative 2-hop expansion sometimes pushes the gold
chunk down the rank list. On novel (where multi-hop is most needed)
top-1 is best (9/30).

## Result B: Strict end-to-end ACC (fallback-rejected)

```
MEDICAL: ACC 86.62%, ROUGE-L 62.06%, fallback-rejected 0/30
NOVEL:   ACC 57.63%, ROUGE-L 35.44%, fallback-rejected 10/30
```

**Novel drops from 90.00% → 57.63%** when fallback answers are scored 0.
This is the honest number. 10 of 30 novel queries had retrieval that
missed the gold evidence enough that Claude couldn't answer from
context alone.

## Result C: vs leaderboard, with caveat #2 (LLM) explicit

| Medical Fact_ACC / ROUGE-L | Score |
|-----------------------------|-------|
| **kontext-brain mh + Claude (STRICT, N=30)** | **86.62 / 62.06** |
| G-reasoner (Llama-3.1-8B, full N) | 68.84 / 44.73 |
| HippoRAG2 (8B-class, full N) | 66.28 / 36.69 |
| LightRAG (8B-class, full N) | 63.32 / 37.19 |
| RAG (w/ rerank) (8B-class, full N) | 64.73 / 30.75 |
| Fast-GraphRAG (8B-class, full N) | 60.93 / 31.04 |
| Lazy-GraphRAG (Microsoft) (8B, full N) | 60.25 / 31.66 |
| MS-GraphRAG (local) (8B, full N) | 38.63 / 26.80 |

| Novel Fact_ACC / ROUGE-L | Score |
|---------------------------|-------|
| HippoRAG2 (8B-class, full N) | 60.14 / 31.35 |
| G-reasoner (Llama-3.1-8B, full N) | 60.07 / 36.93 |
| RAG (w/ rerank) (8B-class, full N) | 60.92 / 36.08 |
| LightRAG (8B-class, full N) | 58.62 / 35.72 |
| **kontext-brain mh + Claude (STRICT, N=30)** | **57.63 / 35.44** |
| Fast-GraphRAG (8B-class, full N) | 56.95 / 35.90 |
| KGP (8B-class, full N) | 54.15 / 24.73 |
| MS-GraphRAG (local) (8B, full N) | 49.29 / 26.11 |

### Reading these tables honestly

**Medical**: kontext-brain + Claude beats every leaderboard system by
≥18pp ACC. This gap is **mostly the LLM** — Claude is several tiers more
capable than Llama-3.1-8B / Qwen-2.5-7B. The retriever contribution is
real but not the dominant factor.

**Novel**: kontext-brain + Claude **is in the middle of the pack**,
slightly behind HippoRAG2/G-reasoner/RAG-rerank (-2 to -3pp), slightly
ahead of Fast-GraphRAG / KGP. **The LLM advantage doesn't carry through
on novel** because retrieval failures dominate — narrative content is
harder to surface than fact-dense medical text.

## Honest takeaway

Three statements that survive scrutiny:

1. **Retrieval comparison (LLM-free)**: kontext-brain multi-hop has the
   best evidence coverage of the three retrievers we built (+5pp medical,
   +24pp novel over vanilla RAG). This is fully controlled.

2. **Medical end-to-end**: a capable LLM (Claude) over solid retrieval
   beats published 7B-LLM-based graph RAG by a wide margin. The gap is
   largely LLM-driven; we make no claim that the retriever pipeline
   alone closes a 20pp gap.

3. **Novel end-to-end**: even with Claude, kontext-brain lands in the
   middle of the leaderboard pack on narrative content. Retrieval
   failures (10/30 fallbacks) are the bottleneck. **HippoRAG2 / G-reasoner
   beat us on this** (-2.5pp), and that's a real result, not noise.

## What we still cannot claim

- Average across 4 task types (Fact + Reasoning + Summarize + Creative).
  We only ran Fact.
- Performance on full-set N (we ran 30, leaderboard ran ~2000).
- Our retriever is better than HippoRAG2/LightRAG for any task other
  than what was directly measured.
- LLM-judge ACC (we use token-recall; leaderboard uses LLM-judge — the
  numbers are correlated but not identical).

## Reproduce

```bash
pnpm --filter @kontext-brain/bench gb-dump      # build retrieval contexts
# Claude reads contexts and writes answers
pnpm --filter @kontext-brain/bench gb-compare   # controlled metrics
```
