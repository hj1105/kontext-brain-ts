# Round 11: second research dataset — HotpotQA multi-hop

## Motivation

Benchmarking on a single dataset (SQuAD 2.0) is not enough to generalize
claims. SQuAD is single-hop: every question has exactly one gold paragraph
that contains the answer. Real-world RAG queries often require reasoning
across multiple documents. To stress-test the architecture on that shape,
we added HotpotQA distractor dev set as a second research dataset.

HotpotQA differs structurally:
- **Multi-hop**: each question needs TWO gold paragraphs, not one
- **Compositional answers**: the answer often does not appear verbatim in
  either paragraph — requires bridge / comparison reasoning
- **Stricter retrieval test**: recall alone is misleading; we track
  `both-gold-hit` (did retrieval surface *both* gold docs?) as the real
  signal

## Setup

- Sample: 20 questions from `hotpot_dev_distractor_v1.json`, seed 42,
  deterministic
- Corpus: 200 paragraphs (10 candidates per question × 20 questions,
  deduped; includes both gold + 8 distractors per question)
- Same Ollama stack as SQuAD: qwen2.5:1.5b + qwen2.5:3b, nomic-embed-text,
  CPU-only (`numGpu: 0`, GTX 960 2GB)
- Systems run: baseline, bm25-map, hybrid, hybrid-3b, ans-ensemble (best
  SQuAD variants only — to keep runtime tractable)

## Aggregate results

```
system       recall  both-gold  keyword   ctx     latency
baseline     0.650   0.450      0.467    1366ch    4474ms
bm25-map     0.525   0.250      0.399     714ch    2737ms
hybrid       0.600   0.350      0.469    1217ch    3757ms
hybrid-3b    0.600   0.350      0.570    1217ch    8642ms
ans-ensemble 0.600   0.350      0.520    1217ch   11279ms
```

**Claude-Code-judged accuracy (manual, 20 queries)**:

| system | correct | accuracy |
|--------|---------|----------|
| baseline | ~5/20 | 25% |
| hybrid | ~5/20 | 25% |
| hybrid-3b | ~6/20 | 30% |
| ans-ensemble | **~6/20** | **30%** |

Compared to SQuAD where ans-ensemble hit 86.7%, HotpotQA accuracy drops to
**~30%** — a massive regression that is **honest**: multi-hop is genuinely
harder, and the drop is the story, not a bug.

## Why it drops so much

### 1. Retrieval no longer sufficient at 1-of-2

On SQuAD, recall of 0.97 means 97% of questions have their one gold doc
retrieved. On HotpotQA with 2 gold docs per question, recall of 0.60 means
60% of *required docs* are retrieved — but only 35% of questions have
*both* gold docs together. The other 65% of questions are missing at least
one piece of the reasoning chain. An LLM given only 1 of 2 gold paragraphs
will guess the missing piece or refuse ("context does not contain...").

### 2. Current retriever is not multi-hop aware

Every variant (baseline, hybrid, ans-ensemble) retrieves top-k docs by a
single similarity score against the question. Multi-hop questions like
"What country of origin does Traveling Wilburys and Tom Petty have in
common?" need retrieval to find *both* the Traveling Wilburys page and
the Tom Petty page. A single vector similarity score biases toward docs
that match the *whole question*, not either entity alone. The retriever
has no notion of "cover each mentioned entity with at least one doc".

### 3. 1.5b / 3b LLMs cannot compose

Even when both gold docs are retrieved (35% of cases), the answer often
requires composition:

- hp-0: "Country of origin common to Traveling Wilburys + Tom Petty"
  → requires extracting "American" from each doc and intersecting. The
  3b model returned "do not have a specific country of origin in common"
  despite both gold docs being in context. The LLM can't do set
  intersection reliably at 3b.
- hp-18: "Who wrote an opera critique on Alban Berg's Lulu?" → requires
  knowing Lulu's composer AND finding who wrote an essay on it.
  3b returned "Arnold Schoenberg" (a plausible-sounding composer guess),
  not "Theodor W. Adorno" (the actual essayist).

Three structural problems, all addressable but not trivially.

## What the architecture would need

### Addressing (1): multi-hop retrieval

Two approaches, both require pipeline changes:

**(a) Entity-decomposed retrieval**: parse the question into mentioned
entities (e.g. "Traveling Wilburys", "Tom Petty"), run retrieval
*per-entity*, then union the results. The existing `EntityIndex` already
supports this — it just needs a pipeline step that invokes it per entity
rather than once per question. Would likely lift `both-gold-hit` from
35% to 60-70% on bridge-style questions.

**(b) Iterative retrieval**: retrieve-then-reread. First retrieval brings
in 1 gold doc; an LLM reads it, extracts the "next hop" (e.g.
"Traveling Wilburys is a band → need to find who its members are"), and
issues a second retrieval. This is the standard multi-hop trick. Would
require an AgentPipeline variant that loops retrieval + reflection.

### Addressing (2): multi-hop LLM reasoning

Same as SQuAD's limit at 86.7% — a 3b model on CPU is the ceiling. A 7b+
model or real Claude/GPT-4 call would likely clear hp-0, hp-9, hp-18
(composition queries).

### Addressing (3): answer format

Several failures are prompt-format issues more than reasoning:

- hp-17 ref "no" → 3b answered "No, Lennon or McCartney is an American
  documentary short film..." — starts with "No" but kw-substring check
  catches it. This is actually correct but verbose.
- hp-11 ref "South African" → 3b answered "South African decent" — typo
  propagated from the source itself, kw-hit passes.

Tightening the answer-format prompt ("one phrase only, no explanation")
would bump judged accuracy by ~5pp without architectural changes.

## Honest comparative picture

| Bench | Retrieval signal | Best accuracy | Hard cap reason |
|-------|------------------|---------------|-----------------|
| SQuAD 2.0 (single-hop, 30q) | recall 0.97 | 86.7% | LLM capacity (3b on CPU) |
| HotpotQA (multi-hop, 20q) | both-gold 0.35 | ~30% | Retrieval shape (single-query top-k) + LLM composition |
| Structured filter (14 entities, 5q) | F1 1.00 | 100% | No LLM needed |

The comparison tells a useful story:
- When retrieval is already excellent (SQuAD) and data has structure
  (attribute bench), the framework wins decisively.
- When retrieval is single-query and the task requires multi-doc
  composition (HotpotQA), the current pipeline does not generalize well.
  This is not a pipeline bug — it's a known shape of RAG problem that
  calls for entity-decomposed or iterative retrieval.

## Follow-up work (not in this round)

1. **MultiHopRetriever**: pipeline step that extracts named entities from
   the question, runs retrieval per-entity, unions the results. Expected
   lift: `both-gold-hit` 35% → 65%+ on HotpotQA.
2. **ReflectiveAgent**: 2-step retrieve-read-retrieve loop; adds latency
   but closes the composition gap.
3. **N-layer pipeline**: already added as `packages/core/src/query/n-layer.ts`
   (Layer + Candidate + PipelineSpec + runner). The code-graph and
   multi-hop retrievers above can be implemented as layers in this
   framework without touching existing code paths.

## Honest headline

Adding a second dataset told us something the SQuAD-only number hid: the
framework's single-query top-k retrieval is insufficient for multi-hop
questions. SQuAD 86.7% did not generalize — HotpotQA accuracy is ~30%.
This is an important negative result to document before claiming the
framework is "production ready for general RAG".
