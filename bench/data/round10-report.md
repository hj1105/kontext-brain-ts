# Round 10: pushing past 86.7% — honest result

## Goal

Prior best: `ans-ensemble` (hybrid@1.5b + hybrid@3b with a 3b judge) at
**26/30 = 86.7%** on the 30-query SQuAD 2.0 sample. The remaining 4 failures
were sq-14 (aerobic), sq-20 (Article 102), sq-24 (belt animals), sq-28
(proteins). Target for this round: ≥27/30 = 90%+.

## What was tried

A multi-candidate `synthesis` variant:

1. **3rd candidate — extractive-on-hybrid**: run hybrid retrieval, then
   extract the single highest-scoring sentence from the retrieved doc bodies
   (no LLM). This was intended to rescue sq-14, where the extractive pipeline
   had previously found the "aerobic exercise" sentence that hybrid's
   BM25-compressed context compressed away.
2. **Query expansion via prefix match**: for any question token not present
   in the corpus vocabulary, add corpus tokens sharing a 4+ char prefix as
   OR'd expansion terms. Intended to fix sq-24 (`cestida` → `cestids`).
3. **Exclusion detection**: regex-parse "besides X", "other than X",
   "except X", "in addition to X" patterns and pass the excluded term list
   to the judge prompt with "reject any candidate that gives those" guidance.
   Intended to fix sq-28 where the question was "besides fats, fatty acids,
   and amino acids, what other..." and all candidates returned "carbohydrates"
   instead of the reference "proteins".
4. **Two-pass judge prompt**: rather than ask the 3b judge to pick from 3
   candidates (initial attempt, regressed to 25/30 because the judge kept
   picking extractive's verbatim sentence over the shorter correct LLM
   answer), the second attempt showed extractive as a "grounding sentence
   from source" and asked the judge to pick A or B based on which aligns
   better with that grounding.

## Actual result

```
system       recall  keyword   ctx     latency
hybrid       0.967   0.795    1508ch    2502ms
ans-ensemble 0.967   0.812    1508ch   14581ms   ← prior best
synthesis    0.967   0.734    1750ch   20900ms   ← new variant, worse
```

Judged against references (Claude Code manual scoring, not kw-substring):

| Variant | Correct | Accuracy |
|---------|---------|----------|
| hybrid@1.5b | 24/30 | 80.0% |
| hybrid@3b | 25/30 | 83.3% |
| ans-ensemble | 26/30 | 86.7% |
| **synthesis (round 10)** | **24/30** | **80.0%** |

The synthesis variant **regressed 2 queries** relative to ans-ensemble:

- **sq-16 "Where can ctenophores be found in large amounts?"** — ref "in bays".
  - ans-ensemble → "Ctenophores can be found in large numbers in bays..." ✓
  - synthesis → "Ctenophores may be abundant during the summer months in some
    coastal locations." ✗
  - Cause: the extractive grounding sentence surfaced a "summer months /
    coastal locations" sentence, and the judge chose hy-3b's matching wording
    over hy-1.5b's correct "in bays".
- **sq-1 "What is the seldom used force unit equal to 1000 newtons?"** — ref
  "sthène".
  - ans-ensemble → "...The sthène is equivalent to 1000 N." ✓
  - synthesis → "The metric slug (or mug or hyl) is equivalent to about 1000 N."
    ✗
  - Cause: extractive picked the metric slug sentence as grounding (its
    `kilogram-force`/`1000 N`/`seldom` overlap is high), biasing the judge
    toward the slug variant.

Targeted fixes (sq-14, sq-20, sq-24, sq-28) did not land:

- **sq-14 aerobic**: extractive did surface the "aerobic exercise" sentence
  as grounding, but the judge still picked hy-3b's "Professional athletes..."
  answer. The judge cannot reliably trade keyword-overlap-with-grounding
  against faithfulness-to-grounding.
- **sq-20 Article 102**: hy-1.5b had the correct "Article 102" and
  extractive also had "Article 102", yet the judge picked hy-3b's
  hallucinated "Article 139/2004/EC". The same 3b judge model is the one
  that hallucinated the answer in the first place — it cannot correct its
  own hallucination reliably.
- **sq-24 belt animals**: query expansion did not activate (corpus does
  contain the exact token "Cestida", so `cestida` is not expanded). The
  failure mode is retrieval: hybrid retrieved `huguenot-1, steam-engine-0,
  ctenophora-0` rather than the correct `ctenophora-19`, because vector
  similarity on the short-question embedding latched onto generic
  "what/called" tokens. Fixing this requires boosting entity-weight when
  the question contains a low-frequency proper noun, which is out of scope
  for the bench script and belongs in the retriever.
- **sq-28 proteins**: exclusion regex parsed correctly ("fats, fatty acids,
  amino acids") and was passed to the judge, but the LLM still returned
  "carbohydrates". The source does say "All fats, fatty acids, amino acids,
  and proteins contain oxygen" — the exact answer is available — but the
  judge model at 3b size cannot reliably parse the conjunction after the
  exclusion filter.

## Root cause

The 3b judge and the 3b answerer are the same model class. It cannot
reliably overrule its own weaker variants. When the 3b model is wrong
(sq-20, sq-28), asking it to pick between its answer and a correct 1.5b
answer, it systematically picks its own. When it's right (sq-1), asking it
to re-evaluate against a grounding sentence sometimes makes it second-guess
correctly — sometimes not.

## What would actually push past 90%

Based on the per-query failure analysis:

| Query | Fix required |
|-------|--------------|
| sq-14 aerobic | Stronger LLM (7B+) that doesn't conflate "aerobic exercise" with "professional athletes in American football" |
| sq-20 Article 102 | Stronger LLM that doesn't hallucinate a plausible-sounding legal citation |
| sq-24 belt animals | Retriever change: boost entity weight for rare proper nouns in question |
| sq-28 proteins | Stronger LLM that parses exclusion conjunctions |

Three of four failures are LLM capacity failures, not pipeline failures.
The qwen2.5:3b model on CPU is a hard ceiling around 86.7% on this SQuAD
sample. Moving to qwen2.5:7b or a real Claude/GPT-4 call would likely hit
≥27/30, but requires either GPU >4GB or paid API.

## Decision

- **Ship ans-ensemble at 86.7% as the current best.**
- Keep the synthesis code in `squad-runner.ts` for reproducibility and as a
  negative result — worth documenting that naive multi-candidate synthesis
  with a same-class judge regresses, not improves.
- Do **not** update README's headline accuracy number; 86.7% stands.
- Do **not** claim 90%+ — honest benchmark reporting matters more than the
  number.

## Honest headline

| Bench | Best | Score | Constraint |
|-------|------|-------|------------|
| SQuAD 2.0 (30-query sample) | ans-ensemble | **26/30 = 86.7%** | CPU qwen2.5:1.5b + 3b, 2GB GPU |
| Structured filter queries (14 entities, 5 queries) | attribute retrieval | **5/5 = 100% F1** | No LLM call |

Pushing SQuAD past 90% on this hardware is blocked on LLM capacity, not
retrieval quality. The 3-layer ontology + entity model already retrieves
the correct document for 29/30 queries (recall 0.967); the remaining loss
is the answerer, not the retriever.
