# Round 13: vanilla vector RAG vs kontext-brain hybrid — LLM-equalized

## Setup

To fairly compare the kontext-brain hybrid retriever against vanilla vector
RAG, both pipelines dump retrieved contexts for the same 30 SQuAD + 20
HotpotQA queries, and Claude Code is the answerer for both. Only the
retriever differs.

- **vanilla RAG**: embed each doc with `nomic-embed-text`, cosine-similarity
  retrieve top-3, BM25 extract per retrieved doc, feed to Claude.
- **kontext-brain hybrid**: same corpus, but retrieval uses the entity
  index (proper-noun + common-noun phrase vocab, mention-weighted) combined
  0.4 × entity score + 0.6 × vector score, top-3, same BM25 extraction.

Everything downstream — chunking, extraction, LLM — is identical.

## Aggregate (kw-substring)

```
SQuAD 30 queries:
  baseline+claude     0.951
  hybrid+claude       0.951   (tied on kw-hit)

HotpotQA 20 queries (multi-hop):
  baseline+claude     0.867
  hybrid+claude       0.817
```

## Judged accuracy (manual, per-query)

| System | SQuAD 30q | HotpotQA 20q |
|--------|-----------|--------------|
| vanilla RAG + Claude Code | 27/30 = **90.0%** | 18/20 = **90.0%** |
| kontext-brain hybrid + Claude Code | 29/30 = **96.7%** | 17/20 = **85.0%** |
| Δ (kontext − vanilla) | **+6.7pp** | **-5.0pp** |

### SQuAD: kontext-brain wins (+6.7pp)

Both systems use `nomic-embed-text` for embeddings. Differences:

- **sq-22 "In what country is Normandy located?"** — ref: France
  - vanilla RAG retrieved: `southern-california-0, victoria-australia-0,
    amazon-rainforest-0` — none mention Normandy ✗
  - kontext-brain retrieved: `normans-0, southern-california-0,
    victoria-australia-0` — gold doc retrieved via entity match on
    "Normandy" ✓
  - The entity vocab includes "Normandy" as a proper noun; vanilla vector
    similarity on a short question ("In what country is Normandy located?")
    latched onto "country" / "located" tokens and surfaced the wrong
    regional articles.

- **sq-29 "common coastal pleurobrachia called?"** — ref: sea gooseberry
  - vanilla RAG: `ctenophora-20, ctenophora-8, ctenophora-0` ✗ (gold
    was `ctenophora-13`)
  - kontext-brain: `ctenophora-13, ctenophora-8, ctenophora-20` ✓
  - Entity match on "Pleurobrachia" surfaced the right paragraph.

- **sq-24 "What are cestida called?"** — ref: belt animals
  - Both retrievers failed. "Cestida" is a rare proper noun but in a
    short question the entity signal is overwhelmed by the generic
    "what / called".

### HotpotQA: vanilla RAG wins (-5.0pp)

This is a genuine negative for kontext-brain on multi-hop. 3 queries
where vanilla RAG retrieved **both** gold paragraphs but hybrid did not:

- **hp-4 "ratio of flow velocity... Saab JAS 39 Gripen?"** — ref: 2
  - vanilla retrieved `mach-number, pitot-tube, linville-falls` —
    Mach number page is there, answer deducible with world knowledge ✓
  - kontext-brain retrieved `volvo-rm12, ps-05-a, linville-falls` —
    entity matches on "Saab/Gripen" dominated, no Mach number page ✗
  - Entity matching hurt here — the question is about a *concept* (Mach
    number), not the entity mentioned (Gripen).

- **hp-6 "NFL team undefeated 1972, coach of 2004 FAU Owls?"** — ref:
  Miami Dolphins
  - vanilla retrieved `howard-schnellenberger + 2004-fau + fau-football`
    — both gold docs in context ✓
  - kontext-brain retrieved only `2004-fau + fau-football + 2016-fau`
    — Schnellenberger page missing; answer depended on world knowledge
  - Entity matching favored "Florida Atlantic" proper-noun overlap at the
    expense of the actual cross-reference.

- **hp-15, hp-16, hp-18, hp-19** — multiple cases where vanilla retrieved
  both gold docs and hybrid retrieved only one.

### Structural reason

The kontext-brain hybrid retriever was tuned for **single-entity
questions** like "What does X do?" where the entity score correctly
boosts the X-page. For **bridge/comparison multi-hop** questions ("X and
Y have what in common?") the entity signal often boosts *one* entity's
page to the top and crowds out the *other* — exactly what hurts.

On SQuAD (single-hop) the entity signal is a pure win. On HotpotQA
(multi-hop) the entity signal sometimes costs you the second gold doc.

## Implication

There is no universal winner. The choice of retriever should depend on
the **query shape**:

| Query shape | Best retriever | Why |
|-------------|----------------|-----|
| Single-hop, entity-anchored | kontext-brain hybrid | entity match wins |
| Single-hop, concept-only | vanilla vector RAG | no entity to anchor |
| Multi-hop bridge / comparison | vanilla vector RAG (or per-entity decomposed) | entity score crowds out the 2nd doc |
| Structured predicate | attribute retrieval | 100% F1, no LLM |

This is actually what the N-layer abstraction (Round 11) was designed
for. A production deployment would:
1. Classify question shape (single-hop vs multi-hop vs structured).
2. Dispatch to the appropriate retriever preset via `PipelineSpec`.

## Honest headline

**Neither retriever dominates.** Claim: "kontext-brain hybrid beats
vanilla RAG" is true on SQuAD (+6.7pp) and false on HotpotQA (-5.0pp).
The right framing is:

- For **entity-anchored factoid QA** (SQuAD shape) — use kontext-brain
  hybrid.
- For **multi-hop composition** (HotpotQA shape) — use vanilla vector RAG
  until we implement entity-decomposed / iterative retrieval (future
  work against the N-layer abstraction).

- For **structured filter queries** — attribute retrieval at 100% F1
  remains the best tool.

## All four benches together

| Bench | Winner | Score | Loser | Score |
|-------|--------|-------|-------|-------|
| SQuAD 2.0 single-hop (Claude answerer) | kontext-brain hybrid | 96.7% | vanilla vector RAG | 90.0% |
| HotpotQA multi-hop (Claude answerer) | vanilla vector RAG | 90.0% | kontext-brain hybrid | 85.0% |
| SQuAD 2.0 single-hop (Ollama answerer) | ans-ensemble | 86.7% | vanilla vector RAG | 80.0% |
| Structured filter queries | attribute retrieval | 100% F1 | vanilla vector RAG | 58.8% F1 |

The honest finding is that kontext-brain's **best contributions** are:
(1) attribute retrieval on structured data (100% F1 vs 58.8%) and
(2) hybrid retrieval on single-hop entity-anchored QA (+6.7pp on SQuAD).
Multi-hop general QA is not currently kontext-brain's strength; it's a
wash or slight loss vs plain vector RAG until dedicated multi-hop
retrievers are added.
