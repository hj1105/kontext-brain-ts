# Round 19: Claude-extracted KG — high-quality ontology, retrieval still loses

## What was built

Round 18 found the heuristic KG (regex co-occurrence) underperformed
hybrid by ~20pp, and LLM-based extraction was blocked by RAM. Round 19
worked around RAM by **dispatching Claude-Code subagents in parallel
(20+ agents)** to extract entities + typed triples from every chunk:

- **Medical**: 1385/1385 chunks extracted (100%) — 14 batches × ~100 chunks
- **Novel**: 2300/3503 chunks extracted (66%) — 23 batches × 100 chunks
  (remaining 12 batches blocked by quota limit "resets 4:40am Asia/Seoul";
  these batches stalled trying to fix entity-validation errors)

Result: **real LLM-extracted KGs with typed predicates**, far cleaner than
the Round 18 heuristic:
- Medical: 2,432 unique entities, 9,333 typed-predicate edges (treats,
  causes, located_in, etc.)
- Novel: 2,909 unique entities, 3,863 typed edges

## End-to-end ACC (Llama-3.1-8B answerer + Llama-3.1-8B judge)

```
Medical (N=30):
  vanilla:    60.00%
  hybrid:     76.67%   ← still best
  multi-hop:  36.67%
  kg (Claude-extracted): 23.33%

Novel (N=30):
  vanilla:    30.00%
  hybrid:     53.33%   ← still best
  multi-hop:  46.67%
  kg (Claude-extracted): 20.00%
```

## Headline finding (honest, painful)

**Even with the best ontology we can build (Claude-quality typed
extraction), the PPR-over-KG retrieval lost decisively** to plain
hybrid (entity-BM25 + vector):
- Medical: kg 23.33% << hybrid 76.67% (**-53pp**)
- Novel: kg 20.00% << hybrid 53.33% (**-33pp**)

This is not what we hoped. The Round 18 hypothesis ("if only we had a
real LLM-extracted KG") is **disconfirmed by the data**. A high-quality
typed knowledge graph does not, by itself, make our retriever
competitive with hybrid.

## Why KG lost

Three honest reasons:

### 1. Personalized PageRank diffuses signal

PPR with α=0.15 over 9,333 medical edges propagates seed score across
the whole entity graph. Even with strong typed seeds, score reaches
many irrelevant docs. A focused BM25 hit on one specific phrase is
sharper than PPR's diffuse activation.

### 2. Coverage cliff on novel

Only 66% of novel chunks have entries in the KG (extraction quota
ran out). The KG retriever cannot return chunks that aren't in the
graph at all — 34% of the corpus is **invisible** to it. Even in the
covered 66%, the gold chunk may be a node with low connectivity.

### 3. Seed extraction is too lenient

`findSeedEntities` matches by substring or 50% token overlap. For
"Which thyroid cancer subtype is the most common?", many entities
contain "thyroid" or "cancer" — they all get seeded, then PPR
activates a broad cluster. The actual gold chunk doesn't necessarily
score highest.

## What HippoRAG2 / G-reasoner do that we don't

To match those systems:
1. **Smarter seed selection** — they use LLM-based entity recognition
   to pick exactly the right anchors (not regex/substring)
2. **Hop-aware retrieval** — they walk specific paths through the KG
   (e.g., 2-hop reasoning), not blanket PPR diffusion
3. **Hybrid with vector reranking** — KG selects candidate chunks,
   vector reranks them. We don't combine; we run KG alone.
4. **Larger entity sets** with neural disambiguation, not just
   token-substring matching

Our framework supports building this; the implementation in this
round is a baseline-grade PPR retriever. Closing the gap to HippoRAG2
would require ~2 weeks of additional retrieval-algorithm work, not
just better extraction.

## Honest takeaways

1. **Claude-extracted KG is not magic**. Quality entities + typed
   predicates, by themselves, do not produce competitive retrieval.
   The retrieval algorithm matters as much as KG quality.

2. **The hybrid retriever is genuinely strong**. Medical hybrid+8B at
   76.67% beats every leaderboard system AND beats our own
   typed-KG approach. This is good news — the simpler approach wins
   on this hardware.

3. **Round 17 result stands as the honest headline**: kontext-brain
   hybrid + Llama-3.1-8B = 76.67% Medical, beating the GraphRAG-Bench
   leaderboard top (G-reasoner+14B = 68.84%) using a *smaller* LLM.

4. **The research direction "build typed KG → use PPR" is not the
   silver bullet we hoped**. Future improvement needs algorithmic
   work on retrieval (smarter seeds, hop-aware walks, hybrid
   reranking), not just better extraction.

## Cost of this round

- ~20 Claude-Code agents dispatched in parallel
- ~3 hours wallclock for medical extraction (100% complete)
- ~6 hours wallclock for novel extraction (66% complete due to quota)
- ~50 minutes Llama-8B answer pass on all 4 retrievers × 2 domains
- ~10 minutes LLM-judge pass

Total: ~10 hours of compute + agent time. Result: a solid negative
finding that the typed-KG-PPR direction doesn't outperform hybrid
on this hardware. Not 40 hours, but enough to settle the question.

## What's shipped

- `bench/src/dump-chunks-for-claude.ts` — chunk dumper for batch
  agent processing
- `bench/data/claude-kg-{medical,novel}-batch-NNN.jsonl` — 37 batch
  files (14 medical + 23 novel)
- `bench/src/merge-claude-kg.ts` — merger that builds the full KG cache
- `bench/data/gb-{medical,novel}-kg.json` — final Claude-extracted KGs

The KG is reusable; future retrieval-algorithm experiments can load
it without re-extracting. If the algorithm side improves, this KG is
the substrate to test against.

## Headline (after Round 19)

| Bench | Best kontext-brain | Score | Notes |
|-------|-------------------|-------|-------|
| GraphRAG-Bench Medical Fact (Llama-8B) | hybrid | **76.67%** | beats #1 G-reasoner (14B) +8pp |
| GraphRAG-Bench Novel Fact (Llama-8B) | hybrid | **53.33%** | mid-pack vs leaderboard |
| HotpotQA multi-hop (Claude) | multi-hop | **100%** | both gold docs in 20/20 |
| SQuAD 2.0 single-hop (Claude) | hybrid | **96.7%** | only retrieval-failed sq-24 missed |
| Structured filter queries | attribute retrieval | **100% F1** | 0.225ms, no LLM |

Round 19 didn't move the headline — the Round 17 hybrid result is
still the best, and Round 19 is a documented negative result on the
typed-KG retrieval direction.
