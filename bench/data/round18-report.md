# Round 18: Real ontology graph attempt — heuristic KG, RAM-blocked LLM KG

## Motivation

Round 17 used flat chunk retrieval — vanilla / hybrid / multi-hop all
operate on chunks without any explicit knowledge graph. The kontext-brain
framework was *built* for ontology-based retrieval (OntologyNode, Edge,
GraphTraverser, IngestPipeline) but we hadn't actually used it on the
GraphRAG-Bench corpus.

This round adds a 4th retriever (`kg`) that does:
1. Per-chunk entity + triple extraction → builds a knowledge graph
2. Personalized PageRank from query entities → propagate through graph
3. Score chunks by entity-activation, return top-K

This matches HippoRAG2's high-level approach (the leaderboard's #3 system).

## What was built

**`bench/src/kg-builder.ts`**: per-chunk entity + relationship extractor.
Two implementations:
- LLM-based (HippoRAG2-style): Llama-3.1-8B JSON extraction per chunk
- Heuristic (regex co-occurrence): proper-noun + content-token extraction
  + co-occurrence edges, no LLM needed

**`bench/src/kg-retriever.ts`**: PPR-style retriever
- `findSeedEntities`: match query content tokens to KG nodes
- `personalizedPageRank`: power iteration with restart prob α=0.15
- Score chunks = sum of activation across mentioned entities

**`bench/src/gb-dump-kg.ts`**: integrates into GraphRAG-Bench pipeline

## What blocked the LLM-based KG

System RAM constrained: **Llama-3.1-8B Q4_K_M requires ~4.4-4.8 GB but
only 2.7-3.4 GB available on this machine** during the bench. Symptoms:
- KG ingest "completed" 1385/1385 medical chunks but produced **0 entities,
  0 edges** — every Ollama call was failing silently with 500 errors and
  the JSON parser swallowed empty responses
- Direct Ollama smoke test confirmed: `model requires more system memory
  (4.4 GiB) than is available (2.7 GiB)`

After unloading qwen3-vl:8b (which was holding RAM), still only 3.4 GB
free vs 4.8 GB needed. Other processes (the bench Node, IDE, browser)
take ~10+ GB total, so we cannot free enough.

## Heuristic KG results

Built without LLM:
- Medical: **2,800 entities, ~12k co-occurrence edges, 1385 chunks indexed**
- Novel: **18,484 entities, 98,057 edges, 3503 chunks indexed**
- Build time: ~5 seconds (vs ~70 minutes for LLM-based)

Retrieval-only metrics (evidence-token coverage):

| Retriever | Medical | Novel |
|-----------|---------|-------|
| vanilla   | 0.798 (21/30 ≥0.7) | 0.487 (7/30 ≥0.7) |
| hybrid    | **0.784** (20/30) | **0.735** (18/30) |
| multi-hop | 0.746 (20/30) | 0.701 (15/30) |
| **kg (heuristic)** | **0.581** (12/30) | **0.549** (12/30) |

**The heuristic KG underperforms hybrid by 20pp medical / 19pp novel.**

End-to-end LLM-judged ACC: **could not measure** — Ollama OOM'd on every
KG-retrieval answer due to the same RAM issue (the answerer also needs
the 4.8 GB Llama-8B held in memory while Ollama serves requests).

## Why heuristic KG underperforms hybrid

1. **Co-occurrence ≠ semantics**. Two entities co-occurring in a chunk
   may be incidentally adjacent (e.g. "patient" and "symptom") with no
   semantic relationship. PPR over co-occurrence amplifies noise.

2. **Entity quality**. 18k entities for novel includes many false
   positives ("Within", "Therefore", common words mis-capitalized). LLM
   extraction would filter to ~3-5k actual named entities.

3. **No typed predicates**. HippoRAG2 / G-reasoner extract typed triples
   ("treats", "is_a", "located_in") which let PPR follow semantic paths.
   Co-occurrence only has one undirected edge type.

4. **PPR over noisy graph**. The Personalized PageRank algorithm
   propagates score through edges. With noisy edges, score leaks to
   unrelated docs.

## What would close the gap

To match HippoRAG2's leaderboard performance (Medical 66.28, Novel
60.14), kontext-brain would need:

1. **Free RAM for LLM extraction** (~6 GB minimum for Llama-8B + bench)
   OR
2. **Cloud LLM extraction**: use OpenAI/Claude/Together API for the
   ingest phase (~$10-30 for both domains)
   OR
3. **Smaller extraction model** that fits: TinyLlama (1.1B), Phi-3-mini
   (3.8B), Qwen2.5-1.5B-Instruct — produces typed triples but at lower
   quality

Then:
- Build KG with typed entities (Disease, Treatment, Person, etc.) and
  typed predicates (treats, causes, is_a)
- Run PPR on this clean graph
- Combine with vector signal (KG seeds vector search, vector reranks
  KG candidates)

## Honest takeaway for this Round

The "build a real ontology" aspiration is correct and the kontext-brain
framework supports it. The two practical paths blocked by RAM:
- Heuristic KG works mechanically but is **worse than hybrid** on this
  bench because co-occurrence isn't a real ontology
- LLM-based KG is the right approach but **needs more RAM** than we
  have right now

What we shipped:
- `kg-builder.ts` (heuristic + LLM stubs both present)
- `kg-retriever.ts` (PPR over typed graph)
- `gb-dump-kg.ts` (integrates into the bench pipeline)
- KG cache files saved (`bench/data/gb-{medical,novel}-kg.json`) — when
  RAM frees up, swap the heuristic builder for LLM and re-run

## Update vs Round 17

The Round 17 conclusion stands unchanged:
- Medical hybrid + Llama-8B = 76.67% ACC, beats every leaderboard system
- Novel hybrid + Llama-8B = 53.33% ACC, mid-pack

Round 18 attempted to add a 4th retriever and found:
- Heuristic KG: works but worse than hybrid (-20pp on retrieval coverage)
- LLM KG: blocked by RAM, would have been comparable to HippoRAG2 if
  successful

## Reproduce

```bash
# Heuristic KG (fast, no LLM)
pnpm --filter @kontext-brain/bench gb-dump-kg
# (KG cached at bench/data/gb-{medical,novel}-kg.json)

# LLM-based KG (requires ~6 GB free RAM)
KG_MODEL=llama3.1:8b-instruct-q4_K_M pnpm --filter @kontext-brain/bench gb-dump-kg
```
