# SQuAD 2.0 bench — Claude Code manual quality judgment

Each of 30 questions scored for whether the answer contains/conveys the
reference answer. Scoring: 1 = correct or substantively correct, 0 = wrong
or missing.

## Per-query scoring

| Query | Ref answer | baseline | kw-map | bm25-map | extractive | entity |
|-------|-----------|----------|--------|----------|------------|--------|
| sq-0 | nerves | ✓ | ✓ | ✓ | ✓ | ✓ |
| sq-1 | sthène | ✓ | ✗ (kgf) | ✗ (kgf) | ✗ (kgf) | ✗ (kgf) |
| sq-2 | Commission v Italy | ✓ | ✗ (unnamed) | ✗ | ✗ off-topic | ✓ |
| sq-3 | OpenTV | ✓ | ✗ | ✗ | ✗ off-topic | ✓ |
| sq-4 | four | ✓ | ✓ | ✓ | ✓ | ✓ |
| sq-5 | 61 | ✗ (46.2%) | ✗ | ✓ | ✓ | ✓ |
| sq-6 | cnidarians | ✓ (Cnidaria) | ✗ (Ctenophora) | ✗ (Ctenophora) | ✓ | ✗ no match |
| sq-7 | 27-30 | ✓ | ✗ (50%) | ✓ | ✓ | ✓ |
| sq-8 | Sony | ✓ | ✓ | ✓ | ✗ off-topic | ✓ |
| sq-9 | organic molecules | ✓ | ✗ | ✓ | ✓ | ✓ |
| sq-10 | tentilla tentacles | ✓ | ✗ (four auricles) | ✗ | ✗ lobates | ✓ |
| sq-11 | Article 101(1) | ✓ | ✗ (107) | ✓ | ✓ | ✓ |
| sq-12 | Mnemiopsis | ✓ | ✗ (Ciona) | ✓ | ✗ Geneva | ✓ |
| sq-13 | 356 ± 47 tonnes/ha | ✓ | ✗ (10B tons) | ✓ | ✓ | ✗ off-topic |
| sq-14 | aerobic | ✓ | ✗ (interval) | ✗ (football) | ✓ | ✓ |
| sq-15 | 1965 | ✓ | ✗ (1970) | ✗ (1970) | ✗ Huguenot | ✗ no match |
| sq-16 | In bays | ✗ (summer months) | ✗ | ✗ (coastlines) | ✗ construction | ✓ |
| sq-17 | Climate fluctuations | ✓ | ✗ | ✗ | ✗ | ✗ |
| sq-18 | risen <2% per year | ✓ | ✗ ($2.80/barrel) | ✓ | ✓ | ✗ |
| sq-19 | Computational complexity theory | ✓ | ✗ (TCS) | ✓ | ✓ | ✗ related fields |
| sq-20 | Article 102 | ✓ | ✗ (Commission) | ✓ | ✓ | ✓ |
| sq-21 | withstand waves | ✓ | ✗ | ✗ | ✗ oxygen | ✓ |
| sq-22 | France | ✓ (said France) | ✓ | ✓ | ✗ irrelevant | ✓ |
| sq-23 | one million | ✓ | ✗ | ✓ | ✓ | ✓ |
| sq-24 | belt animals | ✗ (comb jellies) | ✗ | ✗ | ✗ | ✗ |
| sq-25 | buildings, infrastructure, industrial | ✓ | ✗ | ✓ | ✓ | ✓ |
| sq-26 | Industrial Revolution | ✓ | ✓ | ✓ | ✓ | ✓ |
| sq-27 | O2 | ✗ (singlet oxygen) | ✗ | ✗ | ✗ off-topic | ✗ (dioxygen) |
| sq-28 | proteins | ✗ (carbohydrates) | ✗ | ✗ | ✗ | ✗ |
| sq-29 | sea gooseberry | ✗ | ✗ | ✗ | ✗ | ✓ |

## Claude Code judge totals

| System | Correct / 30 | Answer accuracy |
|--------|-------------|-----------------|
| **baseline** | **24** | **80.0%** |
| kw-map | 5 | 16.7% |
| bm25-map | 16 | 53.3% |
| extractive | 15 | 50.0% |
| **entity** | **20** | **66.7%** |

## Cross-check vs auto-metric

| System | Auto keyword-hit | Claude Code judge | delta |
|--------|-----------------|-------------------|-------|
| baseline | 0.762 | 0.800 | +3.8pp |
| kw-map | 0.205 | 0.167 | −3.8pp |
| bm25-map | 0.545 | 0.533 | −1.2pp |
| extractive | 0.507 | 0.500 | −0.7pp |
| entity | 0.682 | 0.667 | −1.5pp |

The auto keyword-hit metric was within ±4pp of my manual judgment for every
system. It tends to slightly over-credit partial matches (e.g. answer says
"46.2%" but reference is "61" — auto gives 0 but my judge also gives 0, yet
auto sometimes credits partial tokens incorrectly).

## Key observations from reading the answers

1. **Baseline vector RAG is genuinely strong on SQuAD** — 80% answer
   accuracy. It gets the right paragraph 90% of the time and the qwen2.5:1.5b
   answer paraphrase is usually faithful to it. The 20% it missed were
   cases where the right paragraph was retrieved but the model hallucinated
   a nearby number (sq-5 46.2% vs 61, sq-28 carbohydrates vs proteins).

2. **Entity retrieval is the best of the kontext variants** at 67% — 13pp
   behind baseline. Its wins are surprising: sq-21 ("Why are coastal species
   tough?"), sq-29 ("common coastal pleurobrachia"), sq-16 ("Ctenophores in
   bays") — these are all cases where the right paragraph contains a named
   entity (`coastal species`, `Pleurobrachia`, `Ctenophores`) that matches
   the query entity exactly.

3. **Entity's failures** are mostly on queries without proper-noun anchors.
   sq-1 ("sthène"), sq-15 ("Edmonds"), sq-17 ("Savanna"), sq-18 (oil
   prices), sq-19 ("computational complexity theory") — these require
   either concept-level or syntactic matching, not entity matching.
   sq-24 ("cestida" → "belt animals") also failed — "belt animals" is
   never capitalized in the text so the proper-noun extractor didn't
   catch it.

4. **kw-map retrieved junk paragraphs on almost every query** (usually
   ctenophora-17, ctenophora-16, force-43, force-0). The auto-ontology from
   Wikipedia article titles has no signal for SQuAD questions — the
   category labels don't correlate with what the questions ask about. This
   is a configuration failure, not an algorithmic one: with hand-crafted
   node descriptions it would do much better.

5. **bm25-map got 53%** by catching the easier factoid questions with
   clear keyword overlap but fumbling synthesis questions and the same
   junk-category issue in edge cases.

6. **Extractive at 50%** struggled because SQuAD answers often require the
   LLM to paraphrase and compress information; raw top sentences aren't
   always the cleanest answer form on Wikipedia prose.

## Speed / cost / accuracy trade-off (Claude Code judge scored)

| System | Accuracy | Context | Latency | Points per latency-second |
|--------|----------|---------|---------|---------------------------|
| baseline | 80.0% | 1587ch | 5076ms | 0.158 |
| bm25-map | 53.3% | 1883ch | 5154ms | 0.103 |
| extractive | 50.0% | 282ch | 1ms | 500 |
| **entity** | **66.7%** | **275ch** | **1ms** | **667** |

If quality is the only priority: **baseline wins** (80% correct).

If you can accept a 13pp accuracy drop for 5000x lower latency and 6x
smaller context: **entity is the practical winner** — 67% correct in
under 1ms with no LLM call.

Extractive is 17pp worse than entity for the same latency/cost — no
reason to pick it over entity on SQuAD.

## Honest headline

On a real research benchmark (SQuAD 2.0 dev, 30-query sample), **the
standard vector-RAG baseline beats every kontext variant on raw answer
accuracy**. Entity retrieval comes second at 67% but with 5000× lower
latency and 6× smaller context. Keyword-based ontology routing without
hand-crafted descriptions performs very poorly (17%).

The tech-docs bench where kontext variants "beat baseline" was unfair in
its own way: the ontology there was hand-crafted with tight descriptions
matching the domain, and the entity vocabulary was known up-front. On a
dataset where those luxuries don't exist, baseline wins on quality while
kontext's extractive/entity paths win on cost/speed.
