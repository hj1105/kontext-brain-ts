# Round 9 performance re-measurement

After adding the instance-of-node entity model (`nodeId` + `attributes` +
`findByAttributes`), re-measured performance to confirm:

1. No regression on the existing SQuAD open-ended QA bench.
2. Attribute queries work as designed on structured filter queries where
   vector RAG fundamentally cannot compete.

## Result 1: SQuAD 2.0 (open-ended QA) — no regression

Same 30-query sample, same Ollama LLM, same hybrid/entity pipeline.

```
system       recall   auto-kw    ctx     latency     Δ Round 8
baseline     0.900    0.762     1587ch   5284ms      0
kw-map       0.100    0.205     2320ch   7273ms      0
bm25-map     0.500    0.545     1883ch   5298ms      0
extractive   0.567    0.507      282ch      1ms      0
entity       0.700    0.682      275ch      1ms      0
vector-docs  0.900    0.762     1506ch   4225ms      0
hybrid       0.967    0.795     1508ch   2809ms      0
```

Hybrid still wins SQuAD at 0.967 recall / 0.795 kw-hit, +49% faster than
baseline. The attribute model additions are purely additive — they do not
affect code paths used here.

## Result 2: Structured-filter queries — attribute retrieval wins decisively

Synthetic 14-entity corpus (8 databases + 6 message queues) with known
attribute schemas. 5 structured filter queries exercising boolean equality,
numeric range, set membership, and combined predicates.

Baseline indexes the prose description of each entity and retrieves via
vector similarity + LLM extraction. Attribute retrieval runs a structured
predicate query directly over the typed entity index.

```
system      precision  recall    F1     avg_latency
baseline    0.527      0.833     0.588     9595ms
attribute   1.000      1.000     1.000     0.225ms
```

### Per-query detail

| Query | baseline F1 | attribute F1 |
|-------|-------------|--------------|
| Open-source relational DBs that support JSON | 0.50 | 1.00 |
| DBs that shard AND released >2010 | 0.50 | 1.00 |
| Open-source replayable MQs | 0.89 | 1.00 |
| DBs with Apache/BSD-family license | 0.80 | 1.00 |
| DBs that are NOT open source | 0.25 | 1.00 |

### Why baseline fails on structured queries

- **q0** "Open-source relational DBs that support JSON" → baseline returned
  6 databases (postgres, mysql, sqlite, oracle, mongodb, cockroach).
  Vector similarity surfaces all docs mentioning "JSON" and "open source"
  in prose; the LLM can't reliably filter the conjunction.
- **q4** "DBs that are NOT open source" → baseline returned 6 DBs but the
  correct answer is 2 (oracle, cockroach). Negation confuses vector +
  extractive LLM.
- **q3** "Apache or BSD-family license" → baseline found 2/3 (missed
  cockroach which uses BSL); set-membership reasoning requires careful LLM
  prompting that can't be reliable at 1.5B model size.

### Why attribute retrieval wins

Structured filter queries have deterministic answers given the entity
attributes. Attribute retrieval runs as a plain in-memory predicate match
against typed instance data — no inference, no ambiguity. Vector RAG over
prose is the wrong tool for this shape of question; even with a perfect
LLM it would be slower and less precise than a structured lookup.

### Latency difference (43,000×)

0.225ms (attribute) vs 9595ms (baseline). This is honest in the same way
extractive QA vs generative RAG was honest earlier: the two systems do
different work. Attribute retrieval does not call an LLM — it's a DB-style
predicate scan. Baseline calls an LLM for each query to parse prose. The
comparison is only meaningful because both answer the same structured
question; if you have the attribute data, attribute retrieval is a better
tool regardless.

## Headline

| Bench | Winner | Why |
|-------|--------|-----|
| SQuAD 2.0 (open-ended prose QA) | **hybrid** (0.967 recall, 0.795 kw, 2809ms) | vector + entity ensemble handles paraphrase + named-thing matching |
| Structured filter queries | **attribute** (1.000 F1, 0.225ms) | deterministic predicate match on typed instances beats prose vector RAG |

Both live in `@kontext-brain/core`. Pick based on your data shape — prose
docs go through hybrid; structured facts go through attribute queries.
Many real apps need both.
