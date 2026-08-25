# ADR 0006: Adapt traversal authority from query-local route observations

## Status

Accepted as a route-suppression experiment; production activation rejected

## Context

ADR 0005 centralized traversal scoring, but its first `calibrated-v2` profile still applied the
same route priors to every query and corpus. In the Medical regression run, broad
`chunk → resource → chunk` traversal occupied 67.79% of the selected top ten. The adapter had
already ordered those neighbors, but the scorer could not observe route fanout or query-local
rank. A candidate could therefore receive full route authority merely because it was first in a
weak or one-item list.

Replacing those priors with one fitted global weight set would move the fixed-value problem rather
than solve it. Dataset or organization names must also never be runtime scoring features.

## Decision

Keep the versioned observation-based base scorer and add a query-bound policy behind the same
scoring interface. `AdaptiveRouteTraversalScorePolicy.bind(question)` classifies only coarse query
intent and returns an explainable scorer for that retrieval.

Graph adapters report two additional query-local observations for every bounded neighbor list:

- returned and pre-cut candidate counts;
- rank and, when honestly available, an adapter-normalized query score.

For normal lookup/comparison traversal, route authority is gated by:

```text
log2(1 + returned) / log2(1 + candidates)
× 1 / log2(rank + 1)
× normalized query evidence
```

An unavailable normalized score is neutral rather than fabricated. Resource-to-chunk traversal for
a summary query relaxes the fanout and absolute-match gates because broad coverage is the requested
behavior. Non-chunk seeds with an observed normalized match are likewise gated by that observation.

The formula has no dataset-specific threshold or fitted route weight. It changes per question and
per neighbor list from observed selectivity, rank, and query evidence. The versioned base profile
continues to own structural provenance, support, evidence, hop, and missing-signal policies; this
ADR does not claim that every prior has become learned or parameter-free.

Every selected result records the bound query intent, route name, fanout, rank, query evidence, and
adaptive gate in its score breakdown. The benchmark records the policy digest and feature schema.

## Evaluation decision

`adaptive-route-v3` version 2 improved both recall and raw context precision on all 2,062 Medical
and 2,010 Novel queries compared with `calibrated-v2`. Against the previous Medical bidirectional
policy it improved Recall@10 from 0.70223 to 0.71435 and precision from 0.35470 to 0.36508.

Production activation was initially deferred because Medical p95 retrieval latency was 17.30 ms
versus 10.28 ms for the previous policy, above the rollout gate. The selected results also became
91.86% direct on Medical and 98.11% direct on Novel, so this experiment validates route suppression
but does not by itself prove an N-layer advantage over an equivalent direct-only hybrid retriever.

The subsequent matched direct-only ablation disabled only graph expansion (`maxHops: 0`) while
holding candidate generation, embeddings, seed fusion, and scoring fixed. Graph-minus-direct
Recall@10 was -0.00097 on Medical (95% CI [-0.00679, 0.00485]) and -0.00100 on Novel
([-0.00398, 0.00199]). Context precision decreased by 0.00902 on Medical and 0.00174 on Novel;
both precision intervals excluded zero. The current graph-enabled configuration therefore fails
the incremental-value gate and must not be activated.

A second matched ablation repeated the comparison after weighted hybrid fusion and identical
5,000-character source-window hydration. In that setting, graph traversal increased aggregate
Recall@10 by 0.00582 on Medical (95% CI [0.00048, 0.01115]) and 0.00498 on Novel
([0.00100, 0.00896]), but decreased precision by 0.00602 and 0.00277 respectively; both precision
intervals excluded zero. The recall gain also failed to become a strict win on both deterministic
holdouts. The source-hydrated direct control remains the default quality candidate, while graph
traversal is limited to an explicit recall-first policy.

## Consequences

- Broad graph routes must earn authority from their observed selectivity and query evidence.
- The same scoring implementation can behave differently across corpora without inspecting corpus
  identity.
- Adapters that cannot report an observation receive neutral treatment and remain compatible, but
  gain less adaptation.
- Query intent classification is intentionally small and auditable; improving it requires a new
  policy version and paired evaluation.
- Source hydration and hybrid fusion may be promoted independently of graph traversal; their
  direct-only control produced the strongest overall retrieval scores in this evaluation.
- Any redesigned graph route must show a positive paired increment over the matched direct-only
  control before activation; production shadow latency profiling remains required.
