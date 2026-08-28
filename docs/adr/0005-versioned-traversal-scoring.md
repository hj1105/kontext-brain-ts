# ADR 0005: Use versioned observation-based traversal scoring

## Status

Accepted

## Context

The bidirectional retriever previously received final-looking numbers from graph adapters:
`confidence`, `queryRelevance`, `evidenceSupport`, and `score`. In production and benchmark
adapters, many of those fields were constants. This made an unknown signal indistinguishable
from a measured value and duplicated ranking policy across adapters.

The old edge formula also applied `hopPenalty ** totalDepth` on every transition. A path therefore
accumulated a triangular depth penalty rather than one penalty per edge.

## Decision

Adapters report raw observations:

- provider rank and normalized retrieval observations for seeds;
- deterministic, declared, extracted, or inferred structural provenance for edges;
- query-match observations without fabricated defaults;
- active, curated, derived, distinct-resource, conflict, and stale evidence counts;
- evidence origin and freshness when known.

`CalibratedTraversalScorePolicy` is the only module that converts these observations into traversal
priority. Its parameters live in a validated, versioned `TraversalScoringProfile`. Each profile has
a stable digest, and every selected evidence item includes its seed, edge, and evidence score
breakdown. Missing values remain missing, use an explicit profile policy, and appear in the search
trace.

Transitions multiply factors in log space. Every transition factor must be in `[0, 1]`, so graph
cycles cannot increase path priority. Hop decay is applied exactly once per traversed edge, with a
separate KG-expansion factor.

`BalancedTraversalScorePolicy` remains as a deprecated compatibility policy for existing callers
that still provide legacy numeric fields. PostgreSQL runtime uses the observation-based profile.

## Deployment

Scoring profiles are immutable PostgreSQL deployments with staged, active, retired, and failed
states. The organization runtime row points to active and optional shadow profile digests. The
resolver caches the selected policy briefly, supports a full shadow traversal, and clears its cache
on activation, shadow changes, and rollback.

Schema changes are additive. Historical provenance remains `NULL`; it is never fabricated during
migration. New ingestion preserves extraction confidence, extractor version, origin, observed time,
and verified time when supplied.

## Consequences

- Production and benchmark graph adapters share the same seed fusion and scorer.
- Changing a ranking prior requires a new profile version, evaluation artifact, and activation—not
  scattered source edits.
- A shadow profile costs a second graph traversal and must be enabled only during controlled
  evaluation windows.
- Scores are query-local priorities, not probabilities.
- Legacy fields can be removed only after their usage reaches zero and rollback no longer depends
  on them.
