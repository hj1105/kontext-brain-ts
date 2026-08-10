# ADR 0001: Treat the knowledge graph as an evidence-backed derived index

- Status: Accepted

External systems remain the source of truth. The knowledge graph stores Resources, source-native Chunks, Entities, minimal subject-predicate-object Facts, and links to supporting Evidence. Inferred relations are Hypotheses and do not participate in normal factual retrieval until supported or approved.

Source updates make old derived Evidence stale immediately and activate a validated replacement atomically. Facts have a stable key and current state plus append-only lifecycle events, so disappearance and restoration do not create unbounded fact versions. Routine deletion is recoverable; an explicit privacy purge irreversibly removes content and Evidence while retaining only a non-sensitive tombstone.
