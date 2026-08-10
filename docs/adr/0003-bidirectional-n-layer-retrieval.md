# ADR 0003: Use bounded best-first traversal for N-Layer retrieval

- Status: Accepted

N-Layer retrieval is a typed, bidirectional graph search rather than a fixed pipeline or DAG. Multiple seeds may begin at a Chunk, Resource, Entity, Fact, or Ontology Node. The search adaptively lifts, expands sideways, and grounds to accessible Evidence.

A priority queue explores the highest-scoring frontier first while retaining multiple paths. ACL is a hard pre-filter. Hop, KG-hop, visited-node, candidate, score, and time budgets bound the search; a node is revisited only when reached with a better score. The online path may use at most three Entity-Fact-Entity hops, while deeper analytical work runs asynchronously.
