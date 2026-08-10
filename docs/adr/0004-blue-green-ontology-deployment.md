# ADR 0004: Activate ontology YAML through blue-green deployment

- Status: Accepted

The deployed ontology is identified by a SHA-256 hash of its YAML content. When that hash differs from the active deployment, the system validates and compiles a candidate graph, rebuilds affected derived data, and runs regression checks before an atomic active-pointer switch.

An invalid candidate never replaces the active graph. YAML and Git history are the schema history; the database stores the content hash and optional Git commit rather than introducing a separate ontology-version DSL. Schema changes enqueue idempotent extraction jobs keyed by Resource content hash and ontology hash.
