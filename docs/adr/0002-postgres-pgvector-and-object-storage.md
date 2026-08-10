# ADR 0002: Store structured KG state in PostgreSQL and content in object storage

- Status: Accepted

PostgreSQL is the canonical store for organization boundaries, ontology deployments, resource and chunk metadata, Entities, Facts, Evidence, lifecycle events, ACLs, extraction jobs, and audit identifiers. pgvector stores embeddings beside the records whose ACL and lifecycle state govern retrieval.

Normalized current Resource bodies are stored as one compressed object per Resource in an S3-compatible object store; local filesystem storage is the development adapter. The KG and vectors are rebuildable from external sources. Repository and content-store ports keep use cases independent of these adapters.
