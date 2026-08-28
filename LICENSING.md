# Licensing

All original source code, documentation, examples, benchmark code, and published
packages in this repository are licensed under the
[Apache License 2.0](./LICENSE).

## Package scope

The Apache-2.0 license applies uniformly to every kontext-brain package:

| Package | Purpose |
|---------|---------|
| `@kontext-brain/core` | data model, retrieval pipelines, mapping strategies, extractive QA |
| `@kontext-brain/llm` | LangChain.js adapters for Claude, OpenAI, Ollama |
| `@kontext-brain/mcp` | MCP client connectors and source adapters |
| `@kontext-brain/loader` | YAML config loader and `KontextAgent` |
| `@kontext-brain/tool-server` | MCP server exposing kontext as tools |
| `@kontext-brain/postgres` | PostgreSQL/pgvector KG, organization RLS, ACL-aware retrieval, ontology deployments, proposal queue, extraction jobs |
| `@kontext-brain/object-storage` | S3-compatible compressed Resource content storage |
| `@kontext-brain/github` | accumulated ontology-proposal draft-PR publisher |

You may use, modify, distribute, self-host, and offer these packages in
commercial or hosted services, subject to the terms of Apache License 2.0. No
package-specific delayed-license or competing-service restriction applies to
repository-owned code.

## Third-party dependencies

Dependencies and externally sourced datasets retain their respective upstream
licenses. Apache-2.0 applies to kontext-brain's original work, not to third-party
material merely used by or stored alongside the project.

## Contributions

Unless explicitly agreed otherwise in writing, contributions submitted to this
repository are accepted under Apache License 2.0.
