# Licensing

kontext-brain uses an **open-core** model. The framework you build on is
permissively licensed; the enterprise/production governance packages carry a
source-available commercial license that converts to open source over time.

## Open-source packages — Apache License 2.0

These packages are free for any use, including commercial and production, under
the [Apache-2.0](./LICENSE) license:

| Package | Purpose |
|---------|---------|
| `@kontext-brain/core` | data model, retrieval pipelines, mapping strategies, extractive QA |
| `@kontext-brain/llm` | LangChain.js adapters for Claude, OpenAI, Ollama |
| `@kontext-brain/mcp` | MCP client connectors + layer adapters |
| `@kontext-brain/loader` | YAML config loader + `KontextAgent` |
| `@kontext-brain/tool-server` | MCP server exposing kontext as tools |

The repository root is Apache-2.0; unless a subdirectory contains its own
`LICENSE`, Apache-2.0 applies.

## Commercial packages — Business Source License 1.1

These packages provide the multi-tenant, access-controlled, audited production
substrate. They are **source-available** under [BSL 1.1](./packages/postgres/LICENSE):

| Package | Purpose |
|---------|---------|
| `@kontext-brain/postgres` | PostgreSQL/pgvector KG, organization RLS, ACL-aware retrieval, ontology deployments, proposal queue, extraction jobs |
| `@kontext-brain/object-storage` | S3-compatible compressed Resource content storage |
| `@kontext-brain/github` | accumulated ontology-proposal draft-PR publisher |

**What BSL 1.1 means here:**

- ✅ You may read, modify, self-host, and use these packages in production
  **inside your own organization**, and for any non-competing commercial use.
- ✅ Each version automatically re-licenses to **Apache-2.0 four years after
  its release** (Change Date: `2030-08-23` for the current version).
- ❌ You may **not** offer these packages (modified or not) to third parties as
  a hosted/managed knowledge-retrieval, RAG, or search service that competes
  with kontext-brain's own commercial offering. That use requires a commercial
  license.

The intent is simple: build freely, run it in your own company freely, but a
competitor cannot take the governance layer and resell it as a rival hosted
service before it becomes fully open source.

## Commercial license & hosted cloud

Need to embed the commercial packages in a competing service, want a warranty,
support SLA, SSO/audit features, or the managed cloud? Contact the maintainer
to arrange a commercial license.

## Contributions

Contributions to any package are accepted under that package's respective
license. By submitting a contribution you agree it may be relicensed under the
package's Change License (Apache-2.0) on the Change Date.
