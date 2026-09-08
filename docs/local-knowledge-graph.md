# Local knowledge graph

`SqliteKnowledgeGraphRepository.open(dataDirectory)` implements the same
`KnowledgeGraphRepository` interface used by Resource synchronization and Task
Evidence collection. Personal users do not need a PostgreSQL server. Managed
deployments retain the existing PostgreSQL adapter and principal/ACL model.

The adapter uses Node's built-in SQLite module, loaded only when opening a local
graph. It requires Node 22.13+ with that module enabled; older runtimes get an
explicit error and never fall back to ephemeral storage. Node 24's SQLite API is
still experimental. See the [Node SQLite documentation](https://nodejs.org/api/sqlite.html).
No separate native package, service, credentials or network connection is added.

Data lives at `knowledge/graph.sqlite` beneath the host-owned data directory.
The file is created with Unix mode 0600 inside a 0700 directory; Windows uses its
filesystem ACLs. This is private local storage, not encrypted storage. Existing
directories' permissions are not silently rewritten.

SQLite transactions own atomicity and process-lock recovery. Each operation opens
and closes its connection. Writers acquire `BEGIN IMMEDIATE`, execute the callback
once and commit with synchronous durability enabled; errors roll back. Readers
observe committed state, not another asynchronous callback's partial changes.
Busy acquisition/commit retries yield to the event loop and are bounded; callback
mutations are never replayed automatically. A transaction handle cannot be used
after its callback has settled.

Resource, Chunk, Entity, Mention, Fact, Evidence and event history remain scoped to
one Organization and are validated on read/write. Source and Resource/Fact lookups
use database indexes; ontology filtering runs in SQLite. The adapter does not
grant ACLs, invent ontology nodes or approve normative records. Unknown database
versions and malformed ACLs are rejected without resetting the database.

Tests cover reopen, rollback, 16 independent writers, committed-reader visibility,
Organization isolation, Resource synchronization and invalidation, expired handles,
schema rejection, Unix permissions and a separate writer killed mid-transaction.
The collected-Task integration tests also run against this durable adapter.

The production Task sidecar now initializes the graph and local principal lazily
for [host-only Markdown registration](./local-source-registration.md). Connected
source selection, sharing grants, Task preparation and GUI registration remain
pending; storage and source capture alone do not complete the app workflow.
