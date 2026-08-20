# Domain Language

## Organization

The security, ontology, and knowledge-graph boundary for one company. Every stored record belongs to exactly one Organization.

## Ontology

The small, company-wide vocabulary of core business concepts configured in YAML. It describes concepts such as Product, Customer, Order, Shipping, and Payment; it does not contain source facts.

## Ontology Node

A coarse concept declared by the Ontology. Fine-grained names, records, and claims are represented as Entities and Facts instead of new Ontology Nodes.

## Resource

One source-native unit such as a Notion page, Slack thread, GitHub issue, or pull request. Its external system remains the source of truth.

## Chunk

A source-native addressable part of a Resource, such as a Notion block subtree, Slack message, or GitHub comment. Generic documents use recursive text splitting only as a fallback.

## Mention

A source-local occurrence that refers to an Entity. Names, aliases, titles, and pronouns may be Mentions of the same resource-scoped Entity.

## Event

An evidence-backed occurrence or state transition involving Entities. Temporal and causal relationships connect Events without turning unsupported inference into Fact.

## Extraction Capability

A content-driven kind of knowledge extraction such as identity resolution, event extraction, or temporal and causal linking. Capabilities are selected from source evidence rather than corpus names or domain-specific modes.

## Evidence

An accessible Chunk that explicitly supports a Fact. A Fact is visible only when the requesting principal can access at least one active supporting Evidence item.

## Resource-scoped Entity

An Entity created on its first explicit mention and identified only within one Resource. It preserves important one-off concepts without asserting organization-wide identity.

## Global Entity

An Entity promoted to organization scope through a deterministic external identifier, structured business key, manual confirmation, or sufficient entity-resolution evidence.

## Fact

A minimal evidence-backed relationship consisting of subject, predicate, and object. A stable fact key identifies the same semantic relationship across source changes.

## Hypothesis

An inferred relationship that is not explicitly supported by source Evidence. Hypotheses are excluded from normal retrieval until approved or supported by new Evidence.

## Active, Stale, Inactive, and Conflict

Active records are supported by current source content. Stale records came from content being replaced and cannot support an answer. A Fact is Inactive when it has no active Evidence. Conflict means multiple active values remain for a single-value relationship.

## Curated Fact

A manually approved Fact or Evidence item. Curated data is not invalidated by routine source synchronization, but is still subject to access control and explicit privacy deletion.

## Resource-Ontology Link

An evidence-backed many-to-many relevance link between a Resource and an Ontology Node. It is derived primarily from Entities and Facts found in Chunks; document-level classification is only a prior.

## Lift, Expand, and Ground

Lift moves from concrete Chunks, Resources, or Entities toward useful Ontology anchors. Expand follows related Ontology Nodes or Entity-Fact-Entity relationships. Ground returns from those anchors to Resources, Chunks, and Evidence.
