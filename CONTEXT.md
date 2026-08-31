# Domain Language

## Organization

The top-level security, ontology, and knowledge-graph boundary for one tenant. An Organization represents an individual in personal mode or a company in managed mode, and every persisted knowledge, governance, and Task record belongs to exactly one Organization.

## Ontology

The small, Organization-wide vocabulary of core business concepts configured in YAML. It describes concepts such as Product, Customer, Order, Shipping, and Payment; it does not contain source facts.

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

## Claim

A candidate proposition extracted from source text. A Claim becomes a Fact only after its Entity references, capability constraints, and exact quoted Evidence are validated. An inferred Claim remains a Hypothesis and is excluded from factual retrieval.

## Extraction Capability

A content-driven kind of knowledge extraction such as identity resolution, event extraction, or temporal and causal linking. Capabilities are selected from source evidence rather than corpus names or domain-specific modes.

## Evidence

An accessible Chunk used as explicit support for a Fact or as provenance for a normative record. A supported claim is visible only when the requesting principal can access the required active Evidence.

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

## Relation Support

The number of distinct Evidence chunks that explicitly support the same directed subject-predicate-object relationship. It is an observation, not a probability or a query score.

## Relation Origin

The way a relationship entered the graph: source co-occurrence, automated extraction, or manual curation.

## Traversal Score

A query-local priority used to order graph exploration. It is derived for one retrieval and is never stored as an Ontology or Fact property.

## Lift, Expand, and Ground

Lift moves from concrete Chunks, Resources, or Entities toward useful Ontology anchors. Expand follows related Ontology Nodes or Entity-Fact-Entity relationships. Ground returns from those anchors to Resources, Chunks, and Evidence.

## Governance Scope

The boundary at which a normative record is accepted and applied: Personal, Workspace, Codebase, or Organization. A narrower scope may add constraints but cannot weaken a managed Organization rule.

## Knowledge Space

The ACL-filtered view of Resources, Evidence, and normative records available to one principal within an Organization. A delegated capability is limited to one Knowledge Space and cannot reveal or authorize content outside it.

## Codebase

A version-controlled body of code governed as one unit inside an Organization. A Codebase remains distinct from the Organization so that one security boundary can contain multiple repositories or monorepo partitions.

## Code Symbol

A source-language declaration with a stable semantic identity inside a Codebase. A behavior-bearing Code Symbol, such as a function or method, is the minimum unit treated as one piece of logic.

## Planned Symbol

The intended identity and responsibility of a Code Symbol before that symbol exists in synchronized code. It is bound only when semantic synchronization finds one unambiguous actual Code Symbol.

## Code Symbol-Ontology Link

A typed relevance link from a Code Symbol to an Ontology Node. Curated, deterministic, and proposed links remain distinguishable, and a proposed link alone has no enforcement authority.

## Decision

An approved normative choice that code and work must respect within its Governance Scope. Decision content is immutable by revision; a later revision supersedes rather than edits the earlier one.

## Domain Term

An approved name and meaning used consistently in tasks, public code language, schemas, and user-visible behavior. A Domain Term is normative within its Governance Scope rather than merely a frequently observed phrase.

## Invariant

A normative condition that must remain true and is bound to one or more verifiers. Unlike a Fact, an Invariant becoming unverified is an alarm rather than a reason to remove the rule.

## Normative Proposal

An unapproved candidate Decision, Domain Term, or Invariant. A Normative Proposal can inform a user but cannot constrain code or approve work until accepted by a person with the required authority.

## Local Acceptance

A user-approved normative activation in Personal or Workspace scope. It is immediately usable and editable locally but does not become Organization-canonical until promoted through the managed approval path.

## Task

A bounded unit of intended change whose progress is derived from its Task Contract, submitted Evidence, and applicable Invariant status. Agents may submit work and Evidence but do not author Task state directly.

## Task Contract

The explicit intent, acceptance criteria, non-goals, and targets for one Task. Exploration may begin without a complete Task Contract, but the Task cannot be completed without one.

## Task Context Snapshot

The immutable set of applicable Decision, Domain Term, Invariant, source-freshness, and access-control revisions fixed for a Task. A later accepted revision makes the snapshot stale until it is explicitly refreshed.

## Logic Work Item

A schedulable unit of work over one behavior-bearing Code Symbol or a tightly coupled group of Code Symbols. Logic Work Items form a dependency graph and carry bounded write authority.

## Context Receipt

Proof that a Logic Work Item received a particular Task Context Snapshot slice before changing its target Code Symbols. It identifies the governing revisions and Evidence without replacing their canonical content.

## Change Bundle

The structured handoff produced by a Logic Work Item. It binds a patch and changed Code Symbols to the context, Evidence, and Verification Runs claimed for that work.

## Verification Run

An observed verifier result bound to one code revision and one Task Context Snapshot digest. A result from a different revision or digest cannot prove the current Task complete.

## Review Finding

An independently reported concern that binds a code location or Code Symbol to an acceptance criterion, normative rule, or Evidence item. Its author and reviewer runtime provenance is retained, and a runtime whose change caused a Review Finding cannot report or approve its resolution.

## Accuracy Manifest

The end-to-end proof connecting a Task Contract, Task Context Snapshot, normative revisions, Evidence, Logic Work Items, Change Bundles, Code Symbols, Verification Runs, and resulting commit.

## Drift Finding

A recorded mismatch discovered when a new normative revision may affect previously verified code. A Drift Finding proposes revalidation or migration work but does not silently edit code.
