# Accurate Code Ontology and Orchestration — Design Note

Status: **Accepted for implementation.** Architectural choices are recorded in
ADR 0005 through ADR 0009. This note is the implementation contract and phased
delivery plan; it is not itself an ADR.

Last reviewed: 2026-08-31

Implementation snapshot: the local TypeScript vertical slice for Phases 0–4 is
implemented, including concrete workspace verifiers, recovery, quarantine,
Change Bundles, and an independently audited Accuracy Manifest. Phase 5 now has
official Codex and Claude CLI adapters, runtime inspection, isolated Git
worktrees, persisted scope leases, bounded DAG scheduling, and provider-safe
checkpoint transfer. Automatic semantic integration, blind cross-runtime
review, and schedule resume are not yet implemented. Scheduling is asynchronous:
accepted jobs and terminal results are private, digest-checked sidecar state.
Durable cancellation reaches the active Codex or Claude child process and does
not become `cancelled` until workers stop and leases release. An unfinished job
becomes `interrupted` after its owning sidecar process stops. Organization
topology and additional-language certification remain later phases.

## 1. Product outcome

Kontext Brain must help an agent write **accurate code**, not merely retrieve
plausible context. A change is accurate only when all of the following hold for
the same code revision and the same context digest:

1. every Task acceptance criterion has a passing verifier;
2. active Decisions and Invariants are respected;
3. canonical Domain Terms are used where domain language is exposed;
4. the claims above link to accessible, current Evidence;
5. regression checks pass and no mandatory result is stale or inconclusive.

The product combines the existing evidence-backed knowledge graph with a
normative governance layer, a verifiable Task model, and Codex-led multi-runtime
orchestration. The main thread manages the whole flow. Bounded worker agents
write and review code one Logic Work Item at a time after consulting Kontext
Brain for the behavior-bearing Code Symbols they own.

An AI-extracted Decision, Domain Term, or Invariant is a Normative Proposal. It
may be shown and discussed immediately, but it does not block editing, satisfy
approval, or change completion until an authorized person accepts it.

### Non-goals

- Kontext Brain does not replace compilers, tests, linters, Git, or provider
  sandboxes.
- An LLM summary never becomes the enforcement source for a normative rule.
- Agreement between two models is not proof when a deterministic verifier
  fails.
- The product does not copy subscription cookies, store provider credentials,
  or silently switch a user to usage-billed APIs.
- The product does not require Notion, Slack, or an Organization service; a
  person working only from local sessions, Markdown, code, and Git is a complete
  supported topology.

## 2. Existing foundation and preserved boundaries

ADR 0001 remains authoritative: external systems are sources of truth and the
knowledge graph is an evidence-backed derived index. Resource, Chunk, Entity,
Fact, Evidence, ACL, and lifecycle semantics are reused rather than duplicated.

`SyncResourceUseCase.execute(snapshot: ResourceSnapshot)` already accepts
pre-extracted entities and facts. Its current behavior provides the required
code synchronization seam:

- unchanged active Resource content is a no-op;
- replacing content stales old Chunks, mentions, and derived Evidence;
- a Fact with no active Evidence becomes inactive;
- a submitted Fact without Evidence is rejected;
- `affectedFactKeys` identifies relationships disturbed by a change;
- curated Evidence survives routine derived-content synchronization.

Code extraction therefore starts outside `@kontext-brain/core`:

```text
LanguageCodeProvider.analyze(file)
  -> CodeResourceSnapshotAdapter.normalize(analysis)
  -> ResourceSnapshot
  -> SyncResourceUseCase.execute(snapshot)
```

The older `graph/EntityExtractor` model is not this integration seam. Code
targets the evidence-backed `knowledge/` model and projects Code Symbols into
its Entities and Facts for retrieval.

ADR 0002 also remains authoritative. PostgreSQL is canonical for structured KG
state and its projections. Git is canonical only for the new managed normative
manifests described in §11; that distinction avoids conflicting sources of
truth.

## 3. Authority, scope, and conflict rules

### 3.1 Source roles

| Source | Role | Can enforce before human acceptance? |
|---|---|---|
| accepted Decision revision | normative choice | yes |
| active Invariant revision | normative condition | yes |
| accepted Domain Term revision | canonical language | yes, in its defined surfaces |
| Task Contract acceptance criterion | intended observable behavior | yes, for that Task |
| code and test result | descriptive implementation evidence | no; may satisfy a verifier |
| Resource, Chunk, Fact, Evidence | grounded descriptive knowledge | no |
| Hypothesis or Normative Proposal | unapproved inference | no |

Mandatory sources are combined, not selected by whichever ranks highest. An
Organization rule cannot be hidden by a Workspace or Personal overlay. If two
applicable mandatory records contradict each other, Kontext Brain returns
`inconclusive`; it does not choose a winner silently.

Domain Term enforcement applies to public types and interfaces, domain entities,
value objects, commands, events, Use Cases, APIs and schemas, Task and normative
language, and user-visible text. It does not rename every local variable merely
because a preferred term exists; local implementation vocabulary is checked only
when it changes externally meaningful domain language.

### 3.2 Governance Scope

Normative records can be accepted at four scopes:

1. **Personal** — private to the user across workspaces;
2. **Workspace** — local to the current checkout or workspace;
3. **Codebase** — canonical for one governed code body;
4. **Organization** — canonical across managed Codebases.

At session start the product detects readable scopes and shows the effective
set. A new local decision defaults to Workspace scope; the user may instead
choose Personal or propose promotion to Codebase or Organization scope. Managed
Organization rules cannot be excluded. Switching the effective scope invalidates
the current Task Context Snapshot and its leases.

Local Acceptance is immediately visible and editable in the user's local
overlay. Promotion occurs through the configured pull-request approval path. A
merged identical revision replaces the local activation with the canonical one;
a changed merge makes the old Task context stale and presents the diff; a
rejected proposal may remain `local_only` but cannot prove Organization-level
completion.

### 3.3 Approval

- Normative Proposals never enforce.
- Codebase and Organization acceptance initially requires the configured
  CODEOWNER approval and merge.
- Medium-risk code completion requires a Code Owner.
- High-risk completion requires both a Code Owner and Domain Owner.
- A coding agent cannot approve a normative change or Review Finding required
  to legitimize its own patch.

## 4. Domain contract

The types below state identity and proof requirements. Storage adapters may add
indexes and timestamps, but may not weaken these contracts.

### 4.1 Code identity

```ts
export type CodeSymbolKind =
  | "module"
  | "class"
  | "interface"
  | "type"
  | "function"
  | "method"
  | "constructor"
  | "getter"
  | "setter"
  | "named_arrow"
  | "field"
  | "constant";

export interface CodeSymbolIdentity {
  readonly codebaseId: string;
  readonly relativePath: string;
  readonly language: string;
  readonly kind: CodeSymbolKind;
  readonly qualifiedName: string;
  readonly signatureDiscriminator: string;
}

export interface CodeSymbolRecord {
  readonly symbolId: string;
  readonly identity: CodeSymbolIdentity;
  readonly behaviorBearing: boolean;
  readonly signature: string;
  readonly contentHash: string;
  readonly semanticSupport: "certified" | "syntactic";
}

export interface PlannedSymbolRecord {
  readonly plannedSymbolId: string;
  readonly taskId: string;
  readonly intendedIdentity: Partial<CodeSymbolIdentity>;
  readonly responsibility: string;
  readonly boundSymbolId?: string;
}
```

`CodeSymbol` is a Kontext domain record, not an AST node and not a compiler's
internal symbol object. Functions, methods, constructors, getters, setters,
named top-level arrows, and Use Case `execute` methods are behavior-bearing by
default. Anonymous callbacks and local functions are initially attributed to
the nearest behavior-bearing parent; providers may expose them separately only
when identity remains stable.

Types, schemas, classes, and constants can be Code Symbols without being one
piece of behavior. A Logic Work Item normally targets one behavior-bearing Code
Symbol, but may group a tightly coupled cycle or a shared schema transaction
that cannot be proven independently.

### 4.2 Normative revisions

Decision, Domain Term, and Invariant content is immutable. Updating one creates
a new revision with `supersedes`; activation pointers select the effective
revision per Governance Scope. Retirement moves the active pointer and never
deletes history.

Every accepted revision records:

- stable record ID and immutable revision ID;
- Governance Scope and activation state;
- statement or canonical definition;
- supporting Evidence IDs and source spans;
- author and approval provenance;
- `supersedes` when applicable;
- provider data-egress classification;
- Invariant verifier references where relevant.

Original Notion, Slack, Markdown, session, issue, or pull-request text remains
in its source Resource. Normative manifests reference Evidence IDs instead of
copying source bodies.

### 4.3 Task and proof records

```ts
export type VerifierKind =
  | "test"
  | "typecheck"
  | "lint"
  | "query"
  | "manual_review";

export interface AcceptanceCriterion {
  readonly criterionId: string;
  readonly statement: string;
  readonly verifier: {
    readonly kind: VerifierKind;
    readonly ref: string;
  };
}

export interface TaskContract {
  readonly taskId: string;
  readonly intent: string;
  readonly acceptance: readonly AcceptanceCriterion[];
  readonly nonGoals: readonly string[];
  readonly targets: readonly string[];
  readonly risk: "low" | "medium" | "high";
}

export interface NormativeRevisionRef {
  readonly kind: "decision" | "domain_term" | "invariant";
  readonly recordId: string;
  readonly revisionId: string;
}

export interface TaskContextSnapshot {
  readonly taskId: string;
  readonly contextDigest: string;
  readonly baseCodeRevision: string;
  readonly effectiveScopes: readonly string[];
  readonly normativeRevisions: readonly NormativeRevisionRef[];
  readonly requiredEvidenceIds: readonly string[];
  readonly sourceFreshnessDigest: string;
  readonly createdAt: string;
}

export interface ContextReceipt {
  readonly receiptId: string;
  readonly taskId: string;
  readonly workItemId: string;
  readonly plannedSymbolIds: readonly string[];
  readonly contextDigest: string;
  readonly normativeRevisions: readonly NormativeRevisionRef[];
  readonly evidenceIds: readonly string[];
  readonly issuedAt: string;
  readonly expiresAt: string;
}
```

A Task Contract is auto-drafted from the user request. Low-risk drafts may be
accepted with a lightweight confirmation; risk and ambiguity increase the
required human review. Exploration may proceed without a complete contract,
but editing cannot be declared done until intent, acceptance, non-goals, and
targets are present and each acceptance criterion maps to a verifier.

```ts
export interface LogicWorkItem {
  readonly workItemId: string;
  readonly taskId: string;
  readonly plannedSymbolIds: readonly string[];
  readonly dependsOn: readonly string[];
  readonly allowedPaths: readonly string[];
  readonly requiredVerifiers: readonly string[];
  readonly capabilityId: string;
}

export interface ChangeBundle {
  readonly workItemId: string;
  readonly baseRevision: string;
  readonly taskContextDigest: string;
  readonly patchDigest: string;
  readonly changedSymbolIds: readonly string[];
  readonly contextReceiptIds: readonly string[];
  readonly evidenceIds: readonly string[];
  readonly normativeRevisions: readonly NormativeRevisionRef[];
  readonly verificationRunIds: readonly string[];
  readonly proposals: readonly string[];
  readonly unresolved: readonly string[];
}

export interface VerificationRun {
  readonly verificationRunId: string;
  readonly verifierKind: VerifierKind;
  readonly verifierRef: string;
  readonly codeRevision: string;
  readonly contextDigest: string;
  readonly result: "passed" | "failed" | "inconclusive";
  readonly observedAt: string;
}
```

The Accuracy Manifest is the immutable completion projection connecting the
Task Contract, Task Context Snapshot, normative revisions, Evidence, Logic Work
Items, Change Bundles, Code Symbols, Verification Runs, and resulting commit.
The pull-request summary links to it; the complete manifest remains queryable in
Kontext Brain.

## 5. Code synchronization and ontology linkage

### 5.1 Code to knowledge mapping

| Kontext concept | Code representation |
|---|---|
| `ResourceSource` | `{ connectorId: "code", externalId: "<codebaseId>:<relativePath>", type: <language module type> }` |
| `Resource` | one source file |
| `Chunk` | one stable semantic region: normally a Code Symbol, plus a module region when needed |
| `CodeSymbolRecord` | exact language-facing identity, signature, hash, and support tier |
| global `Entity` | a cross-Resource symbol with deterministic Codebase identity |
| resource-scoped `Entity` | a symbol intentionally local to one Resource |
| `Fact` | established `calls`, `imports`, `implements`, `extends`, `returns`, `throws`, `reads_env`, or language-specific relation |
| `Evidence` | the Chunk in which the relationship is established |

The stable `sourceChunkId` is derived from kind, qualified name, and signature
discriminator, never a line number or unqualified symbol name. Inserting lines
or formatting another symbol must not re-key unchanged Chunks. Renames create a
new identity unless a deterministic refactor event or curated mapping proves
continuity.

`contentHash` is calculated from a provider-defined normalized syntax form that
preserves semantic literals, decorators, modifiers, and signatures while
ignoring formatting-only differences. Documentation text may be synchronized
as separate Evidence; removing comments from a behavior hash must not silently
discard that documentary Evidence.

### 5.2 Semantic support tiers

Language providers implement one interface and advertise a support tier:

- **certified semantic** — compiler or language-server identity and relationship
  resolution has passed the Kontext conformance suite;
- **syntactic** — Tree-sitter or equivalent parsing can identify regions but
  cannot prove all semantic relationships.

Only certified semantic providers may support negative graph-query Invariants
such as “this module performs no outbound I/O.” Syntactic fallback can retrieve
context and detect touched regions, but a semantic proof request returns
`inconclusive`. Deterministic extraction does not imply perfect knowledge:
dynamic dispatch, generated code, reflection, unresolved modules, and runtime
configuration remain explicit limitations.

Initial certified targets are TypeScript/JavaScript with the TypeScript compiler,
Rust with rust-analyzer, Python with Pyright, and Go with gopls. Tree-sitter is
the cross-language syntactic fallback.

### 5.3 Links to the business Ontology

Context compilation starts from a Task, Planned Symbol, or Code Symbol, lifts to
Ontology Nodes, expands through relevant normative and descriptive relationships,
and grounds back to accessible Evidence.

Code Symbol-Ontology Links retain origin:

- `curated` — accepted by an authorized person;
- `deterministic` — derived from an approved mapping or structured identifier;
- `proposed` — suggested by AI or similarity.

A proposed link may improve optional retrieval ranking but cannot bind an
Invariant or enforce a Domain Term by itself.

## 6. Context compilation

### 6.1 Preflight

At Task start the local sidecar incrementally synchronizes required local and
connected sources. Preflight classifies the effective context:

- complete, current mandatory context permits editing;
- stale, conflicting, inaccessible, or unavailable mandatory context permits
  exploration but blocks ordinary editing or completion as specified below;
- missing or conflicting proposal-only context warns but does not block editing.

An inaccessible required record fails closed without revealing its title,
content, or existence beyond the minimum generic error allowed by policy.

### 6.2 Deterministic mandatory context

The Context Compiler includes all applicable active Decisions, Invariants,
Domain Terms, Task acceptance criteria, and conflict markers losslessly and in
a deterministic order. Token budgets never trim mandatory rules.

Bounded best-first Lift/Expand/Ground retrieval from ADR 0003 selects supporting
and neighboring Evidence. An LLM may summarize only optional Evidence and must
retain Evidence IDs and source spans. Enforcement always reads canonical
records, never the summary.

### 6.3 Evidence is data, not instruction

Text from Slack, Notion, Markdown, issues, code comments, or prior sessions is
untrusted Evidence. Instructions embedded in that text cannot alter tool
authority, ignore rules, approve a proposal, or expand scope. The compiler
passes it in typed evidence fields and quoted excerpts. Prompt-injection
detection may add warnings but is not the security boundary.

### 6.4 ACL and provider egress

Every retrieval uses the current session principal. Evidence, normative records,
and Tasks carry data classification and allowed runtime/provider policy. A Work
Item is scheduled only to a runtime allowed to receive all mandatory context.
If redaction would remove mandatory context, the scheduler chooses another
eligible runtime or blocks; it never sends an incomplete normative slice and
pretends it is complete.

The audit log records provider, record and Evidence IDs, revisions, digest, and
time. It does not duplicate secret source bodies or provider credentials.

## 7. Task and Logic Work Item lifecycle

1. **Contract** — draft and confirm intent, acceptance, non-goals, targets, and
   risk.
2. **Synchronize** — refresh required sources and current code index.
3. **Snapshot** — freeze the effective normative revisions and Evidence
   requirements in a Task Context Snapshot.
4. **Plan** — create Planned Symbols, lift them to Ontology anchors, and build a
   dependency DAG of Logic Work Items.
5. **Lease** — grant each Work Item a least-privilege capability over one
   Knowledge Space, Task, evidence scope, paths or symbols, verifiers, and
   expiry.
6. **Begin logic** — compile symbol-level just-in-time context and issue a
   Context Receipt before editing.
7. **Edit** — a worker modifies only its isolated worktree and leased scope.
8. **Resynchronize** — bind Planned Symbols to actual Code Symbols and compute
   affected Facts, symbols, Invariants, and dependants.
9. **Verify** — run the fast and targeted tiers for the same revision and
   context digest.
10. **Handoff** — return a Change Bundle; conversation prose is not the handoff
    contract.
11. **Integrate** — the main thread validates receipts, applies bundles in
    dependency order, resolves semantic conflicts, and runs integrated checks.
12. **Complete** — create the Accuracy Manifest only when every required proof
    is current and passing.

If a new normative revision is accepted after snapshot creation, the Task is
marked stale. The user or orchestrator must explicitly refresh, show the
effective diff, invalidate obsolete leases and Verification Runs, and revalidate
affected work. It must not silently inject new rules into a running Task.

## 8. MCP, hooks, and enforcement

The tool server exposes provider-neutral operations. Exact transport names may
evolve, but their contracts are stable:

| Operation | Contract |
|---|---|
| `kontext_prepare_task` | validate Task Contract, synchronize sources, create snapshot and plan seeds |
| `kontext_begin_logic` | acquire capability and return symbol context plus Context Receipt |
| `kontext_authorize_write` | provider hook revalidates a receipt and exact write paths |
| `kontext_refresh_task_context` | show revision diff and create a replacement snapshot |
| `kontext_check_change` | resynchronize touched code and execute required symbol checks |
| `kontext_submit_change_bundle` | validate the worker handoff against lease, revision, and digest |
| `kontext_propose_transition` | accept Evidence and compute Task state; reject direct state writes |
| `kontext_inspect_runtimes` | report CLI installation, auth, billing path, and scheduling eligibility |
| `kontext_schedule_logic` | durably enqueue sidecar-planned Work Items for isolated provider-bound execution |
| `kontext_get_schedule` | read queued, running, cancelling, completed, failed, interrupted, or cancelled state and terminal proof |
| `kontext_cancel_schedule` | durably request cancellation and report state while the owner stops workers and releases leases |

Codex and Claude adapters install their native plugin, MCP, and hook bindings.
`AGENTS.md` and `CLAUDE.md` contain only the minimal bootstrap rule to use these
operations; they do not duplicate Decisions or Domain Terms.

Codex pre-tool hooks guard `apply_patch`; Claude pre-tool hooks guard `Write` and
`Edit`. Both require an active Work Item capability and Context Receipt for
known writes. Because shell commands and external processes may bypass a
provider's high-level edit tool, matching post-tool hooks and a two-second local
sidecar observer compare actual workspace content with the receipt baseline.
Any unreceipted or out-of-scope change is quarantined immediately after
detection. Post-write blocking cannot undo a completed write. A runtime that
cannot provide reliable pre-write enforcement or authoritative post-write
observation is restricted to exploration and review.

### 8.1 Verification tiers

| Tier | When | Required checks |
|---|---|---|
| fast | after each affected behavior-bearing Code Symbol | parse/semantic sync, stable identity, Domain Term and graph-query checks |
| targeted | at Logic Work Item checkpoint | relevant package typecheck, tests, lint, and bound Invariants |
| full | before Task completion | integrated test and build suites, all acceptance verifiers, Invariants, reviews, manifest audit |

After an edit, affected symbols remain `verifying`, `violating`, or
`inconclusive` until the required checks settle. A violating Work Item may edit
its own scope to fix the problem and run tests, but cannot start unrelated new
logic or claim completion.

Verifier infrastructure failure produces `inconclusive`, not pass or fail. The
current Task may continue corrective edits inside existing leases, but cannot
start unrelated Tasks or reach done. A durable queue in the local sidecar retries
the same revision and context digest when infrastructure returns; a newer edit
supersedes the obsolete run. Revalidation therefore happens automatically after
recovery.

An emergency bypass requires explicit authorized approval, expiry, reason, and
an Accuracy Manifest audit entry. It cannot forge a passing Verification Run.

## 9. Codex-led multi-agent orchestration

### 9.1 Responsibilities

The main Codex thread is orchestration-only:

- owns the Task Contract and Task Context Snapshot;
- creates the Planned Symbol and Logic Work Item dependency graph;
- schedules workers and maintains leases;
- validates and semantically integrates Change Bundles;
- requests approvals and independent reviews;
- runs full verification and issues the final verdict.

It does not write feature logic. Mechanical branch, state, manifest, and merge
operations are allowed because they do not bypass worker proof boundaries.

Workers:

- run in isolated Git worktrees and branches;
- receive only their capability, context slice, and dependency artefacts;
- consult Kontext Brain for every behavior-bearing Code Symbol they implement;
- submit Change Bundles rather than merging directly;
- cannot broaden their own paths, Evidence, verifier set, or expiry.

### 9.2 Decomposition and merge

The scheduler builds a Code Symbol dependency DAG. Strongly connected cycles and
shared schema or transaction changes are grouped. Work touching the same symbol,
file transaction, migration, or shared state is serialized unless independence
is proven.

Two bundles changing the same Code Symbol never auto-merge. Shared state,
schema, API, and Invariant meaning conflicts are semantic conflicts even when
Git has no textual conflict. Deterministic generated-file conflicts may be
reproduced automatically; other conflicts create an Integration Work Item with
a fresh Context Receipt and complete revalidation.

### 9.3 Concurrency and retry

Default global concurrency is four Logic Work Items across all runtimes. CPU,
memory, provider quota, data-egress eligibility, and conflict topology may
reduce it. A lower limit queues work and never skips context or verification.

Retry belongs to a Work Item checkpoint, not an agent conversation. The same
failure cause is retried at most twice before escalation to the main thread or
user. When a runtime is unavailable, queued work may move to another eligible,
authenticated subscription runtime. In-progress transfer requires a durable
checkpoint, confirmed process termination, and released write lease before a
fresh provider session starts.

### 9.4 Risk-based cross-runtime review

- **Low risk** — one eligible runtime implements; deterministic verification is
  mandatory.
- **Medium risk** — one runtime implements and the other independently reviews.
- **High risk** — planning and implementation use different runtimes and the
  non-implementing runtime performs final cross-review.

Users may pin a runtime. `auto` uses project benchmark results, provider
capabilities, eligibility, and available subscription capacity rather than a
hard-coded belief that one model is always better at a role.

Independent reviewers receive the Task Context Snapshot, diff, acceptance
criteria, normative revisions, and verifier output, not the implementer's full
conversation. Implementer explanations are labeled as unverified claims rather
than Facts. Reviewers return Code-Symbol-scoped Review Findings with rule and
Evidence references, and an implementer cannot close its own finding.

### 9.5 User visibility

The product shows a live Task graph with each Logic Work Item's runtime, status,
target symbols, applicable Decisions and Invariants, Evidence references,
verification state, unresolved Review Findings, and merge readiness. Raw agent
and tool logs remain expandable rather than dominating the default view.

Routine progress does not interrupt the user. Approval is requested for material
Task Contract changes, new or changed normative acceptance, scope or data-egress
expansion, high-risk completion, and audited bypasses.

## 10. Agent runtimes, subscriptions, and compatibility

```ts
export interface AgentRuntimePort {
  inspectCapabilities(): Promise<RuntimeCapabilitySnapshot>;
  start(input: RuntimeWorkInput): Promise<RuntimeSession>;
  resume(sessionId: string, input: RuntimeWorkInput): Promise<RuntimeSession>;
  terminate(sessionId: string): Promise<void>;
}
```

First-class adapters are:

- `CodexRuntimeAdapter` for open-source Codex CLI, SDK, and App Server surfaces;
- `ClaudeCodeRuntimeAdapter` for the authenticated Claude Code CLI;
- optional `OrcaRuntimeAdapter` for interoperability when ORCA is installed.

ORCA is not a core dependency. The product internalizes the useful pattern of
driving official logged-in CLIs while retaining its own context, proof, and
scheduling contracts.

Each CLI owns its credentials. Kontext Brain stores only health, billing-path,
capability, model, version, and eligibility metadata. The default sidecar passes
only an operating-system/configuration allowlist plus `KONTEXT_PLUGIN_DATA` to
workers; it does not forward API-key variables. `RuntimeDoctor` checks without
performing a billed inference call:

- installation and authentication;
- subscription versus usage-billed API path;
- required structured output and workspace-isolation capabilities reported by
  the adapter;
- CLI version and diagnostics.

The first scheduled invocation is the live structured-protocol check. A failed
call creates a failed attempt and checkpoint rather than upgrading eligibility.
Future compatibility certification may add an explicit opt-in smoke call to the
doctor.

An environment variable that would select API billing is surfaced before work
and blocked until the user explicitly chooses it. The scheduler never silently
falls back from subscription use to paid API use.

Runtime sessions are backend-bound. Switching a role from Codex to Claude or
back creates a fresh provider session seeded from the same Logic Work Item,
Task Context Snapshot, artefacts, and checkpoint. Private conversation state is
not translated or impersonated across providers.

Capabilities are discovered at session start and frozen in a
`RuntimeCapabilitySnapshot`; model names and CLI behavior are not hard-coded.
An unknown version may run required smoke tests. If a mandatory feature cannot
be proven, that runtime is limited to exploration or review. Changing CLI
version during a Task stales its runtime lease and triggers reinspection.

## 11. Local, connected, and managed topology

### 11.1 Local sidecar

The cross-platform local sidecar owns:

- local code, Git, Markdown, and Codex-session synchronization;
- Personal and Workspace Local Acceptance overlays in plugin `PLUGIN_DATA`;
- Task Context Snapshots, capabilities, worktree leases, and provider sessions;
- durable Verification Run and recovery queues;
- code filesystem observation and language-provider processes;
- export of local normative proposals to pull requests.

Private overlays and snapshots are keyed by Organization, Codebase, and
Workspace and are not committed to the target repository. Only explicit
proposal manifests are exported.

### 11.2 Organization service

The optional Organization service owns:

- Notion, Slack, GitHub, and other connected-source synchronization;
- Organization-canonical governance manifests and approval projection;
- pull-request checks, merge webhooks, and periodic active-pointer refresh;
- PostgreSQL and object-storage adapters from ADR 0002;
- row-level access control, provider egress policy, and audit history.

Git manifests are canonical for accepted managed Decision, Domain Term, and
Invariant revisions. Organization-wide manifests live in a central governance
repository under `.kontext`; Codebase rules live in the governed repository's
`.kontext`. PostgreSQL is a query and enforcement projection of those manifests,
while the Evidence KG retains source provenance.

`CONTEXT.md` remains a glossary and becomes a generated human- and agent-readable
projection of accepted Domain Terms. Migration imports the current file once as
curated Domain Term revisions with file Evidence; subsequent edits go through
the normative proposal and approval path rather than creating an independent
source of truth.

### 11.3 Freshness and offline behavior

Canonical pointer updates arrive through merge webhooks, session start or
resume checks, and a five-minute fallback poll. Any pointer change marks
dependent snapshots stale.

Personal mode works fully offline. Managed Codebase editing may continue only
with an unexpired signed Organization snapshot lease. It cannot claim managed
completion or merge until online revalidation. An expired lease restricts a
managed Codebase to exploration. When infrastructure returns, queued
Verification Runs and snapshot validation resume automatically against the same
revision and digest unless superseded.

## 12. Package boundaries

| Package | Responsibility |
|---|---|
| `@kontext-brain/code` | provider-neutral Code Symbol records, LanguageCodeProvider port, ResourceSnapshot adapter, TypeScript provider first |
| `@kontext-brain/spec` | Decision, Domain Term, Invariant, Task, revision, activation, and pure transition contracts; no I/O |
| `@kontext-brain/context` | context compilation, Task Context Snapshot, Context Receipt, freshness and digest logic |
| `@kontext-brain/orchestrator` | Logic Work Item DAG, capabilities, scheduling, Change Bundle and Accuracy Manifest validation |
| `@kontext-brain/runtime-codex` | Codex runtime, hook, plugin, and structured-event adapter |
| `@kontext-brain/runtime-claude` | Claude Code runtime, hook, and structured-event adapter |
| `@kontext-brain/local` | sidecar process, local stores, worktrees, filesystem observation, retry queue |
| `@kontext-brain/tool-server` | provider-neutral MCP operations from §8 |
| `@kontext-brain/postgres` | normative projections, Tasks, manifests, audits, and existing KG adapters |

Core domain packages depend on ports, not provider CLIs, PostgreSQL, GitHub, or
filesystem implementations. The initial Code adapter uses the existing
`ResourceSnapshot` seam without modifying knowledge synchronization semantics.

## 13. Implementation sequence and gates

This is sequencing for the complete product, not an MVP scope cut. A failed gate
pauses the dependent phase for correction; it does not silently remove required
product behavior.

### Phase 0 — contract and benchmark foundation

- accept ADR 0005 through ADR 0009;
- update the Domain Language and this design note;
- define serialization schemas for the §4 contracts;
- create an end-to-end Accurate Code benchmark manifest.

Gate: documents, schemas, and fixture terminology agree; every benchmark Task
has intent, acceptance, non-goals, targets, gold normative context, and expected
verifiers.

### Phase 1 — TypeScript Code intelligence

- implement `@kontext-brain/code` and TypeScript semantic provider;
- synchronize stable Code Symbols, Entities, Facts, and Evidence;
- implement affected-symbol and reverse-dependency lookup;
- benchmark identity stability and extraction precision.

Gate: resynchronizing an unchanged tree is a no-op for more than 95% of Chunks,
format-only edits do not stale behavior, and labelled relationships meet the
precision threshold required by their intended verifier use.

### Phase 2 — normative and Task records

- implement immutable revisions, activation pointers, scopes, proposal and
  approval transitions;
- implement Task Contract and evidence-derived state transitions;
- add Git manifest codecs, local overlay adapter, and PostgreSQL projection;
- import and project Domain Terms to `CONTEXT.md`.

Gate: proposals cannot enforce, superseded revisions remain auditable, scope
conflicts fail closed, and direct agent state writes are rejected by tests.

### Phase 3 — context and single-runtime vertical slice

- implement Task Context Snapshot, Context Compiler, Context Receipt, and MCP
  contracts;
- connect Codex through plugin/MCP/hooks and the local sidecar;
- enforce one Context Receipt per behavior-bearing logic unit;
- capture exact context and token ablations in the benchmark.

Gate: Kontext context improves Accurate Code Rate against vanilla Codex at equal
or acceptable token cost, mandatory context is never trimmed, and stale or ACL-
inaccessible mandatory context cannot reach done.

### Phase 4 — Invariant and completion enforcement

- implement verifier registry, query verifiers, three verification tiers,
  quarantine, durable retry, Change Bundle, and Accuracy Manifest;
- add reverse impact and Drift Finding creation;
- integrate pull-request checks and manifest summaries.

Gate: false rejection is below the release threshold, every done transition has
same-revision and same-context proof, and recovery revalidates without accepting
superseded runs.

### Phase 5 — dual runtime and multi-agent integration

- implement Codex and Claude adapters plus RuntimeDoctor;
- implement durable asynchronous Work Item DAG scheduling, worktrees,
  capabilities, semantic merge, checkpoint transfer, and blind cross-runtime
  review;
- enforce default concurrency four and risk-based provider policy.

Gate: provider switching never resumes the wrong conversation, no concurrent
writer exceeds its lease, billing-path changes require consent, and integrated
accuracy is at least as high as the best eligible single-runtime path.

### Phase 6 — complete personal and Organization topology

- support personal-only, Organization-only, and mixed sources;
- add managed canonical repositories, CODEOWNER promotion, webhooks, ACL and
  provider-egress policies, offline leases, and reconnect revalidation;
- certify macOS, Linux, and Windows local behavior.

Gate: personal use requires no external connector, managed rules cannot be
weakened locally, offline work cannot falsely claim managed completion, and ACL
leak tests remain zero.

### Phase 7 — language and platform certification

- certify Rust, Python, and Go semantic providers;
- ship Tree-sitter syntactic fallback;
- support Codex CLI, TypeScript and Python SDK integrations, and App Server;
- test IDE and cloud compatibility only through public integration surfaces.

Gate: unsupported semantic proofs return inconclusive, not false pass; certified
provider and OS accuracy deltas remain within release thresholds.

## 14. Production acceptance

The release benchmark contains at least 100 representative Tasks across local,
Organization, mixed, offline, recovery, stale, conflicting, and ACL-restricted
scenarios. The production gate is:

- end-to-end Accurate Code Rate at least 15 percentage points above vanilla
  Codex OSS using the same model, Task, and sandbox;
- the 95% confidence-interval lower bound for that improvement is above zero;
- active Decision violations: zero;
- stale-context completions: zero;
- ACL or provider-egress leaks: zero;
- false blocking rate: below 1%;
- Evidence citation precision: at least 95%;
- context tokens: no more than baseline plus 20%, unless accuracy per token is
  demonstrably better;
- supported-OS accuracy delta: at most 2 percentage points.

Retrieval recall, model agreement, number of agents, and graph size are diagnostic
metrics. They are not substitutes for Accurate Code Rate.

## 15. Required acceptance scenarios

The implementation and benchmark must exercise at least these boundary cases:

1. A person with only local Markdown, Git, code, and prior sessions accepts a
   Workspace Decision and later promotes it through a pull request.
2. A new Organization Decision conflicts with a local overlay while a Task is
   running; the snapshot becomes stale and completion blocks until explicit
   refresh.
3. Claude reaches a subscription limit midway through a Work Item; its process
   terminates, the lease is released, and Codex resumes from a checkpoint in a
   fresh session without paid API fallback.
4. Two workers plan to edit the same method or shared schema; the scheduler
   serializes or groups them rather than textually auto-merging.
5. A Slack message contains malicious tool instructions; it remains quoted
   Evidence and cannot expand authority.
6. A syntactic-only language provider observes a dynamic call; a negative query
   Invariant returns inconclusive instead of guarded.
7. Organization infrastructure fails after a verifier is queued; the same
   revision and digest are revalidated when service returns, while a newer run
   supersedes the old one.
8. A reviewer agrees with the implementer but a deterministic test fails; the
   Task remains incomplete.
9. A Domain Term changes after code was previously verified; reverse lookup
   creates Drift Findings and migration Tasks without automatically editing code.
10. Required Evidence exists but the session principal cannot access it; the
    system fails closed without leaking the hidden source.
