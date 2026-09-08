# Goal-to-reviewed-plan workflow

The host sidecar exposes five planning management tools: `kontext_start_plan`,
`kontext_refine_plan`, `kontext_inspect_plan`, `kontext_cancel_plan`, and `kontext_approve_plan`. Ordinary
worker sidecars advertise none of them. The private management token is removed
from the environment before constructing planners or spawning subprocesses.

`LocalTaskPlanningOperations` reserves an owner-scoped request UUID durably before
calling a provider. Identical requests retrieve that reservation; changed input
with the same UUID conflicts. The background planner uses the existing Codex or
Claude runtime adapter's new optional `plan` operation, not another model SDK or
terminal launcher. Missing planning support is an explicit refusal. Verified
subscription authentication is required; API/unknown billing is refused.

The host resolves a committed Git workspace or captures a
[private working-file baseline](./coding-workspace-seeds.md), recaptures selected registered
Resources, collects actual source-owned Evidence and its provenance, and loads
effective local normative rules. Required unavailable/conflicting/unshared context
prevents dispatch. Code and context are rechecked after capability inspection and
after the provider result. This reuses the same preparation module as Task creation.

Codex planning uses its existing `exec --json --sandbox read-only` path. Claude
uses print JSON with `--permission-mode plan`; the implementation plugin argument
is not attached. Neither receives a fabricated Logic Work Item, implementation
capability or worker lease. CLI permission modes and instructions are **not** a
new OS security boundary: inherited provider configuration and filesystem access
remain governed by that CLI. The host filters the context it transmits; it cannot
recall already transmitted context or atomically prevent every concurrent source
change at the instant of a network transmission.

The response must be a bounded JSON proposal containing a Task Contract and
behavior-bearing Planned Symbols. Validation rejects missing symbol descriptions,
duplicate ownership, path traversal/globs, unknown/cyclic dependencies, asserted
symbol bindings, and model-supplied capabilities. It does not prove that the
proposed implementation or verifier commands are correct or that tests passed.

The result is a private, integrity-checked `review` record, not an approved Task.
Approval names the exact proposal digest. It calls the existing Task creation
operation with the reviewed Git revision and source-freshness digest; changed code,
rules or grants refuse creation. Approval is replay-safe even if initial Task
registration succeeded but recording the approval response did not. Creation
returns `stored_context` inspection; normal workbench inspection revalidates it.

## Recovery and UI

Kondex sends all five RPCs to the selected owning runtime. Direct SSH/WSL paths
without a host-side integration are refused, never redirected to the client.
Methods are additive JSON RPCs; older hosts fail explicitly without local fallback.
The GUI persists owner, original request and UUID before dispatch; a lost reply can
be inspected or explicitly recovered with the same input. Re-pairing invalidates
the client authority check. Read-only status polling never starts/resumes a model.

If a planning record has no active owner in this sidecar instance, inspection says
`unverifiable`, never that a process died. No automatic redispatch occurs after a
restart. Cancellation only targets an in-memory owned execution; uncertain provider
errors remain `unverifiable`. A complete proposal arriving after cancellation is
not offered for approval. This is not cross-process planner reattachment.

The GUI reviews intent, acceptance/verifier references, non-goals, targets, risk,
code/context identity, Evidence IDs, each symbol's responsibility/identity, allowed
paths and dependencies. Approval and implementation use separate consent controls.
Approval populates the existing Task workbench; the user then starts its existing
WorkItemScheduler flow. Plans can be regenerated as new requests. The review UI
also connects the draft-refinement operation below through `kontext.refinePlan`.
Its expandable feedback form requires separate subscription consent; changing the
feedback clears consent. Parent UUID/digest and feedback remain visible on the
new draft. The parent is retained and approval of the new draft requires its own
digest. The journal saves a distinct refinement recovery payload before dispatch;
lost replies are recovered with the same UUID and exact feedback, never ordinary
planning. Model-generated clarification questions and
in-place approved-contract amendments remain separate unfinished work.

## Feedback on an unapproved draft

`kontext_refine_plan` accepts a new `requestId`, a `parentRequestId`, the exact
`expectedParentDigest`, and bounded user `feedback`. It is host-only and can consume
the selected CLI subscription, so it requires an explicit user action, not a read
or automatic poll. Workspace, goal, sources and provider are derived from the
owner's stored parent, not supplied as overrides by the caller. It reuses the
existing reservation, planning adapter, cancellation and exact-approval operations.

The parent must be an unapproved review draft. Before transmitting any prior
proposal, the host recaptures current code and sources, verifies subscription
eligibility, rechecks sharing after capability inspection, and requires the same
code/context basis as that parent. A changed basis requires a fresh plan: this
route must not retransmit potentially revoked information from the old proposal.
The prompt marks prior model output as untrusted and includes only the exact
parent proposal and new feedback, not an unbounded conversation history.
Existing prompt/output limits apply to the whole refinement too.

Each refinement has a new planning identity and a digest-bound parent reference;
the parent is not mutated, superseded or approved. It is an independent draft,
not an amendment to an approved Task. The final parent check and child publication
hold the parent's existing mutation lock, serializing them with parent approval;
an already approved parent prevents the late refinement from being offered for review. New approval must name the
child's exact digest, with the usual current-context checks, and still does not
start implementation. Refining a previously approved Task is refused.

The same refinement UUID and input retrieve the existing record across concurrent
calls and restarts without another model dispatch, even if the parent was approved
after the child had finished. Different feedback under the same UUID conflicts.
Uncertain executions retain the existing `unverifiable` behavior; inspection or
replay cannot resume a process. The new optional record metadata preserves reading
old records with the new host; an older host cannot parse new refinement records
under its strict stored schema. No downgrade compatibility is claimed for them.

The additive tool name lets old sidecars refuse with method-not-found rather than
silently dropping feedback and dispatching an ordinary plan. Ordinary worker
sidecars do not advertise it; an incorrect host token is rejected before mutation.

## Verification and remaining work

Tests cover real temporary Git/SQLite preparation, concurrent replay, source
revocation during generation and after review, cancellation, corrupt records,
subscription-only gating and existing adapter output parsers. Hidden Electron tests
exercise both Codex and Claude through the generated MCP bundle, using deterministic
offline fake CLIs. They require real source provenance in the prompt, reload the
GUI, approve a Task and confirm implementation remains disabled without consent.
They do not use real subscriptions, paid APIs, user profiles or foreground windows.

This is a usable initial coordinator planning/approval path, not proof of model
coding quality or of the entire requested product. Registered Markdown and native
journal selection, completion assessment and explicit finalization now have separate
GUI paths. Managed connectors, task-level egress, approved-contract amendments,
model-led clarification, approved-Task replanning,
trusted owner authority and installed-release validation remain.

Refinement verification (852–858): nine focused cases failed before implementation;
the final planning/creation/real-MCP authorization regression passes 32 tests. Tests
cover independent exact approval, concurrent/restarted replay, wrong parent/digest,
request collisions and overrides, changed code/source/sharing, revocation during
capability inspection, and parent approval during generation. Two additional red
cases (857) showed that both ordinary and refined planning previously transmitted
after code changed during capability inspection; the shared pre-dispatch code
recheck now prevents those calls (858). Production and strict selected-test typing
pass. No real model or subscription was invoked. The canonical backend MCP bundle is rebuilt.

Kondex integration verification (863–880): focused renderer/RPC/journal tests pass
22 cases; the combined local MCP and DNS regression passes 32. Hidden Electron
tests pass eight Codex/Claude × committed/dirty/unborn/folder combinations, using
offline CLI fixtures. Folder cases pass exact-parent refinement, two preserved
draft entries, new digest approval and reload with no third planning invocation.
Original coding files stay unchanged. Full typing and production builds pass.
Kondex's resource copy now matches the canonical bundle SHA-256
`37ee197a11247cf3616503ffceeadc27555b6682d26cbb3e1277f35aab6aed19`.
Temporary package 847 remains older; no installed application was upgraded.
This does not certify live provider behavior or the complete release gates.
