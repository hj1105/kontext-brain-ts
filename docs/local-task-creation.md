# Host-reviewed Task creation

`kontext_create_task` is an executable host-only management tool. Its caller must
already have a user-reviewed Task Contract and behavior-level work plan. It is not
an automatic planner, does not grant normative/source permissions, does not start
models or workers, and never marks a Task complete.

The host derives Task identity from the private principal and request UUID. The
request fingerprint covers the complete submitted plan and source selection.
An exact retry returns the existing Task, even after later context updates or a
source outage; a changed request using that UUID conflicts. The creation response's
inspection is explicitly `stored_context`, not proof of current source freshness.
The workbench's normal inspection and the execution path revalidate live context.

Creation verifies the canonical source workspace and reviewed code revision before
and after context assembly. Uncommitted, unborn and non-Git coding workspaces now
use a host-owned [private baseline](./coding-workspace-seeds.md); clean committed
Git roots retain the existing path. Source Markdown may also live in a non-Git folder. Paths
with spaces, including trailing spaces on supporting platforms, are preserved.
Git inspection is bounded, noninteractive and disables fsmonitor hooks; the private
host-management token is excluded from subprocess environments.

Each selected Resource is explicitly recaptured using its saved locator. Actual
source-owned Evidence and provider policy pass through the existing collector and
assembler. The local normative overlay is loaded using the host principal and
workspace ID; request payloads cannot supply approved normative JSON, principal
IDs, source text or source grants. Missing normative Evidence remains blocked.
The reviewed plan's Planned Symbols receive the host-generated Task ID.

The final publication is a single initial Task envelope containing ownership,
source selection, current state, contract and frozen snapshot. See
[Task context publication](./task-context-publication.md). It becomes usable by
the existing inspection/compiler/work-item workflow without a separate import.

## Revalidation and authority

The executable sidecar uses `RegisteredTaskContextProvider` for new registered
Tasks in preparation, logic entry, write checks, scheduling, completion and
verification recovery. It re-reads the trusted local overlay and recaptures only
the source selection recorded by the host. Revocation, source changes and missing
files invalidate editing. Source failures retain known Evidence IDs as unavailable
without reusing their old bodies. No unknown Evidence IDs are invented.

The current view does not silently advance the Task Snapshot. Explicit refresh
rebuilds the snapshot with all current required source Evidence, including newly
added sections. Source permission still needs explicit user approval for changed
content. A frozen snapshot refresh cannot grant access. Already transmitted model
context cannot be recalled; this checks subsequent operations, not retroactive use.

Worker `kontext_prepare_task` cannot replace a registered Task's reviewed contract.
Legacy Task workflows retain their existing behavior. The eventual host-approved
contract amendment route must coordinate atomically with worker preparation;
contract amendment is not yet exposed as a finished user workflow.

## Verification and remaining product work

Tests use actual temporary Git repositories, SQLite source storage, private local
overlays and executable MCP processes. They exercise idempotent creation, concurrent
requests, existing-Task preservation, original provenance, accepted local rules,
permission revocation before another logic start, changed source sections, explicit
snapshot refresh, wrong host capability and unauthorized contract replacement.
They do not invoke a model or bill an API.

Kondex now offers a goal-to-plan GUI and exact-digest approval RPC through the
[host planning lifecycle](./local-task-planning.md). It reuses this creation
operation and supplies `expectedSourceFreshnessDigest`, so approval cannot silently
adopt context that changed since planning. The raw `kontext_create_task` operation
still accepts already-reviewed plans from trusted host callers.

Managed context selection, task-level
egress and contract-amendment approval, final completion GUI and production
packaging remain required. This is not the complete requested product.
