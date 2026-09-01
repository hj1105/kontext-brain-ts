# Kontext Brain agent plugin

This development plugin gives Codex CLI and Claude Code the same provenance-backed context workflow for each behavior-bearing logic unit. The main orchestration path uses official logged-in CLIs; it does not fork either agent runtime or copy provider credentials.

## What is authoritative

- Collected source material remains Evidence with its Evidence ID and source span.
- Accepted Decision, Domain Term, and Invariant revisions are the normative layer.
- `CONTEXT.md` is a generated glossary projection of accepted Domain Terms, not a second source of truth.
- Private Personal and Workspace acceptance, current Task state, prepared snapshots, and short-lived write capabilities live under Codex `PLUGIN_DATA` with owner-only file permissions.
- Verification Runs, retry jobs, Quarantine Records, accepted Change Bundles, and the final Accuracy Manifest are also private, digest-checked sidecar state.

## Build and validate

From the repository root:

```sh
pnpm -r build
pnpm --filter @kontext-brain/tool-server build:codex-plugin
python3 /path/to/plugin-creator/scripts/validate_plugin.py plugins/kontext-brain
claude plugin validate plugins/kontext-brain
```

The plugin contains a self-contained `server.mjs`. Codex starts it through `.mcp.json` and `.codex-plugin/plugin.json`; Claude Code uses `.claude.mcp.json` and `.claude-plugin/plugin.json`. No fork is required. Review the provider-specific hook file before enabling it.

## Publish current sidecar state

The collector or sidecar exports an assembly JSON containing the current Git revision, Personal and managed normative manifests, provenance-bearing Evidence, and Logic Work Item plans. See `examples/task-state-assembly.example.json`.

For source development, set the same private directory the plugin will use and publish the assembly:

```sh
KONTEXT_PLUGIN_DATA=/private/path/kontext-brain-data \
  pnpm --filter @kontext-brain/local exec kontext-sidecar \
  publish-task-state plugins/kontext-brain/examples/task-state-assembly.example.json
```

When installed as an Agent Plugin, Codex injects `PLUGIN_DATA` into the MCP server and command hooks. The implementation also supports `KONTEXT_PLUGIN_DATA`, `CLAUDE_PLUGIN_DATA`, Windows application data, XDG data, and a user-local fallback for compatible hosts.

The assembler:

- reconciles Local Acceptance with managed canonical revisions;
- keeps changed local/managed collisions as explicit conflicts;
- computes freshness from manifest and Evidence content digests;
- intersects Evidence and normative provider-egress policies;
- materializes a missing normative source as unavailable Evidence instead of guessing it;
- rejects absolute or workspace-escaping allowed paths.

## Codex workflow

1. Call `kontext_prepare_task` once with the Task Contract.
2. For each behavior-bearing symbol, call `kontext_begin_logic` with the exact Work Item and Planned Symbol IDs.
3. Edit only when the result is current and contains a Context Receipt.
4. After each affected behavior-bearing symbol, call `kontext_check_change` with `workspacePath` and the `fast` tier; at the Logic Work Item checkpoint call it with `targeted`. The sidecar derives the revision and changed symbols.
5. Submit `workspacePath` plus an ID-free bundle draft with `kontext_submit_change_bundle`; the sidecar independently derives the patch digest, changed paths and behavior-bearing Code Symbols, Planned Symbol bindings, and receipt, then issues the immutable Bundle ID only when the worker claims match current proof and quarantine state.
6. After integration, call `kontext_check_change` with `full`, then call `kontext_propose_transition` with Evidence and evaluations. The sidecar creates the Accuracy Manifest and computes state; callers cannot write `done` or inject a manifest.
7. Use `kontext_refresh_task_context` after a reported revision or Evidence change.

The server exposes eleven operations: the seven Task/write/completion operations above plus `kontext_inspect_runtimes`, `kontext_schedule_logic`, `kontext_get_schedule`, and `kontext_cancel_schedule`. Inspect runtimes before scheduling. Scheduling first persists a private job and returns immediately; poll `kontext_get_schedule` until it reports `completed`, `failed`, `interrupted`, or `cancelled`. Enqueue acceptance is not completion. To stop work, call `kontext_cancel_schedule` and keep polling while it is `cancelling`; `cancelled` is recorded only after active CLI processes stop and write leases release. The scheduler accepts only sidecar-planned Work Item IDs, intersects requested providers with Evidence egress policy, creates deterministic isolated Git worktrees, acquires persisted scope leases, and issues a provider-bound Context Receipt before each attempt. Concurrency is capped at four and retries at two.

Codex is eligible only when `codex login status` proves ChatGPT subscription authentication. Claude is eligible only when `claude auth status --json` proves a non-API login. Usage-billed API credentials are not forwarded by the bundled sidecar and an API or unknown billing path is not eligible. Provider switching starts a fresh conversation from the same immutable attempt checkpoint; a Claude conversation is never resumed as Codex or vice versa.

## Verifier configuration

The standard refs `workspace:typecheck`, `workspace:test`, `workspace:build`, and `workspace:lint` execute the matching root `package.json` script. Other verifier refs must be declared in `.kontext/verifiers.json`; see `examples/verifiers.example.json`. Commands are executed directly with an argument array (`shell: false`), from the receipt-bound workspace, with bounded time and output. A missing or malformed definition is `inconclusive`, never pass.

## Hook and observation boundary

The Codex `PreToolUse` hook reloads private sidecar state for each `apply_patch`; the Claude hook does the same for `Write` and `Edit`. It allows only exact receipt-bound paths. A missing, expired, stale, conflicting, inaccessible, unavailable, tampered, or out-of-scope capability is denied. Matching PostToolUse hooks reconcile the actual workspace content with the pre-write event. The long-running MCP sidecar also polls bound workspaces every two seconds, so writes through shell or an external process are detected and quarantined even if no provider post-hook arrives.

Hooks are provider edit-tool boundaries, not operating-system sandboxes. Keep the normal provider sandbox and approval policy enabled. Shell and specialized tools are observed after the fact; a PostToolUse block cannot undo a completed write. Active quarantine blocks bundle acceptance until the change is reviewed and released.

## Current implementation boundary

The current vertical slice implements context compilation, receipt-bound editing, sidecar-owned file and semantic-symbol proof, deterministic verification, recovery retries, quarantine, Change Bundles, manifest audit, dual CLI adapters, worktrees, durable leases, provider switching, and durable asynchronous bounded scheduling and cancellation. A worker may continue in its approved worktree and consume the selected CLI subscription after `kontext_schedule_logic` returns. Schedule input and results remain private digest-checked sidecar state; a process restart marks unfinished work `interrupted` instead of silently resuming stale authority. Automatic schedule resume, semantic bundle integration, and risk-based blind cross-runtime review remain Phase 5 work; the main thread must perform those steps explicitly.
