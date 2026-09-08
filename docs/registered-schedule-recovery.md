# Registered schedule inspection and explicit control

The execution host can inspect and control an exact saved schedule belonging to a
Task registered to its current local principal. These operations do not require
the client that originally submitted the request or its local request journal.
They reuse the existing schedule store and runtime manager, not a second runner.

## Host entry points

| MCP tool | Owning-runtime RPC | Required inputs beyond host authentication |
| --- | --- | --- |
| `kontext_list_registered_schedules` | `kontext.listRegisteredSchedules` | `taskId`, optional `limit` and `cursor` |
| `kontext_inspect_registered_schedule` | `kontext.inspectRegisteredSchedule` | `taskId`, `jobId` |
| `kontext_resume_registered_schedule` | `kontext.resumeRegisteredSchedule` | Selector, `expectedJobIdentityDigest`, `allowSubscriptionExecution: true` |
| `kontext_cancel_registered_schedule` | `kontext.cancelRegisteredSchedule` | Selector, `expectedJobIdentityDigest` |
| `kontext_inspect_registered_integration` | `kontext.inspectRegisteredIntegration` | Selector |
| `kontext_integrate_registered_schedule` | `kontext.integrateRegisteredSchedule` | Selector, `expectedJobIdentityDigest`, nullable `expectedIntegrationDigest`, `allowSubscriptionExecution: true` |

These are host-management tools, absent from ordinary worker MCP surfaces. The
owning runtime is authoritative, including for SSH. An unavailable or older host
returns an error; the client must not substitute local state or execution.

## Read-only observation

History listing reuses the bounded hashed-record scan and saved summary projection.
It requires an owned registration and returns only that Task's executions, newest
request first with job ID as the tie-breaker. The page limit is 1–100 (default 50).
The digest/offset cursor binds the principal, registered owner, Task and projected
history. Two captures and an ownership recheck refuse observed changes; callers
reload the first page after a stale or invalid cursor. Missing schedule storage is
an empty history, not permission to create, resume or adopt an execution.

List metadata includes saved status, requested time, code/context identity and job
ID. It excludes cancellation intent and private progress detail: a cancellation
that leaves the saved status interrupted need not change the history digest. Exact
execution inspection is still required before controls appear. The scan is not an
indexed query, a filesystem transaction, a live-process verdict or a completion
assessment. Existing scan and private-profile limitations still apply.

Inspection reads the saved job directly. It does not use the runtime manager's
potentially resuming `getSchedule`, probe process liveness, open project files,
capture sources, assess completion or dispatch a model. The response explicitly
states `saved_metadata_only` and `not_revalidated`.

The projection contains saved job status/timestamps, code and context identity,
repository path, WorkItem IDs, stored provider choices and saved result metadata.
Prompts, source bodies, process identities and raw diagnostics are excluded; only
the presence of a diagnostic is exposed. Saved results are not current settlement
evidence, and a completed schedule is not proof that the Task is complete.

The reviewed digest binds the principal, registered owner and immutable execution
identity, including the stored request. It does not change merely because progress
or cancellation intent changes. Inspection compares two saved-state observations
and checks ownership again. A changed observation fails instead of returning mixed
state. This is not a filesystem transaction or a hostile-local-file security seal;
private-profile assumptions and the possibility of changes after a response remain.
No new per-job byte or WorkItem-count limit is established by this operation.

## Explicit actions and uncertain outcomes

Resume and cancellation first inspect ownership and require the reviewed identity
digest. They delegate once to the existing runtime operation and inspect again,
rejecting a different returned job or changed ownership/immutable identity. They
never reconstruct an enqueue request or generate another execution UUID.

Resume requires explicit subscription consent. Existing context, provider,
subscription availability, lease and resume-limit checks still apply. A response
may report `resumeBlocked`; absence of that blocker is not proof a worker started.
No API-billing fallback is introduced. Post-command failure cannot undo a dispatched
action: the result is uncertain and must be inspected, not automatically replayed.
Ownership checks around dispatch do not constitute a global authorization lock.

Cancellation persists intent under the existing job-store lock. An interrupted
job remains interrupted with `cancellationRequestedAt`; it does not become cancelled
without evidence of settlement. Future resume attempts refuse that intent, including
after restart or when cancellation races resume preparation. Active jobs continue
through the existing cancelling/abort path. This change does not redefine every
existing process-liveness check or prove that an interrupted worker has stopped.

## Integration of a selected execution

Integration inspection reads the existing Task-level integration store alongside
the exact owned schedule, with repeated observations. The Task's latest integration
may belong to a different execution; that distinction is returned rather than
substituting the selected job ID. Matching selected records must agree on base
revision, context digest and WorkItem identities. No Git, project verification or
model execution is performed by this read. A missing record does not prove a prior
attempt made no changes.

Explicit integration requires the reviewed job identity, the reviewed integration
record digest (or null for no record), completed saved WorkItem results and fresh
subscription consent. The registered and requested repository paths are compared
on the owning host. Observation/retry timestamps are host-derived, not client claims.
The existing integrator rechecks prepared/current context and Change Bundles,
constructs its integration worktree/commit, executes verification and performs
independent review when required. This can replace the Task's latest integration
record, which is stated in the consent; it does not rewrite finalization history.

All calls through `LocalScheduleIntegrator`, including the legacy entry point,
share the existing local-file-lock mechanism keyed by Task. New owned calls check
their expected integration digest inside that lock before Git/verification/review
effects. Concurrent callers cannot both consume the same reviewed empty record.
Legacy callers retain their existing request shape and do not acquire new approval
authority. The lock serializes owning-host integrator calls; it is not global
ownership isolation, a rollback mechanism or proof that orphaned subprocesses have
exited. Private-profile assumptions remain.

The command delegates once and verifies the returned state against persisted
integration and selected schedule identity. Ownership or state changes after
dispatch can produce an uncertain outcome; no retry or rollback is implied.
Reading the record recovers a persisted result after response loss. Returned data
remains `saved_metadata_only` / `not_revalidated`, even when integration returned
normally: current completion evidence and trusted owner approval remain separate.

## Kondex delivery boundary

The registered Task list exposes its selected Task's latest saved execution and
an explicitly loaded, searchable, paginated execution history. Opening a history
row selects its exact saved job without inspecting, resuming or integrating it.
The user explicitly reads it before controls appear. Resume needs fresh consent;
duplicate requests and stale pairing replies are rejected. Each action clears the
previous snapshot and consent, and failures require a fresh read. No execution is
silently adopted into the local request journal.

Switching executions discards the old detail and consent. Loading another Task or
owner isolates pending replies; failed history refresh removes selected controls.
Completed selected executions now expose explicit integration-record inspection and
consented integration without the local journal. Once the selected result is saved,
the existing completion-assessment/finalization controls are available without an
automatic assessment or invented owner approval. A different saved execution is
clearly identified before replacement. Failed actions clear the old result and
consent; recovery is a new read, not a replay.
Historical completion inspection and current-evidence revalidation retain their
existing independent controls.

## Verification

Owned integration extension 791–799: sequential backend regression 797 passes 45
tests across seven files. Its temporary-Git test crosses the owned-operation path,
integrates an accepted worker bundle, verifies the final revision with seeded
deterministic verifiers and accepts only one concurrent request against the same
reviewed empty record. Existing-result reuse, owner changes, lost-response read
recovery, bundled MCP authorization and completion evidence are covered separately.
App regression 796 passes 119 tests across eleven files. Hidden Electron 798 passes
six flows, including older-run integration, intentional response loss and renderer
reload without replay or cached completion. English/Korean light/dark screenshots
were inspected. No real model, paid API or live SSH was used.

Production typechecks, strict selected backend test typing, formatting/lint and the
E2E-mode build pass. E2E typing still reports 38 inherited errors. The initial combined
parallel validation timed out the Git test at 30 seconds; after those runs ended,
the unchanged test passed in 4.4 seconds in the sequential suite. Its timeout was
not increased. Installed-app and production-package validation remain separate.

History extension 784–790: 47 backend regression tests across six files pass,
including real bundled host MCP listing/denial, deterministic paging, stale cursors,
changed ownership/status, empty storage and cancellation of an older execution
without changing a newer one. App regression 789 passes 86 tests across ten files.
Hidden Electron 788 passes five flows, including searching the second history page,
selection/consent isolation, old-run cancellation and renderer reload. English and
Korean light/dark screenshots were inspected. No actual model or live SSH ran.
Production typechecks and the E2E-mode build pass. E2E typing still reports 38
inherited errors; only displayed object-property ordering differs from the prior
log, not the diagnostic locations/codes or underlying missing properties.

Regression 783 passed 55 backend tests across seven files, including actual bundled
host MCP authorization/inspection/cancellation, restart refusal, cancellation during
resume preparation, owner replacement, malformed progress, existing recovery gates
and completion evidence. App validation passed 54 UI/RPC/catalog tests (780); the
earlier service/RPC regression passed 37 tests (778). These suites overlap.

Hidden Electron validation 783 passed four flows through the actual app/MCP
transport, using seeded protocol fixtures without real model execution. The new
flow covers empty local journals, explicit blocked resume, cancellation intent and
renderer reload. English and Korean light/dark screenshots were inspected. Production
typechecks and the E2E-mode build pass; E2E typing still has the same 38 inherited
errors. Actual subscription execution, live SSH and production packaging were not
validated in this step.
