# Registered Task metadata inventory

The host-only `kontext_list_tasks` projects the current local principal's registered
Tasks. Kondex exposes it through the owning runtime's `kontext.listTasks`; it must
not substitute a client's local data when a remote host is unavailable or older.
Ordinary workers do not advertise this management tool.

## Scope and effects

Initial Task registrations are authoritative for ownership and workspace identity.
Only registrations matching both organization and subject, with an explicit context
selection, are included. Standalone legacy prepared context is not adopted as a
registered Task. Task intent, risk and context digest come from current prepared
context; registration time remains the original snapshot time.

Existing stores supply schedule summaries, the saved integration record and latest
principal-scoped finalization record. The output excludes source bodies, worker
prompts, approval claims, credentials and process identities. Workspace paths and
Task intent are intentional host-authorized metadata, not public information.

Listing never runs models, resumes schedules, probes process liveness, opens project
files, recaptures sources or assesses completion. Existing local-principal loading
may initialize a missing profile identity. Results always carry
`observation: saved_metadata_only` and `currentEvidence: not_revalidated`.

Schedule count and unsettled count are saved-record counts. Queued, running,
cancelling and interrupted records are unsettled; this does not establish whether
a process is live. Completed schedules do not establish Task completion. Integration
and historical finalization remain separate nullable metadata, never a fabricated
aggregate `done` state. Current validity requires explicit completion revalidation.

## Paging and integrity

Requests accept an optional workspace ID, a limit from 1 to 100 (default 50), and
an opaque digest/offset cursor. Tasks sort by descending registration time and then
Task ID. Each page scans saved records and compares two observations, then checks
the principal again. The digest binds owner, workspace filter and projected rows.
Observed changes, replaced ownership, invalid cursors or corrupt records fail the
whole read; callers explicitly reload the first page instead of mixing snapshots.

Hashed-record discovery inspects at most 10,000 directory entries per store and
refuses a canonical record name that is not a regular file. Missing directories
are empty; other I/O failures propagate. Stores retain their own envelope, digest
and filename/record-identity checks. Ignored temporary names still count toward the
scan limit; exceeding it is an error, not a silently truncated inventory.

This is a bounded full scan, not an indexed query or a filesystem transaction.
Two matching observations do not prevent a change after the response. Per-record
byte limits, hostile local-file races and OS-level isolation are not established
by this operation. Existing private-profile storage assumptions still apply.

## Verification and delivery boundary

Tests cover owned/foreign and workspace filtering, legacy exclusion, pagination
changes, owner replacement, corrupt/renamed registrations, saved schedule changes,
and preservation of running records. Actual packaged MCP tests create and list a
Task and refuse an unauthorized host caller; completion tests project integration
and immutable history using seeded trusted verification, not a real model.

Kondex validates response identities, status enums, counts, filters, cursor offsets
and cancellation, strips unexpected private fields and preserves older-host errors.
Kondex now exposes an explicit, searchable, paginated registered-task list. Opening
a row loads its existing prepared Task and workspace without starting a model or
adopting a saved schedule into the local execution journal. A listed completion
record opens the existing inspection/revalidation view, fenced to the listed record
and schedule IDs, even when the local request journal is empty. Reload or failed
paging removes the displayed record. No completion is recorded by this read path.

The selected Task's latest saved schedule and explicitly loaded execution history
now open host-owned inspection, subscription-consented resume and cancellation
without a local request journal; see
[registered schedule recovery](registered-schedule-recovery.md). Opening the Task
does not automatically read or resume that execution. History supports paging and
search over loaded rows. Completed host-discovered runs also expose explicit saved
integration inspection and consented integration, with reviewed-record checks and
separate completion assessment. The GUI does not
invent live process status or a persisted current-completion flag; those remain
separate observations.
