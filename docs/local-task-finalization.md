# Durable local Task finalization

An assessment answers whether completion requirements hold now. A finalization
records the user's explicit decision to finish against that reviewed evidence.
Neither is plan approval or Code/Domain Owner approval.

The host-only `kontext_finalize_task` accepts `taskId`, `jobId`, a `requestId` UUID
and the assessment's `expectedCompletionBasisDigest`. It cannot accept a caller's
commit evidence, owner identity, approval roles, timestamps or verifier results.
Ordinary worker sidecars advertise neither it nor `kontext_inspect_finalization`.

## Exact evidence, then immutable history

The basis digest binds registered ownership, Task Contract/frozen snapshot,
integration, current source freshness and persisted verification/bundle/review
inputs. Only generated full-query manifest-audit runs are excluded, so unchanged
reassessment does not invalidate review merely by creating another audit timestamp.
A changed test result or source does invalidate the reviewed basis.

A new finalization reruns the existing assessment. Nothing is recorded unless the
basis matches and the evaluator reports `done` with current context, no issues and
a valid Accuracy Manifest. It does not start models/verifiers, change project code,
commit, publish, merge or synthesize an approval.

Principal/Task-scoped history uses the existing local mutation lock, schema/digest
checks, a private fsynced temporary file and atomic rename. Immutable records retain
the **whole Accuracy Manifest**, not only an ID in a replaceable latest-manifest
slot. UUID reuse with different input is rejected. At 1,000 records per Task the
store refuses another append rather than discarding history. This is not an OS
security boundary, a filesystem-wide code/context transaction or a power-loss
guarantee.

Exact UUID replay returns the original record with `currentEvidence: not_revalidated`.
It does not record again or claim old code remains current. A new record reports
`validated_at_recording` and the assessment timestamp. Later changes still require
reassessment; the record does not freeze a working directory.

Inspection reads a selected UUID or latest Task record without assessment. Missing
history is not proof that an in-flight request stopped. Corrupt history is an error,
not an empty result. The local principal must match initial Task registration.

## Explicit current-evidence revalidation

`kontext_revalidate_finalization` takes only `taskId` and the latest inspected
`expectedRecordId`, plus host authority. It reuses completion assessment for the
recorded schedule, then rechecks ownership and the latest history identity. Missing
history, a replaced record, dirty/unavailable code or assessment failure is an error,
never a cached success. A concurrent new finalization invalidates the observation.

The result is `revalidated_current` only when the recorded and observed completion
basis digests agree, the evaluator reports `done` with no issues, context is current,
and the manifest binds this Task, code revision and context. Otherwise a completed
assessment reports `changed`. It returns observation time, identities and counts,
not source bodies. It does not append or rewrite finalization history, run models,
grant approval, publish code or manufacture completion evidence.

This is an explicit assessment operation, not a read-only or automatically retried
poll: assessment may write audit artifacts and recapture source state. Its MCP
annotations reflect those effects. A valid result describes one observation, not
an immutable current Task state or a whole-filesystem transaction. Ordinary workers
do not advertise this host tool. Old host errors must remain visible, without
falling back to historical inspection and calling that revalidation.

## Remaining approval decision

This route creates no Code/Domain Owner approval. Medium/high-risk Tasks remain
pending when those approvals are absent. Whether a personal-only project's user
may explicitly attest both owner roles is awaiting user direction; it has not been
inferred from local ownership. Managed authorization is not replaced by personal
attestation. Role-authorized approval recording and aggregate native Task-list
status projection remain unfinished. Kondex's existing completion record detail
now exposes explicit revalidation, retaining the historical record alongside its
time-bound current-evidence observation.

## Verification

Tests use real temporary Git, Markdown/SQLite provenance, packaged MCP and private
file storage with explicitly seeded trusted verification artifacts. They cover
concurrent UUID replay, restart, changed input, absent approval, corrupt history
and historical manifest preservation across later audits. This establishes the
recording/recovery contract, not real-model coding quality.
