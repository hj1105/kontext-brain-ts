# Conditional Task context publication

`FileTaskContextRepository` serializes each current/prepared file mutation across
updated local sidecar processes. Conditional comparison, integrity validation and
atomic replacement occur while the writer owns the Task-level mutation lock and
the existing per-file lock. Task-level serialization also excludes initialization.
The previous implementation compared the digest before acquiring any lock: a
16-publisher regression reproduced all 16 accepting the same original version.

For a collected-context refresh, read the payload and its version together:

```ts
const observed = await repository.getCurrentVersion(taskId);
await repository.publishCurrent(taskId, nextState, {
  expectedDigest: observed.digest,
});
```

Use `{ expectedDigest: null }` when first publishing standalone current state; a concurrent or
previous creation then causes a conflict instead of being adopted or overwritten.
Omitting `expectedDigest` retains the existing **trusted unconditional publish**
behavior. New host registration code must not omit it when claiming create-only
or optimistic-concurrency semantics. A content digest compares content, not a
monotonic sequence number; returning to identical content returns the same digest.

Existing envelopes are fully decoded and checked before replacement. A future
schema, wrong Task identity, invalid payload or digest mismatch fails without
overwriting the evidence of corruption. New data is written to an owner-only
temporary file, file-synced and atomically renamed. No power-loss guarantee for
directory metadata is claimed.

## Lock ownership

The host-local lock directory contains an incarnation-specific PID/UUID marker.
Only a contender that observes its marker alone may execute the mutation. A
paused initializer that reaches a replacement directory cannot enter while
another marker is present. Cleanup unlinks only the exact observed marker;
`rmdir` removes only an empty directory, never a replacement writer's marker.
Initialization races retry without running the mutation callback.

Waiting does not expire a live owner. Only `ESRCH` from the local process probe
permits reclaiming that owner's marker. Permission failures, unknown process
state and a recycled live PID retain the lock. Empty interrupted initialization
directories can be reclaimed; unrecognized owner data is preserved and reported.
Waiting is bounded, so a busy writer may receive an error and must re-read/retry
explicitly. A failed mutation is never replayed, including `ENOENT` or `EINVAL`.

This is serialization for private storage on **one execution host**, not a
distributed lock, an OS security boundary or compatibility with older writers
that do not participate in this protocol. Later current/prepared updates remain
two separate records. Initial Task registration now has a separate single-envelope
publication path described below; it does not make all later multi-record updates
atomic.

## Atomic initial Task registration (648–659)

`initializeTask` now stores the owner, initial current state, reviewed Task Contract
and first prepared snapshot in one integrity-checked `initial` envelope. It does
not first publish two partially visible current/prepared files. It constructs the
snapshot using the existing compiler, rather than trusting a caller's digest.
Exact duplicate initialization returns the same registration; different payloads,
owners or pre-existing legacy state are rejected. `LocalTaskCreationOperations`
recovers by host-owned request ID before recapturing sources or changing anything.

Current/prepared readers fall back to this initial envelope until later updates
exist. Updates continue using the existing current/prepared files, but share a
Task-level lock with initialization. Create-only/CAS comparisons include seeded
state, and later updates never mutate the initial record. Corrupted ownership is
not hidden by current/prepared overrides. Existing legacy files remain readable.
Older readers that do not understand initial envelopes cannot execute new Tasks;
all worker sidecars for these Tasks need the matching updated bundle.

The source selection recorded by host creation is revalidated by the executable
sidecar before context is consumed. It does not silently overwrite the frozen
snapshot. Current required source Evidence IDs are distinct from explicit fixed
Evidence references: an explicit snapshot refresh can adopt newly added/removed
source sections while retaining the Task's original source selection.

Atomic registration and source grants now exist; automatic planning, managed
ingestion and the complete new-Task GUI are still unfinished.

## Evidence

- Same-version publication and create-only races: 16 independent repository
  instances, exactly one accepted publication.
- Independent process publication: six initialized Node processes released
  together, one accepted write and five conflicts; final state names the winner.
- Lock contention: 24 callers with at most one active mutation, a live owner held
  beyond the wait limit, callback-error non-replay and recovery after a separate
  owning child process is positively observed to exit.
- Local/context/tool-server regression: 41 files / 151 tests pass (631), after
  rebuilding the local package and actual plugin bundle. Production local and
  tool-server typechecks and strict new-test typechecks pass.

Tests use disposable directories and credential-free processes; none invokes an
agent model. These gates do not establish that new Task creation is connected to
the GUI or that the overall Kondex release is complete.
