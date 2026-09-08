# Private coding workspace baselines

Host planning and Task creation now accept uncommitted Git workspaces, repositories
without a first commit and ordinary folders. Clean committed Git workspaces retain
their existing execution path. Git must be available on the owning host in all cases.

`prepareTaskWorkspace` captures working-file contents, not the source index, into
private `coding-workspace-seeds` storage. Tracked files and non-ignored untracked
files are included; folders use their Git ignore rules without creating `.git` in
the source. Tracked deletions are reflected. Original files, staged versions,
branches and commits are not modified. Raw Git blobs preserve bytes and executable
modes without applying clean/smudge filters during materialization. Existing Git
parents are imported locally with depth one, not an unbounded history copy.

Two observations compare source commit/Codebase identity, paths, modes and bytes.
The seed commit is deterministic and publication uses an atomic directory rename;
concurrent identical captures must validate the same published result. This is not
an OS snapshot or a transaction against arbitrary concurrent filesystem writers.
Changed observations fail rather than silently accepting a mixed capture.

Limits are 10,000 paths and 64 MiB of file contents, with bounded Git output and
timeouts. Conflicted Git indexes, nested repositories/submodules, non-regular
files and escaping/absolute symlinks are refused. Internal relative symlinks are
retained where the host supports creating them. Private seed storage must be
outside the selected source workspace. An unsupported or oversized capture is an
explicit failure, not a partial baseline advertised as complete.

## Approval, execution and recovery

The planner reads the private baseline. Before a proposal is offered, and again
during exact-plan approval, the host recaptures the original workspace and requires
the reviewed revision. New edits require a new reviewed plan. A successful Task
registration retains the source path as its owner identity and an optional private
seed reference in the integrity-checked initial envelope. Identical creation retries
still recover the existing registration without replacing its original contract.

`resolveTaskExecutionRepository` checks the registered Task owner and exact source
workspace, then validates the private seed's path, metadata, commit, file bytes,
modes and absence of added files before handing it to the existing
`GitRuntimeWorktreeManager` or `GitChangeBundleIntegrator`. Public schedule and
integration records continue to identify the source workspace. The client cannot
select or inject another seed repository through this route. Source edits after
approval do not rewrite the approved baseline; workers operate in isolated worktrees.
Integration results remain in the existing reviewed integration worktree and do not
overwrite the user's original working files or index.

Copies retain the original Codebase and Code Symbol identities. A private Git
configuration value is bound to the seed commit message and checked by the existing
symbol observer, including from linked worker and integration worktrees. No fetch
or push remote is configured in the private copy for this purpose. This is identity
continuity, not a new source grant, domain approval or authorization mechanism.

`inspectTaskWorkspace` remains read-only and requires a clean Git commit. Completion
assessment uses this strict path; it cannot manufacture a replacement seed to make
changed integrated code look verified. Existing context freshness, source grants,
per-logic receipts, leases, accepted bundles, verification and owner gates remain.

Older initial Task envelopes without a seed reference retain their old behavior.
Older strict sidecar versions may refuse the new optional owner field; they must
not fall back to the source repository or claim the Task is usable. This feature
requires the updated owning-host bundle, not a client-local workaround for SSH.

These are developer-workspace changes. Offline fixture tests do not prove native
Windows/SSH behavior, real subscription coding quality or production deployment.
Private baselines are retained for Task recovery; no automatic deletion policy or
automatic application of integration changes to the source has been introduced.
