# Host completion assessment

`kontext_assess_completion` accepts only a registered `taskId` and its integrated
`jobId`, plus the private host-management capability. Ordinary worker sidecars do
not advertise it. The host removes the capability from the environment before
constructing subprocesses. Caller-provided approval, commit, invariant verdict,
workspace path or Task state is not an input to this operation.

The operation uses `LocalKontextCompletionOperations.proposeTransition`, the
existing Accuracy Manifest audit and `evaluateTaskState`. It does not replace
their rules with a GUI-specific success predicate.

## Evidence path

1. Require the Task's initial registration to belong to the current local principal.
   Legacy unregistered Tasks are not silently adopted. The integrated repository
   must match the registered workspace and the selected schedule.
2. Observe the stored integration workspace on its owning host. Require its actual
   clean Git HEAD to equal the integration commit and its observed content revision
   to equal the integrated revision. Active Task quarantine prevents assessment.
3. Recapture registered Markdown sources and current local normative rules. Keep
   the Task's frozen context unchanged. Derive each bound invariant evaluation
   from every exact verifier's persisted full-tier runs for this code/context;
   missing, failed or inconclusive evidence is never promoted to passing.
4. Derive commit Evidence from the observed Git commit. Invoke the existing
   manifest audit/transition evaluation. This writes audit runs and, when valid,
   the Accuracy Manifest. It does not execute tests, models or independent review.
5. Recheck the workspace, integration, quarantine, prepared Task, current context
   and non-audit verification/review/bundle inputs before returning the observation.
   Concurrent changes invalidate the answer. This is not a filesystem-wide atomic
   snapshot or a guarantee against changes after observation.

The response includes a stable completion-basis digest, the verdict, blocking/missing issues, observation time,
commit/context identities, current verification-run metadata, invariant evaluations
and Accuracy Manifest. Source bodies and provider credentials are not returned.

## Deliberate limitations

A manifest is evidence, not an approval or an immutable `done` flag. A stale-source
assessment can be blocked even if a code-bound manifest exists. A failed response
must not retain a prior successful verdict. Reassessment can create another audit
artifact, so clients do not automatically replay or poll this operation.

Code Owner approval for medium/high risk and Domain Owner approval for high risk
are **not inferred** from local ownership, plan approval or a passing independent
review. Their trusted approval-recording workflow remains to be implemented.
[Durable finalization history](./local-task-finalization.md) now records explicitly
reviewed completion through this assessment, but is not an owner-approval shortcut.
The underlying legacy worker `propose_transition`
interface is unchanged; this new host route supplies its own evidence.

## Verification

`local-task-completion-assessment.test.ts` covers real temporary Git, registered
Markdown/SQLite provenance, the actual packaged MCP, authority checks, source/code
changes, quarantine, concurrent verifier changes and exact invariant binding.
Verification runs/bundles in that fixture are explicitly seeded trusted test
artifacts, not claims of a real model fixing code. The existing integration test
continues to exercise actual worker-patch integration and core completion audit.
