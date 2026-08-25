# Scoring profile rollout runbook

## Preconditions

Before staging a profile, record its development, validation, and holdout results separately. The
artifact must identify the profile digest, feature schema, corpus and candidate-pool digests, code
commit, split IDs, and the exact evaluation command.

Do not activate a profile if it violates any of these gates:

- Evidence Recall@10 drops by more than 1 percentage point on any required dataset cell.
- strict faithfulness or citation F1 has a statistically significant regression.
- ACL/RLS or conflict-disclosure tests fail.
- score-breakdown coverage is below 100%.
- retrieval p95 rises by more than 10% in an isolated comparison.
- visited or candidate budgets are exceeded.

## Stage

1. Validate the profile with `validateTraversalScoringProfile`.
2. Call `PostgresScoringProfileRepository.stage(organizationId, profile, evaluationSummary)`.
3. Save the returned digest with the evaluation report.
4. Confirm `getActive()` still returns the previous profile.

Staging is idempotent for the same profile content. Reusing the same profile ID and version with
different content is rejected by the database uniqueness constraint; increment the version.
Normal activation rejects a profile without an evaluation summary. `allowUnevaluated` is a
break-glass option and its use must be recorded as an incident action.

## Shadow

1. Call `setShadow(organizationId, stagedDigest)`.
2. Verify that active responses still report the active profile at `trace.scoring.profileDigest`.
3. Collect `trace.scoring.shadow` overlap, normalized rank disagreement, latency, visited count, and
   candidate count.
4. Also monitor missing-signal counts, zero-evidence rate, conflict evidence rate, and ACL outcomes.
5. Call `setShadow(organizationId, null)` when the evaluation window ends.

Shadow mode runs an independent frontier and top-K inside the same read-only search session. It
doubles graph work by design, so do not leave it enabled indefinitely.

## Activate and canary

Set the initial canary percentage before activation so a new profile never receives an accidental
100% traffic window. Activation is atomic:

```ts
await scoringProfiles.setCanaryPercent(organizationId, 5);
await scoringProfiles.activate(organizationId, stagedDigest);
```

Start with an internal organization, then set a stable subject-level cohort with
`setCanaryPercent(organizationId, percent)`. Recommended checkpoints are 5%, 25%, 50%, and 100%.
The resolver hashes organization and subject IDs, so the same subject remains in the same cohort.
At each checkpoint compare retrieval quality,
citation coverage, abstention, zero-evidence rate, p95 latency, and budget utilization against the
previous profile.

Activating the configured shadow profile clears the shadow pointer automatically.

## Roll back

Keep the previous active digest in the deployment record. Rollback is an atomic pointer and status
change; no schema rollback or reindex is required:

```ts
await scoringProfiles.rollback(organizationId, previousDigest);
```

After rollback, confirm that new traces contain the previous digest and that the resolver cache was
invalidated. Rollback resets the canary percentage to 100 so every subject receives the restored
profile. Preserve the failed profile and its telemetry for analysis; do not edit its stored content.

## Incident checks

If retrieval quality changes unexpectedly, inspect in this order:

1. active and shadow profile IDs, versions, and digests;
2. seed provider counts and missing-signal counts;
3. selected evidence score breakdowns and path lengths;
4. conflict, stale, origin, and freshness observations;
5. ACL-filtered support aggregates and RLS session organization;
6. candidate, visited, hop, and time stop reasons;
7. corpus, index, candidate-pool, and feature-schema digests in the benchmark artifact.

Never repair an incident by editing constants in an adapter. Create a new profile version, evaluate
it, and move it through stage, shadow, and activation.
