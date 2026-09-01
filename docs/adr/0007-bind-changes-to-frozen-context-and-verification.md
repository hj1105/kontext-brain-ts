# ADR 0007: Bind changes to frozen context and verification

- Status: Accepted

Each Task fixes its applicable normative revisions in a Task Context Snapshot, and each Logic Work Item must obtain a Context Receipt for its target before editing. Completion is accepted only when the current code revision and context digest have passing Verification Runs and an Accuracy Manifest connecting the Task Contract to the resulting commit; conflicting, stale, inaccessible, or unavailable mandatory context produces an inconclusive result rather than guessed compliance. A coding agent cannot approve a rule change or review finding needed to legitimize its own patch.
