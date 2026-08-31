# ADR 0006: Separate normative governance and Task state from Facts

- Status: Accepted

Facts remain descriptive, evidence-backed observations, while Decisions, Domain Terms, and Invariants are separate normative record families with immutable revisions and active pointers; Tasks are action-driven records whose state is derived from submitted evidence. AI output enters these families only as a Normative Proposal and cannot enforce or approve anything before authorized human acceptance. Managed normative manifests are canonical in Git and projected into PostgreSQL, while Local Acceptance remains in private plugin data until explicitly promoted.
