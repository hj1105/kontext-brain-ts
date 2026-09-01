# ADR 0009: Combine local-first work with managed governance

- Status: Accepted

A local sidecar owns code and session synchronization, private overlays, worktree leases, and durable verification retries, while an optional Organization service owns managed connectors, canonical approval, pull-request and webhook integration, ACL projection, and shared storage. Personal, Workspace, Codebase, and Organization scopes can coexist, but a narrower overlay cannot hide or weaken managed Organization rules and provider-specific data-egress policy remains authoritative. Offline managed editing requires an unexpired signed context lease and cannot claim Organization completion until online revalidation; personal work remains available offline.
