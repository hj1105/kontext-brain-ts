# Host-owned local source registration

The executable Task sidecar now registers `kontext_register_markdown_source` only
when started with a host-management capability. Kondex generates that ephemeral
capability, passes it to its private sidecar and injects it into management calls;
callers cannot override it. Ordinary worker sidecars do not advertise the tool.

The sidecar removes the capability from its process environment before creating
runtime adapters, observers or verifiers. Subscription worker environments also
exclude it. This separates management calls from normal worker MCP operations;
it is not an OS security boundary against another malicious process with the same
filesystem privileges. Capabilities are never returned as registration metadata.

Registration currently accepts a selected workspace and a relative `.md` or
`.markdown` path. It verifies canonical workspace confinement, regular-file type,
UTF-8, a 512 KiB byte limit and file identity/version across capture. Source bodies
are not truncated. Existing H2 section chunking is reused with a one-character
minimum so short rules are retained; duplicate headings have distinct source IDs.
There is no automatic semantic classification or normative extraction at this step.

The local principal is generated once and atomically published in a private
identity file; concurrent initializers reuse it, and malformed existing identity
files are not overwritten. Resource synchronization uses the durable SQLite graph
and private file-backed content storage under the host-owned data directory. The
Resource ACL names that local principal. The response contains Resource/Evidence
IDs and the captured content version, not source text.

New content objects use four bounded path segments (`objects-v2` plus organization,
Resource and content-version hashes). This avoids filesystem filename limits when
a valid source path produces a long Resource ID. Native identity and provenance
remain inside the object and graph unchanged. Existing three-segment object keys
continue to read and purge without migration; the extra namespace depth prevents
new keys from aliasing legacy keys. New content files/directories are private.

A changed source is resynchronized; if a previously registered selected file cannot
be captured, its known Resource is marked stale. Restoring and explicitly collecting
it again can make it active. Old source content is not deleted by capture failure.
This does not yet watch the filesystem or refresh already-prepared Task state.
The private source registry now retains the canonical workspace, relative path,
principal and Resource identity. `kontext_refresh_source` recaptures by Resource ID;
even a missing workspace root can therefore mark the known Resource stale. A
replacement workspace resolving to another identity is refused. Preflight wiring
and filesystem watching remain separate work.

Kondex now exposes registration through its owning-runtime RPC and a Markdown
form. A hidden Electron integration test exercises this generated MCP bundle with
a real non-Git folder, capture hash changes and deletion failure; responses shown
in the GUI contain metadata only. Direct SSH/WSL source handling is not substituted
with a client-local read. The form requires a registered workspace selector;
graphical workspace/file selection is still pending.

Registration itself grants neither provider sharing nor normative approval. Its
legacy `providerSharing: not_granted` response describes the registration operation,
not the current permission configuration. An unchanged capture preserves an
existing explicit permission; `kontext_inspect_source` reports that configuration.

Kondex now also exposes explicit permission inspection, file recapture and
`kontext_set_source_sharing`. The user selects Codex and/or Claude, a data
classification, and confirms the captured version. Selecting neither provider
revokes permission. The host binds this choice to the inspected content hash and
registry revision, records its own principal and timestamp, and rejects competing
or replayed stale revisions. Source text and worker claims cannot grant access.
All five source-management tools require the host capability and are absent from ordinary
worker sidecars. The GUI supports entering a previously saved Resource ID after a
reload, and now has a searchable registered-source picker in the planning form.

`kontext_list_sources` enumerates only the current principal's registered sources
that remain accessible in the graph; missing, purged or ACL-revoked Resources are
omitted. It returns saved metadata and permissions, never content bodies, and does
not recapture files or imply live freshness. Pages (default 50, maximum 100) bind
to a digest of the principal and visible metadata. Changed membership, sharing,
capture versions or status refuse old cursors; callers must reload the first page.
Corrupt registrations, mismatched identity slots or non-regular registry entries
fail the read without deleting or repairing records. Enumeration is limited to
10,000 directory entries and refuses oversized inventories rather than returning
a misleading complete list. It holds the existing source-management lock; legacy
graph writers outside that lock are not part of an atomic cross-store snapshot.
The list is a metadata observation, not a capability or Task-selection authority.

The owning-runtime RPC validates the returned identity/version/permission choice
and strips source bodies and capability fields. The GUI checks the pairing revision
before and after each call, clears consent when choices change, and never retries
automatically. A lost save acknowledgement leaves the outcome unknown; another
inspection is required before a new change. Unsupported older hosts do not fall
back to client-local operations. Captured active status is not live file freshness.

The registry stores only locator, permission and audit metadata in private JSON
files. Host mutations are serialized using the existing local-file mutation lock;
publication uses file sync and atomic rename. Directory metadata is not synced, so
this is not a power-loss durability guarantee, nor protection against malicious
same-OS-user filesystem access. The graph and registry are separate stores: old
permissions are cleared *before* publishing changed graph content, and a
graph/registry hash mismatch denies provider use. Failures never silently repair
corrupt registry files or discard the approval audit. Legacy graph writers that
bypass this host workflow do not participate in its serialization.

`LocalKnowledgeOperations.collectTaskEvidence` now supplies the registry-backed
egress policy to the existing graph/ACL/provenance collector. It intersects provider
permissions before and after hydration; missing registration, hash mismatch or
revocation cannot produce an allowed provider. This does not retroactively revoke
already-dispatched context or silently advance already-prepared Task snapshots.
The host-reviewed personal Task creation path now uses it and revalidates sources
and the local normative overlay for subsequent operations; see
[host-reviewed Task creation](./local-task-creation.md). The host-owned planner and
new-Task GUI now use this path; managed normative projection selection remains
unfinished. Markdown session notes can
be collected as files; automatic native session ingestion and managed connector
selection are not implemented by this local Markdown path.

Tests exercise real file and SQLite storage, stable identity across restart,
changed/deleted/restored files, short sections, invalid paths, outside-workspace
symlinks, invalid encoding and size limits. The generated MCP bundle is tested
through three independent processes: two differently keyed host sessions and one
ordinary worker. It checks persisted grants across restart, capability rejection,
stale-revision rejection and recapture. Storage tests also cover whole-workspace
loss, eight conflicting permission writes, ACL revocation, corrupt registry
preservation, and failed graph publication followed by restoration of old bytes.
The same protocol test previously passed with Kondex's Electron binary
running as Node, with disposable HOME/data, empty PATH and no provider credentials.
