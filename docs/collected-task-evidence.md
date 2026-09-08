# Collected Evidence → Task context

`CollectedTaskEvidence` bridges the existing knowledge graph to the Task-context
assembler. It accepts Resource/Evidence identifiers, not caller-supplied source
bodies, provenance, freshness or approval claims. The host supplies the current
authenticated principal and the provider-egress policy; neither is established
by an identifier in a renderer request.

The collector reads the existing Resource, Chunk and Evidence records in the
principal's Organization, applies their ACLs with `DefaultAccessPolicy`, loads
the native chunk from `ResourceContentStore`, and rechecks indexed metadata after
hydration. Missing, purged, stale, inaccessible or concurrently replaced sources
return an explicit unavailable state with no text or provenance metadata. The
content object's Organization, Resource and content-hash identity must match the
index; this is not independent cryptographic verification of the source bytes.

Provenance retains the collected Resource/Chunk IDs, source connector/external ID,
title, observation time, resource revision hash and ontology links. Markdown and
session sources use the same contract as Notion or Slack; their source adapters
must first synchronize real Resource snapshots. This collector does not implement
those adapters, infer Facts, extract decisions or accept normative revisions.

Provider grants are intersected across hydration and can only narrow. Indexed
ACL/freshness checks do not replace an upstream connector refresh, nor revoke
information already delivered to an authorized reader. Host integration must
recollect when synchronizing or refreshing a Task and continue the existing
pre-write context/egress checks.

The integration test exercises actual Resource synchronization, file-backed
content and Task-state storage, assembly, preparation and Context Receipt
compilation. A removed source remains unusable after Task-context refresh. The
test now uses the durable local SQLite graph implementation and no live connectors or models.

The Task-evidence collector is not yet used for executable Task registration.
Host-only local Markdown registration now initializes the durable graph and
synchronizes real selected files; see [local source registration](./local-source-registration.md).
Remaining wiring includes source egress grants, automatic refresh, authenticated
managed graph selection, trusted normative-layer loading, Task/logic planning and
a user-facing registration workflow. Do not enable Task registration by trusting arbitrary assembly JSON as
proof of source access or normative acceptance.
