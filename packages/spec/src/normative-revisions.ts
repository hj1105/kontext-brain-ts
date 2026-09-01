import type { NormativeActivation, NormativeApproval, NormativeProposal } from "./domain.js";

export function acceptNormativeProposal(
  proposal: NormativeProposal,
  approval: NormativeApproval,
): NormativeActivation {
  const revision = proposal.candidate;
  if (revision.evidence.length === 0 || revision.evidence.some((item) => !item.evidenceId.trim())) {
    throw new Error("Accepted normative revisions require Evidence");
  }
  if (
    revision.egress.allowedRuntimeProviders.length === 0 ||
    revision.egress.allowedRuntimeProviders.some((provider) => !provider.trim())
  ) {
    throw new Error("Accepted normative revisions require a provider egress policy");
  }
  if (!approval.approvedBy.trim() || !approval.approvedAt.trim()) {
    throw new Error("Normative acceptance requires approval provenance");
  }

  if (approval.kind === "local") {
    if (revision.scope.kind !== "personal" && revision.scope.kind !== "workspace") {
      throw new Error("Local Acceptance is limited to Personal or Workspace scope");
    }
    return Object.freeze({
      organizationId: revision.organizationId,
      kind: revision.kind,
      recordId: revision.recordId,
      revisionId: revision.revisionId,
      scope: revision.scope,
      state: "accepted_local" as const,
      acceptedBy: approval.approvedBy,
      acceptedAt: approval.approvedAt,
    });
  }

  if (revision.scope.kind === "personal" || revision.scope.kind === "workspace") {
    throw new Error("Merged acceptance is limited to Codebase or Organization scope");
  }

  if (!approval.roles.includes("code_owner")) {
    throw new Error("Managed acceptance requires Code Owner approval");
  }
  if (!approval.mergeCommit.trim()) {
    throw new Error("Managed acceptance requires a merge commit");
  }

  return Object.freeze({
    organizationId: revision.organizationId,
    kind: revision.kind,
    recordId: revision.recordId,
    revisionId: revision.revisionId,
    scope: revision.scope,
    state: "accepted" as const,
    acceptedBy: approval.approvedBy,
    acceptedAt: approval.approvedAt,
    mergeCommit: approval.mergeCommit,
  });
}

export function isEnforcingActivation(
  value: NormativeProposal | NormativeActivation,
): value is NormativeActivation {
  return "state" in value && (value.state === "accepted" || value.state === "accepted_local");
}
