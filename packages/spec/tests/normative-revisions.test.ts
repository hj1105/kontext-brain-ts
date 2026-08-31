import { describe, expect, it } from "vitest";
import {
  type DecisionRevision,
  type NormativeProposal,
  acceptNormativeProposal,
  isEnforcingActivation,
} from "../src/index.js";

const workspaceDecision: DecisionRevision = {
  kind: "decision",
  organizationId: "personal:heejae",
  recordId: "decision:runtime",
  revisionId: "decision:runtime@1",
  scope: { kind: "workspace", workspaceId: "workspace:kontext" },
  statement: "Codex is the main orchestrator.",
  evidence: [{ evidenceId: "evidence:session-1" }],
  egress: {
    dataClassification: "internal",
    allowedRuntimeProviders: ["codex", "claude"],
  },
  authoredBy: "user:heejae",
  authoredAt: "2026-08-28T00:00:00.000Z",
};

function proposal(candidate: DecisionRevision): NormativeProposal {
  return {
    proposalId: `proposal:${candidate.revisionId}`,
    candidate,
    proposedBy: "agent:planner",
    proposedAt: "2026-08-28T00:01:00.000Z",
  };
}

describe("normative revision acceptance", () => {
  it("keeps an AI proposal non-enforcing until a person accepts it", () => {
    const draft = proposal(workspaceDecision);

    expect(isEnforcingActivation(draft)).toBe(false);

    const activation = acceptNormativeProposal(draft, {
      kind: "local",
      approvedBy: "user:heejae",
      approvedAt: "2026-08-28T00:02:00.000Z",
    });

    expect(activation).toMatchObject({
      recordId: "decision:runtime",
      revisionId: "decision:runtime@1",
      state: "accepted_local",
    });
    expect(isEnforcingActivation(activation)).toBe(true);
  });

  it("rejects local acceptance for a Codebase or Organization revision", () => {
    const organizationDecision: DecisionRevision = {
      ...workspaceDecision,
      revisionId: "decision:runtime@2",
      scope: { kind: "organization", organizationId: "org:acme" },
      supersedesRevisionId: workspaceDecision.revisionId,
    };

    expect(() =>
      acceptNormativeProposal(proposal(organizationDecision), {
        kind: "local",
        approvedBy: "user:heejae",
        approvedAt: "2026-08-28T00:03:00.000Z",
      }),
    ).toThrow("Local Acceptance is limited to Personal or Workspace scope");
  });

  it("requires Code Owner approval for a managed revision", () => {
    const codebaseDecision: DecisionRevision = {
      ...workspaceDecision,
      revisionId: "decision:runtime@2",
      scope: { kind: "codebase", codebaseId: "codebase:kontext-brain-ts" },
      supersedesRevisionId: workspaceDecision.revisionId,
    };

    expect(() =>
      acceptNormativeProposal(proposal(codebaseDecision), {
        kind: "merged",
        approvedBy: "user:reviewer",
        approvedAt: "2026-08-28T00:03:00.000Z",
        mergeCommit: "abc123",
        roles: [],
      }),
    ).toThrow("Managed acceptance requires Code Owner approval");

    expect(
      acceptNormativeProposal(proposal(codebaseDecision), {
        kind: "merged",
        approvedBy: "user:reviewer",
        approvedAt: "2026-08-28T00:04:00.000Z",
        mergeCommit: "abc123",
        roles: ["code_owner"],
      }),
    ).toMatchObject({ state: "accepted", mergeCommit: "abc123" });
  });

  it("does not use a managed merge to activate a Personal or Workspace revision", () => {
    expect(() =>
      acceptNormativeProposal(proposal(workspaceDecision), {
        kind: "merged",
        approvedBy: "user:reviewer",
        approvedAt: "2026-08-28T00:04:00.000Z",
        mergeCommit: "abc123",
        roles: ["code_owner"],
      }),
    ).toThrow("Merged acceptance is limited to Codebase or Organization scope");
  });

  it("requires Evidence before any normative proposal can be accepted", () => {
    expect(() =>
      acceptNormativeProposal(proposal({ ...workspaceDecision, evidence: [] }), {
        kind: "local",
        approvedBy: "user:heejae",
        approvedAt: "2026-08-28T00:05:00.000Z",
      }),
    ).toThrow("Accepted normative revisions require Evidence");
  });
});
