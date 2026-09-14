import type { NormativeManifest } from "@kontext-brain/spec";
import { describe, expect, it } from "vitest";
import { assembleCurrentTaskContextState, resolvePluginDataDirectory } from "../src/index.js";

describe("assembleCurrentTaskContextState", () => {
  it("reconciles normative layers, constrains egress, and exposes missing provenance", () => {
    const state = assembleCurrentTaskContextState({
      taskId: "task:assembly",
      organizationId: "org:acme",
      codeRevision: "commit:1",
      baseScopes: [{ kind: "personal", subjectId: "user:owner" }],
      localManifest: manifest("local"),
      managedManifest: manifest("managed"),
      evidence: [
        {
          evidenceId: "evidence:local",
          text: "The user approved the local workflow.",
          availability: "current",
          allowedRuntimeProviders: ["codex", "unapproved-provider"],
          provenance: {
            resourceId: "resource:local",
            chunkId: "chunk:local",
            resourceTitle: "Local decision",
            source: {
              connectorId: "codex",
              externalId: "codex://session/local",
              type: "session",
            },
            observedAt: "2026-08-28T00:00:00.000Z",
            contentHash: "sha256:local",
            ontologyNodeIds: ["workflow", "engineering", "workflow"],
          },
        },
      ],
      logicPlans: [
        {
          workItemId: "work-item:handler",
          plannedSymbolIds: ["planned-symbol:handler"],
          plannedSymbols: [
            {
              plannedSymbolId: "planned-symbol:handler",
              taskId: "task:assembly",
              intendedIdentity: {
                relativePath: "./src\\handler.ts",
                kind: "function",
                qualifiedName: "handler",
              },
              responsibility: "Handle the request",
            },
          ],
          allowedPaths: ["./src\\handler.ts"],
        },
      ],
      governanceLinks: [
        {
          plannedSymbolId: "planned-symbol:handler",
          recordId: "decision:workflow",
          revisionId: "revision:local",
          origin: "curated",
        },
      ],
    });

    expect(state.normativeRecords.map((record) => record.origin)).toEqual(["local", "managed"]);
    expect(state.evidence).toEqual([
      {
        evidenceId: "evidence:local",
        text: "The user approved the local workflow.",
        availability: "current",
        allowedRuntimeProviders: ["codex"],
        provenance: {
          resourceId: "resource:local",
          chunkId: "chunk:local",
          resourceTitle: "Local decision",
          source: {
            connectorId: "codex",
            externalId: "codex://session/local",
            type: "session",
          },
          observedAt: "2026-08-28T00:00:00.000Z",
          contentHash: "sha256:local",
          ontologyNodeIds: ["engineering", "workflow"],
        },
      },
      {
        evidenceId: "evidence:managed",
        text: "",
        availability: "unavailable",
        allowedRuntimeProviders: ["codex"],
      },
    ]);
    expect(state.logicPlans[0]?.allowedPaths).toEqual(["src/handler.ts"]);
    expect(state.logicPlans[0]?.plannedSymbols?.[0]?.intendedIdentity.relativePath).toBe(
      "src/handler.ts",
    );
    expect(state.governanceLinks).toEqual([
      {
        plannedSymbolId: "planned-symbol:handler",
        recordId: "decision:workflow",
        revisionId: "revision:local",
        origin: "curated",
      },
    ]);
    expect(state.effectiveScopes).toContainEqual({ kind: "personal", subjectId: "user:owner" });
  });

  it("changes freshness when Evidence content changes and rejects escaping paths", () => {
    const evidence = {
      evidenceId: "evidence:local",
      text: "first",
      availability: "current" as const,
      allowedRuntimeProviders: ["codex"],
    };
    const logicPlan = {
      workItemId: "work-item:handler",
      plannedSymbolIds: ["planned-symbol:handler"],
      allowedPaths: ["src/handler.ts"],
    };
    const base = {
      taskId: "task:assembly",
      organizationId: "org:acme",
      codeRevision: "commit:1",
      localManifest: manifest("local"),
      evidence: [evidence],
      logicPlans: [logicPlan],
    };
    const first = assembleCurrentTaskContextState(base);
    const second = assembleCurrentTaskContextState({
      ...base,
      evidence: [{ ...evidence, text: "second" }],
    });
    const withProvenance = assembleCurrentTaskContextState({
      ...base,
      evidence: [
        {
          ...evidence,
          provenance: {
            resourceId: "resource:local",
            chunkId: "chunk:local",
            resourceTitle: "Local decision",
            source: { connectorId: "codex", externalId: "codex://session/1", type: "session" },
            observedAt: "2026-08-28T00:00:00.000Z",
            contentHash: "sha256:first",
            ontologyNodeIds: ["workflow"],
          },
        },
      ],
    });
    expect(second.sourceFreshnessDigest).not.toBe(first.sourceFreshnessDigest);
    expect(withProvenance.sourceFreshnessDigest).not.toBe(first.sourceFreshnessDigest);
    expect(() =>
      assembleCurrentTaskContextState({
        ...base,
        logicPlans: [{ ...logicPlan, allowedPaths: ["../outside.ts"] }],
      }),
    ).toThrow("stay inside the workspace");
    expect(() =>
      assembleCurrentTaskContextState({
        ...base,
        logicPlans: [
          {
            ...logicPlan,
            plannedSymbols: [
              {
                plannedSymbolId: "planned-symbol:other",
                taskId: "task:assembly",
                intendedIdentity: { qualifiedName: "handler" },
                responsibility: "Handle the request",
              },
            ],
          },
        ],
      }),
    ).toThrow("describe every Planned Symbol ID exactly once");
    expect(() =>
      assembleCurrentTaskContextState({
        ...base,
        governanceLinks: [
          {
            plannedSymbolId: "planned-symbol:unknown",
            recordId: "decision:workflow",
            revisionId: "revision:local",
            origin: "curated",
          },
        ],
      }),
    ).toThrow("unknown Planned Symbol");
  });

  it("does not let an inactive historical revision restrict current Evidence egress", () => {
    const activeManifest = manifest("local");
    const activeRevision = activeManifest.revisions[0];
    if (!activeRevision || activeRevision.kind !== "decision") {
      throw new Error("Test fixture requires an active Decision");
    }
    const withHistory: NormativeManifest = {
      ...activeManifest,
      revisions: [
        {
          ...activeRevision,
          revisionId: "revision:historical",
          egress: {
            dataClassification: "internal",
            allowedRuntimeProviders: ["historical-provider"],
          },
          statement: "Historical content that is no longer active.",
        },
        activeRevision,
      ],
    };
    const input = {
      taskId: "task:history",
      organizationId: "org:acme",
      codeRevision: "commit:1",
      evidence: [
        {
          evidenceId: "evidence:local",
          text: "current",
          availability: "current" as const,
          allowedRuntimeProviders: ["codex", "historical-provider"],
        },
      ],
      logicPlans: [
        {
          workItemId: "work-item:handler",
          plannedSymbolIds: ["planned-symbol:handler"],
          allowedPaths: ["src/handler.ts"],
        },
      ],
    };
    const withoutHistory = assembleCurrentTaskContextState({
      ...input,
      localManifest: activeManifest,
    });
    const historical = assembleCurrentTaskContextState({
      ...input,
      localManifest: withHistory,
    });

    expect(historical.evidence[0]?.allowedRuntimeProviders).toEqual(["codex"]);
    expect(historical.sourceFreshnessDigest).toBe(withoutHistory.sourceFreshnessDigest);
    expect(historical.normativeRevisionCatalog).toHaveLength(2);
    expect(() =>
      assembleCurrentTaskContextState({
        ...input,
        localManifest: withHistory,
        governanceLinks: [
          {
            plannedSymbolId: "planned-symbol:handler",
            recordId: activeRevision.recordId,
            revisionId: "revision:historical",
            origin: "curated",
          },
        ],
      }),
    ).toThrow("non-effective normative revision");
  });
});

describe("resolvePluginDataDirectory", () => {
  it("prefers the host-injected plugin directory and has an XDG fallback", () => {
    expect(
      resolvePluginDataDirectory({ PLUGIN_DATA: "/private/plugin" }, "/home/user", "linux"),
    ).toBe("/private/plugin");
    expect(resolvePluginDataDirectory({ XDG_DATA_HOME: "/data" }, "/home/user", "linux")).toBe(
      "/data/kontext-brain",
    );
  });
});

function manifest(layer: "local" | "managed"): NormativeManifest {
  const local = layer === "local";
  const revisionId = local ? "revision:local" : "revision:managed";
  const recordId = local ? "decision:workflow" : "invariant:verification";
  const scope = local
    ? ({ kind: "workspace", workspaceId: "workspace:local" } as const)
    : ({ kind: "organization", organizationId: "org:acme" } as const);
  return {
    schemaVersion: 1,
    organizationId: "org:acme",
    revisions: [
      local
        ? {
            kind: "decision",
            organizationId: "org:acme",
            recordId,
            revisionId,
            scope,
            evidence: [{ evidenceId: "evidence:local" }],
            egress: {
              dataClassification: "internal",
              allowedRuntimeProviders: ["codex", "claude"],
            },
            authoredBy: "user:owner",
            authoredAt: "2026-08-28T00:00:00.000Z",
            statement: "Use current Kontext context.",
          }
        : {
            kind: "invariant",
            organizationId: "org:acme",
            recordId,
            revisionId,
            scope,
            evidence: [{ evidenceId: "evidence:managed" }],
            egress: {
              dataClassification: "confidential",
              allowedRuntimeProviders: ["codex"],
            },
            authoredBy: "domain:owner",
            authoredAt: "2026-08-28T00:00:00.000Z",
            statement: "Verification must pass.",
            verifiers: [{ kind: "test", ref: "pnpm test" }],
          },
    ],
    activations: [
      {
        organizationId: "org:acme",
        kind: local ? "decision" : "invariant",
        recordId,
        revisionId,
        scope,
        state: local ? "accepted_local" : "accepted",
        acceptedBy: local ? "user:owner" : "domain:owner",
        acceptedAt: "2026-08-28T00:01:00.000Z",
        mergeCommit: local ? undefined : "commit:managed",
      },
    ],
  };
}
