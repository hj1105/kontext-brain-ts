import { rm } from "node:fs/promises";
import { afterEach, describe, expect, it } from "vitest";
import { allRules } from "./rules.js";
import { buildLargeScaleStateAssembly } from "./state.js";
import { createLargeScaleWorkspace } from "./workspace.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories
      .splice(0)
      .map((directory) => rm(directory, { recursive: true, force: true })),
  );
});

describe("large-scale sidecar state", () => {
  it("derives narrow governance links through ontology nodes", async () => {
    const workspace = await createLargeScaleWorkspace();
    temporaryDirectories.push(workspace.workspacePath);
    const state = await buildLargeScaleStateAssembly({ workspace, runtime: "codex" });
    const assembly = state.assembly as {
      readonly localManifest: { readonly revisions: readonly unknown[] };
      readonly logicPlans: readonly { readonly plannedSymbolIds: readonly string[] }[];
      readonly governanceLinks: readonly {
        plannedSymbolId: string;
        recordId: string;
        revisionId: string;
      }[];
    };

    expect(assembly.localManifest.revisions).toHaveLength(allRules().length);
    expect(assembly.logicPlans).toHaveLength(workspace.repository.governedNames.length);
    expect(assembly.governanceLinks).toHaveLength(workspace.repository.governedNames.length * 3);
    expect(new Set(assembly.governanceLinks.map((link) => link.plannedSymbolId)).size).toBe(
      workspace.repository.governedNames.length,
    );
    expect(state.governingRecordIds).toEqual([
      "decision:billing-retry-ceiling",
      "domain-term:recovery-ceiling",
      "invariant:billing-retry-bounded",
    ]);
    expect(
      assembly.governanceLinks.every((link) => state.governingRecordIds.includes(link.recordId)),
    ).toBe(true);
  });
});
