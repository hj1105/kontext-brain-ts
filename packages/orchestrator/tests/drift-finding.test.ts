import {
  CodeImpactIndex,
  TypeScriptCodeProvider,
  createCodeSymbolOntologyLink,
} from "@kontext-brain/code";
import { describe, expect, it } from "vitest";
import { createDriftFindings, isDriftFindingValid } from "../src/index.js";

const provider = new TypeScriptCodeProvider();
const files = [
  { path: "src/helper.ts", content: "export function helper() { return 1; }" },
  {
    path: "src/service.ts",
    content: 'import { helper } from "./helper.js"; export function run() { return helper(); }',
  },
];
const helperAnalysis = provider.analyze({
  codebaseId: "codebase:drift",
  targetPath: "src/helper.ts",
  files,
});
const serviceAnalysis = provider.analyze({
  codebaseId: "codebase:drift",
  targetPath: "src/service.ts",
  files,
});
const helper = helperAnalysis.symbols.find((symbol) => symbol.identity.qualifiedName === "helper");
if (!helper) throw new Error("expected helper symbol");
const from = {
  kind: "decision" as const,
  recordId: "decision:calculation",
  revisionId: "decision:calculation@1",
};
const to = { ...from, revisionId: "decision:calculation@2" };

describe("Drift Finding", () => {
  it("follows curated normative bindings through reverse code dependencies", () => {
    const binding = createCodeSymbolOntologyLink({
      symbolId: helper.symbolId,
      target: {
        kind: "normative",
        normativeKind: from.kind,
        recordId: from.recordId,
        revisionId: from.revisionId,
      },
      origin: "curated",
      evidenceIds: ["evidence:decision"],
      createdAt: "2026-08-28T10:00:00.000Z",
    });
    const [finding] = createDriftFindings({
      from,
      to,
      codeRevision: "commit:current",
      links: [binding],
      impactIndex: new CodeImpactIndex([helperAnalysis, serviceAnalysis]),
      createdAt: "2026-08-28T10:05:00.000Z",
    });

    expect(finding?.affectedSymbolIds).toEqual(
      expect.arrayContaining([
        helper.symbolId,
        serviceAnalysis.symbols.find((symbol) => symbol.identity.qualifiedName === "run")?.symbolId,
      ]),
    );
    if (!finding) throw new Error("expected Drift Finding");
    expect(isDriftFindingValid(finding)).toBe(true);
  });

  it("does not enforce proposed links and retains unresolved bound symbols", () => {
    const proposed = createCodeSymbolOntologyLink({
      symbolId: helper.symbolId,
      target: {
        kind: "normative",
        normativeKind: from.kind,
        recordId: from.recordId,
        revisionId: from.revisionId,
      },
      origin: "proposed",
      evidenceIds: ["evidence:similarity"],
      createdAt: "2026-08-28T10:00:00.000Z",
    });
    expect(
      createDriftFindings({
        from,
        to,
        codeRevision: "commit:current",
        links: [proposed],
        impactIndex: new CodeImpactIndex([helperAnalysis, serviceAnalysis]),
        createdAt: "2026-08-28T10:05:00.000Z",
      }),
    ).toEqual([]);

    const missing = createCodeSymbolOntologyLink({
      symbolId: "symbol:missing",
      target: proposed.target,
      origin: "deterministic",
      evidenceIds: proposed.evidenceIds,
      createdAt: proposed.createdAt,
    });
    const [finding] = createDriftFindings({
      from,
      to,
      codeRevision: "commit:current",
      links: [missing],
      impactIndex: new CodeImpactIndex([helperAnalysis, serviceAnalysis]),
      createdAt: "2026-08-28T10:05:00.000Z",
    });
    expect(finding?.unresolvedSymbolIds).toEqual(["symbol:missing"]);
  });
});
