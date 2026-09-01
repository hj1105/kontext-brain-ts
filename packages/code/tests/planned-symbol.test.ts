import { describe, expect, it } from "vitest";
import type { CodeSymbolRecord, PlannedSymbolRecord } from "../src/index.js";
import { resolvePlannedSymbols } from "../src/index.js";

describe("resolvePlannedSymbols", () => {
  it("binds a new Planned Symbol only when its intended identity has one exact match", () => {
    const symbols = [symbol("code-symbol:handler", "src/handler.ts", "handler")];
    const planned: PlannedSymbolRecord[] = [
      {
        plannedSymbolId: "planned-symbol:handler",
        taskId: "task:handler",
        intendedIdentity: {
          relativePath: "./src/handler.ts",
          kind: "function",
          qualifiedName: "handler",
        },
        responsibility: "Handle one request",
      },
    ];

    expect(resolvePlannedSymbols(planned, symbols)).toEqual({
      bindings: [
        {
          plannedSymbolId: "planned-symbol:handler",
          symbolId: "code-symbol:handler",
          boundBy: "intended_identity",
        },
      ],
      issues: [],
    });
  });

  it("fails closed when an intended identity is ambiguous or a recorded binding disappeared", () => {
    const symbols = [
      symbol("code-symbol:a", "src/a.ts", "handler"),
      symbol("code-symbol:b", "src/b.ts", "handler"),
    ];
    const planned: PlannedSymbolRecord[] = [
      {
        plannedSymbolId: "planned-symbol:ambiguous",
        taskId: "task:handler",
        intendedIdentity: { qualifiedName: "handler" },
        responsibility: "Handle one request",
      },
      {
        plannedSymbolId: "planned-symbol:removed",
        taskId: "task:handler",
        intendedIdentity: {},
        responsibility: "Removed behavior",
        boundSymbolId: "code-symbol:removed",
      },
    ];

    expect(resolvePlannedSymbols(planned, symbols).issues).toEqual([
      {
        plannedSymbolId: "planned-symbol:ambiguous",
        code: "identity_ambiguous",
        candidateSymbolIds: ["code-symbol:a", "code-symbol:b"],
      },
      {
        plannedSymbolId: "planned-symbol:removed",
        code: "bound_symbol_missing",
        candidateSymbolIds: [],
      },
    ]);
  });
});

function symbol(symbolId: string, relativePath: string, qualifiedName: string): CodeSymbolRecord {
  return {
    symbolId,
    sourceChunkId: `chunk:${symbolId}`,
    identity: {
      codebaseId: "codebase:test",
      relativePath,
      language: "typescript",
      kind: "function",
      qualifiedName,
      signatureDiscriminator: "()",
    },
    behaviorBearing: true,
    exported: true,
    signature: "(): number",
    contentHash: `sha256:${symbolId}`,
    text: "return 1",
    position: 1,
    semanticSupport: "certified",
  };
}
