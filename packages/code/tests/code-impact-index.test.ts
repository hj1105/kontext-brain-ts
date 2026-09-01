import { describe, expect, it } from "vitest";
import { CodeImpactIndex, TypeScriptCodeProvider, compareCodeAnalyses } from "../src/index.js";

const provider = new TypeScriptCodeProvider();
const files = [
  {
    path: "src/helper.ts",
    content: "export function helper(value: number) { return value; }",
  },
  {
    path: "src/service.ts",
    content: `
      import { helper } from "./helper.js";
      export function run() { return helper(1); }
    `,
  },
] as const;

describe("CodeImpactIndex", () => {
  it("finds direct and transitive reverse dependencies across project files", () => {
    const helperAnalysis = provider.analyze({
      codebaseId: "codebase:example",
      targetPath: "src/helper.ts",
      files,
    });
    const serviceAnalysis = provider.analyze({
      codebaseId: "codebase:example",
      targetPath: "src/service.ts",
      files,
    });
    const index = new CodeImpactIndex([helperAnalysis, serviceAnalysis]);
    const helper = helperAnalysis.symbols.find(
      (symbol) => symbol.identity.qualifiedName === "helper",
    );
    const run = serviceAnalysis.symbols.find((symbol) => symbol.identity.qualifiedName === "run");
    const helperModule = helperAnalysis.symbols.find((symbol) => symbol.identity.kind === "module");
    const serviceModule = serviceAnalysis.symbols.find(
      (symbol) => symbol.identity.kind === "module",
    );

    expect(index.findDirectDependents(helper?.symbolId ?? "")).toContainEqual(
      expect.objectContaining({
        dependentSymbolId: run?.symbolId,
        predicate: "calls",
      }),
    );
    expect(index.findDirectDependents(helperModule?.symbolId ?? "")).toContainEqual(
      expect.objectContaining({
        dependentSymbolId: serviceModule?.symbolId,
        predicate: "imports",
      }),
    );
    expect(
      index
        .findAffectedSymbols([helper?.symbolId ?? ""])
        .affectedSymbols.map((symbol) => symbol.identity.qualifiedName),
    ).toEqual(["helper", "run"]);
  });

  it("reports unknown symbol ids instead of claiming a complete impact result", () => {
    const analysis = provider.analyze({
      codebaseId: "codebase:example",
      targetPath: "src/helper.ts",
      files,
    });
    const result = new CodeImpactIndex([analysis]).findAffectedSymbols(["missing"]);

    expect(result.affectedSymbols).toEqual([]);
    expect(result.missingSymbolIds).toEqual(["missing"]);
  });
});

describe("compareCodeAnalyses", () => {
  it("ignores formatting changes and identifies behavior changes by stable symbol id", () => {
    const before = provider.analyze({
      codebaseId: "codebase:example",
      targetPath: "src/value.ts",
      files: [
        {
          path: "src/value.ts",
          content: "export function value(){return 1}",
        },
      ],
    });
    const formatted = provider.analyze({
      codebaseId: "codebase:example",
      targetPath: "src/value.ts",
      files: [
        {
          path: "src/value.ts",
          content: "export function value() {\n  return 1;\n}\n",
        },
      ],
    });
    const changed = provider.analyze({
      codebaseId: "codebase:example",
      targetPath: "src/value.ts",
      files: [
        {
          path: "src/value.ts",
          content: "export function value() { return 2; }",
        },
      ],
    });

    expect(compareCodeAnalyses(before, formatted)).toEqual([]);
    expect(compareCodeAnalyses(before, changed)).toEqual([
      expect.objectContaining({
        kind: "modified",
        symbolId: before.symbols.find((symbol) => symbol.identity.qualifiedName === "value")
          ?.symbolId,
      }),
    ]);
  });
});
