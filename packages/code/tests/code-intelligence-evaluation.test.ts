import { describe, expect, it } from "vitest";
import {
  TypeScriptCodeProvider,
  evaluateCodeIdentityStability,
  evaluateCodeRelationshipExtraction,
} from "../src/index.js";

const provider = new TypeScriptCodeProvider();

describe("Code intelligence evaluation gates", () => {
  it("measures labelled semantic relationship precision without implementation ids", () => {
    const files = [
      {
        path: "src/helper.ts",
        content: "export function helper(value: number) { return value; }",
      },
      {
        path: "src/service.ts",
        content:
          'import { helper } from "./helper.js"; export function run() { return helper(1); }',
      },
    ];
    const analyses = files.map((file) =>
      provider.analyze({
        codebaseId: "codebase:example",
        targetPath: file.path,
        files,
      }),
    );

    expect(
      evaluateCodeRelationshipExtraction(
        analyses,
        [
          {
            subject: { relativePath: "src/service.ts", qualifiedName: "<module>" },
            predicate: "imports",
            object: {
              kind: "symbol",
              relativePath: "src/helper.ts",
              qualifiedName: "<module>",
            },
          },
          {
            subject: { relativePath: "src/service.ts", qualifiedName: "run" },
            predicate: "calls",
            object: {
              kind: "symbol",
              relativePath: "src/helper.ts",
              qualifiedName: "helper",
            },
          },
        ],
        ["imports", "calls"],
      ),
    ).toEqual({
      truePositives: 2,
      falsePositives: 0,
      falseNegatives: 0,
      precision: 1,
      recall: 1,
    });
  });

  it("measures behavior identity and content stability across format-only edits", () => {
    const analyze = (content: string) => [
      provider.analyze({
        codebaseId: "codebase:example",
        targetPath: "src/math.ts",
        files: [{ path: "src/math.ts", content }],
      }),
    ];

    expect(
      evaluateCodeIdentityStability(
        analyze("export function add(a:number,b:number){return a+b}"),
        analyze("export function add(a: number, b: number) {\n return a + b;\n}\n"),
      ),
    ).toEqual({
      comparableBehaviorSymbols: 1,
      stableSymbolIds: 1,
      stableContentHashes: 1,
      identityStability: 1,
      contentStability: 1,
    });
  });
});
