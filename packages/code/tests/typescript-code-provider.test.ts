import { describe, expect, it } from "vitest";
import { TypeScriptCodeProvider } from "../src/index.js";

const provider = new TypeScriptCodeProvider();

describe("TypeScriptCodeProvider", () => {
  it("keeps behavior-bearing symbol identity and content stable across formatting-only edits", () => {
    const compact = provider.analyze({
      codebaseId: "codebase:example",
      targetPath: "src/math.ts",
      files: [
        { path: "src/math.ts", content: "export function add(a:number,b:number){return a+b}" },
      ],
    });
    const formatted = provider.analyze({
      codebaseId: "codebase:example",
      targetPath: "src/math.ts",
      files: [
        {
          path: "src/math.ts",
          content: "export function add(a: number, b: number) {\n  return a + b;\n}\n",
        },
      ],
    });

    const compactAdd = compact.symbols.find((symbol) => symbol.identity.qualifiedName === "add");
    const formattedAdd = formatted.symbols.find(
      (symbol) => symbol.identity.qualifiedName === "add",
    );

    expect(compactAdd).toMatchObject({
      behaviorBearing: true,
      exported: true,
      identity: { kind: "function" },
    });
    expect(formattedAdd?.symbolId).toBe(compactAdd?.symbolId);
    expect(formattedAdd?.contentHash).toBe(compactAdd?.contentHash);
    expect(formatted.contentHash).toBe(compact.contentHash);
  });

  it("uses behavior-bearing declarations as logic units and attributes callbacks to their parent", () => {
    const analysis = provider.analyze({
      codebaseId: "codebase:example",
      targetPath: "src/counter.ts",
      files: [
        {
          path: "src/counter.ts",
          content: `
            export class Counter {
              constructor(private current = 0) {}
              get value() { return this.current; }
              increment() { return [1].map((step) => this.current += step)[0]; }
            }
            export const createCounter = () => new Counter();
          `,
        },
      ],
    });

    const behavior = analysis.symbols
      .filter((symbol) => symbol.behaviorBearing)
      .map((symbol) => `${symbol.identity.kind}:${symbol.identity.qualifiedName}`);

    expect(behavior).toEqual([
      "constructor:Counter.constructor",
      "getter:Counter.value",
      "method:Counter.increment",
      "named_arrow:createCounter",
    ]);
    expect(analysis.symbols.some((symbol) => symbol.identity.qualifiedName.includes("step"))).toBe(
      false,
    );
  });

  it("extracts only resolved semantic relationships and records unresolved calls separately", () => {
    const analysis = provider.analyze({
      codebaseId: "codebase:example",
      targetPath: "src/service.ts",
      files: [
        {
          path: "src/helper.ts",
          content: "export function helper(): string { return 'ok'; }",
        },
        {
          path: "src/service.ts",
          content: `
            import { helper } from "./helper.js";
            export function run(dynamic: unknown) {
              const result = helper();
              if (process.env.API_URL) return result;
              return (dynamic as { execute?: () => string }).execute?.();
            }
          `,
        },
      ],
    });

    const run = analysis.symbols.find((symbol) => symbol.identity.qualifiedName === "run");
    expect(analysis.relationships).toContainEqual(
      expect.objectContaining({
        predicate: "calls",
        subjectSymbolId: run?.symbolId,
        object: expect.objectContaining({ kind: "symbol", qualifiedName: "helper" }),
      }),
    );
    expect(analysis.relationships).toContainEqual(
      expect.objectContaining({
        predicate: "reads_env",
        subjectSymbolId: run?.symbolId,
        object: { kind: "literal", value: "API_URL" },
      }),
    );
    expect(analysis.unresolvedRelationships).toContainEqual(
      expect.objectContaining({
        predicate: "calls",
        expression: expect.stringContaining("execute"),
      }),
    );
  });
});
