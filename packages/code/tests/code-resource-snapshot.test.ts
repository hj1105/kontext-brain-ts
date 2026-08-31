import { describe, expect, it } from "vitest";
import {
  CodeKnowledgeSynchronizer,
  type CodeResourceSnapshot,
  CodeResourceSnapshotAdapter,
  TypeScriptCodeProvider,
} from "../src/index.js";

const provider = new TypeScriptCodeProvider();
const adapter = new CodeResourceSnapshotAdapter();

describe("CodeResourceSnapshotAdapter", () => {
  it("maps Code Symbols and relationships to evidence-backed ResourceSnapshot data", () => {
    const analysis = provider.analyze({
      codebaseId: "codebase:example",
      targetPath: "src/example.ts",
      files: [
        {
          path: "src/example.ts",
          content: `
            function privateHelper() { return 1; }
            export function publicApi() { return privateHelper(); }
          `,
        },
      ],
    });

    const snapshot = adapter.normalize({
      organizationId: "org:acme",
      analysis,
      acl: { organizationWide: true },
      ontologyNodeIds: ["product"],
    });

    expect(snapshot.source).toEqual({
      connectorId: "code",
      externalId: "codebase:example:src/example.ts",
      type: "typescript-module",
    });
    expect(snapshot.chunks.map((chunk) => chunk.id)).toEqual(
      analysis.symbols.map((symbol) => symbol.sourceChunkId),
    );
    expect(snapshot.entities).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "privateHelper", scope: "resource" }),
        expect.objectContaining({
          name: "publicApi",
          scope: "global",
          promotionEvidence: "deterministic",
        }),
      ]),
    );
    expect(snapshot.facts).toContainEqual(
      expect.objectContaining({
        predicate: "calls",
        evidenceChunkIds: [
          analysis.symbols.find((symbol) => symbol.identity.qualifiedName === "publicApi")
            ?.sourceChunkId,
        ],
      }),
    );
  });

  it("sends the normalized snapshot through the existing Resource sync-shaped port", async () => {
    const received: CodeResourceSnapshot[] = [];
    const synchronizer = new CodeKnowledgeSynchronizer(
      {
        async execute(snapshot) {
          received.push(snapshot);
          return { resourceId: "resource:1", changed: true, affectedFactKeys: [] };
        },
        async remove() {
          return true;
        },
      },
      provider,
      adapter,
    );

    const result = await synchronizer.sync({
      organizationId: "org:acme",
      codebaseId: "codebase:example",
      targetPath: "src/example.ts",
      files: [{ path: "src/example.ts", content: "export const answer = () => 42;" }],
      acl: { organizationWide: true },
      ontologyNodeIds: ["product"],
    });

    expect(result.changed).toBe(true);
    expect(received).toHaveLength(1);
    expect(received[0]?.facts?.every((fact) => fact.evidenceChunkIds.length > 0)).toBe(true);
  });
});
