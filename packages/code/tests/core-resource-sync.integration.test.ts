import { describe, expect, it } from "vitest";
import {
  InMemoryKnowledgeGraphRepository,
  InMemoryResourceContentStore,
  SyncResourceUseCase,
} from "../../core/src/index.js";
import {
  CodeKnowledgeSynchronizer,
  CodeResourceSnapshotAdapter,
  TypeScriptCodeProvider,
} from "../src/index.js";

describe("CodeKnowledgeSynchronizer core integration", () => {
  it("persists evidence-backed relationships and ignores formatting-only changes", async () => {
    const organizationId = "org:acme";
    const repository = new InMemoryKnowledgeGraphRepository();
    const contentStore = new InMemoryResourceContentStore();
    const synchronizer = new CodeKnowledgeSynchronizer(
      new SyncResourceUseCase(repository, contentStore),
      new TypeScriptCodeProvider(),
      new CodeResourceSnapshotAdapter(),
    );
    const common = {
      organizationId,
      codebaseId: "codebase:example",
      targetPath: "src/math.ts",
      acl: { organizationWide: true },
      ontologyNodeIds: ["product"],
    } as const;

    const first = await synchronizer.sync({
      ...common,
      files: [
        {
          path: "src/math.ts",
          content:
            "function normalize(value:number){return value} export function add(a:number,b:number){return normalize(a+b)}",
        },
      ],
    });
    const formattingOnly = await synchronizer.sync({
      ...common,
      files: [
        {
          path: "src/math.ts",
          content: `
            function normalize(value: number) {
              return value;
            }

            export function add(a: number, b: number) {
              return normalize(a + b);
            }
          `,
        },
      ],
    });

    expect(first.changed).toBe(true);
    expect(formattingOnly.changed).toBe(false);
    expect(contentStore.putCount).toBe(1);

    const callFact = (await repository.listFacts(organizationId)).find(
      (fact) => fact.predicate === "calls",
    );
    expect(callFact).toMatchObject({
      status: "active",
      object: { kind: "entity" },
    });
    expect(
      callFact ? await repository.listEvidenceForFact(organizationId, callFact.factKey) : [],
    ).toEqual([expect.objectContaining({ origin: "derived", status: "active" })]);
  });
});
