import type { DeepSweContextCorpus } from "./contracts.js";
import { sha256 } from "./corpus.js";

export function fixtureCorpus(taskId = "demo"): DeepSweContextCorpus {
  const body = "The parser must preserve stable ordering for equal typed values.";
  return {
    schemaVersion: 1,
    taskId,
    snapshotAt: "2026-01-02T00:00:00.000Z",
    generator: { name: "kontext-brain", revision: "ontology-fixture-v1" },
    documents: [
      {
        documentId: "doc:design",
        title: "Parser design",
        body,
        sourceUri: "https://example.invalid/design/parser-ordering",
        observedAt: "2026-01-01T00:00:00.000Z",
        contentSha256: sha256(body),
        ontologyNodeIds: ["resource:design"],
      },
    ],
    normativeRecords: [
      {
        kind: "invariant",
        recordId: "invariant:stable-order",
        revisionId: "revision:1",
        text: "Parser.equal_values must preserve stable ordering.",
        evidenceIds: ["doc:design"],
        ontologyNodeIds: ["record:stable-order"],
        symbolSelectors: [{ relativePath: "src/parser.py", qualifiedName: "Parser.equal_values" }],
      },
    ],
  };
}
