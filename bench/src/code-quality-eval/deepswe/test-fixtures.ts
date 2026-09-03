import type { DeepSweContextCorpus } from "./contracts.js";
import { sha256 } from "./corpus.js";

export function fixtureCorpus(
  taskId = "demo",
  baseCodeRevision = "a".repeat(40),
): DeepSweContextCorpus {
  const text = "The parser must preserve stable ordering for equal typed values.";
  return {
    schemaVersion: 1,
    taskId,
    organizationId: "organization:fixture",
    baseCodeRevision,
    contextDigest: "sha256:fixture-context",
    sourceFreshnessDigest: "sha256:fixture-freshness",
    snapshotAt: "2026-01-02T00:00:00.000Z",
    generator: { name: "kontext-brain", revision: "ontology-fixture-v1" },
    evidence: [
      {
        evidenceId: "evidence:design",
        resourceId: "resource:design",
        chunkId: "chunk:parser-ordering",
        title: "Parser design",
        text,
        sourceUri: "https://example.invalid/design/parser-ordering",
        observedAt: "2026-01-01T00:00:00.000Z",
        contentSha256: sha256(text),
        ontologyNodeIds: ["resource:design"],
      },
    ],
    normativeRecords: [
      {
        revision: {
          kind: "invariant",
          organizationId: "organization:fixture",
          recordId: "invariant:stable-order",
          revisionId: "revision:1",
          scope: { kind: "codebase", codebaseId: "codebase:fixture" },
          evidence: [{ evidenceId: "evidence:design", sourceSpan: "parser ordering" }],
          egress: {
            dataClassification: "public",
            allowedRuntimeProviders: ["openai"],
          },
          authoredBy: "user:fixture",
          authoredAt: "2026-01-01T00:00:00.000Z",
          statement: "Parser.equal_values must preserve stable ordering.",
          verifiers: [{ kind: "test", ref: "tests/test_parser.py" }],
        },
        symbolSelectors: [{ relativePath: "src/parser.py", qualifiedName: "Parser.equal_values" }],
      },
    ],
  };
}
