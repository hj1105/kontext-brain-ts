import { describe, expect, it } from "vitest";
import { buildContextBundle, stableJson, validateCorpus } from "./corpus.js";
import { fixtureCorpus } from "./test-fixtures.js";

describe("DeepSWE context corpus", () => {
  it("keeps the raw corpus identity fixed while changing only arm projection", () => {
    const corpus = fixtureCorpus();
    const baseline = buildContextBundle("baseline", corpus);
    const rag = buildContextBundle("rag", corpus);
    const kontext = buildContextBundle("kontext", corpus);

    expect(new Set([baseline.corpusSha256, rag.corpusSha256, kontext.corpusSha256]).size).toBe(1);
    expect(baseline.evidence).toEqual([]);
    expect(baseline.normativeRecords).toEqual([]);
    expect(rag.evidence).toEqual(corpus.evidence);
    expect(rag.normativeRecords).toEqual([]);
    expect(kontext.evidence).toEqual(corpus.evidence);
    expect(kontext.normativeRecords).toEqual(corpus.normativeRecords);
    expect(
      new Set([baseline.projectionSha256, rag.projectionSha256, kontext.projectionSha256]).size,
    ).toBe(3);
  });

  it("rejects post-snapshot evidence and ungrounded decisions", () => {
    const corpus = fixtureCorpus();
    const evidence = required(corpus.evidence[0]);
    const record = required(corpus.normativeRecords[0]);
    expect(() =>
      validateCorpus(
        {
          ...corpus,
          evidence: [{ ...evidence, observedAt: "2026-01-03T00:00:00.000Z" }],
        },
        corpus.taskId,
        "/tmp/deep-swe/tasks/demo",
      ),
    ).toThrow(/newer than the snapshot/);
    expect(() =>
      validateCorpus(
        {
          ...corpus,
          normativeRecords: [
            {
              ...record,
              revision: { ...record.revision, evidence: [{ evidenceId: "missing" }] },
            },
          ],
        },
        corpus.taskId,
        "/tmp/deep-swe/tasks/demo",
      ),
    ).toThrow(/unknown Evidence/);
  });

  it("rejects benchmark solution and verifier provenance from files or URLs", () => {
    const corpus = fixtureCorpus();
    const evidence = required(corpus.evidence[0]);
    for (const sourceUri of [
      "file:///tmp/deep-swe/tasks/demo/solution/solve.py",
      "https://github.com/datacurve-ai/deep-swe/blob/main/tasks/demo/verifier/test.sh",
    ]) {
      expect(() =>
        validateCorpus(
          {
            ...corpus,
            evidence: [{ ...evidence, sourceUri }],
          },
          corpus.taskId,
          "/tmp/deep-swe/tasks/demo",
        ),
      ).toThrow(/Forbidden benchmark artifact/);
    }
  });

  it("serializes objects deterministically before hashing", () => {
    expect(stableJson({ z: 1, a: { y: 2, x: 3 } })).toBe('{"a":{"x":3,"y":2},"z":1}');
  });
});

function required<T>(value: T | undefined): T {
  if (value === undefined) throw new Error("Missing test fixture value");
  return value;
}
