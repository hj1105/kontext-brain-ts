import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { loadExtractedBeirDataset, prepareBeirDataset } from "./prepare-beir.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { force: true, recursive: true });
  }
});

describe("BEIR dataset preparation", () => {
  it("maps the extracted test split into deterministic retrieval-only records", () => {
    const inputDirectory = createExtractedDataset({
      corpus: [
        { _id: "doc-b", title: "", text: "Beta evidence." },
        { _id: "doc-a", title: "Alpha", text: "Alpha evidence." },
      ],
      queries: [
        { _id: "q-b", text: "Where is beta?" },
        { _id: "q-a", text: "Where is alpha?" },
        { _id: "q-c", text: "Where is gamma?" },
      ],
      qrels: [
        "query-id\tcorpus-id\tscore",
        "q-a\tdoc-b\t0",
        "q-a\tdoc-a\t1",
        "q-a\tdoc-a\t2",
        "q-a\tdoc-b\t-1",
        "q-b\tdoc-b\t3",
      ].join("\n"),
    });

    const bundle = loadExtractedBeirDataset("beir-scifact", inputDirectory);

    expect(bundle.id).toBe("beir-scifact");
    expect(bundle.track).toBe("static-kb");
    expect(bundle.documents).toEqual([
      {
        id: "doc-a",
        sourceId: "doc-a",
        title: "Alpha",
        text: "Alpha evidence.",
        metadata: { beirDataset: "beir-scifact", upstreamId: "doc-a" },
      },
      {
        id: "doc-b",
        sourceId: "doc-b",
        title: "doc-b",
        text: "Beta evidence.",
        metadata: { beirDataset: "beir-scifact", upstreamId: "doc-b" },
      },
    ]);
    expect(bundle.queries).toEqual([
      {
        id: "q-a",
        text: "Where is alpha?",
        referenceAnswer: null,
        goldEvidenceIds: ["doc-a"],
        goldEvidenceText: [],
        answerable: true,
        category: "retrieval-only",
        metadata: { beirDataset: "beir-scifact", split: "test", positiveQrels: 1 },
      },
      {
        id: "q-b",
        text: "Where is beta?",
        referenceAnswer: null,
        goldEvidenceIds: ["doc-b"],
        goldEvidenceText: [],
        answerable: true,
        category: "retrieval-only",
        metadata: { beirDataset: "beir-scifact", split: "test", positiveQrels: 1 },
      },
    ]);
    expect(bundle.provenance.source).toBe(
      "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    );
    expect(bundle.provenance.version).toMatch(/^sha256:[0-9a-f]{64}$/);
  });

  it("writes canonical files and fingerprints the exact upstream inputs", () => {
    const inputDirectory = createExtractedDataset({
      corpus: [{ _id: "doc-1", title: "One", text: "Evidence one." }],
      queries: [{ _id: "q-1", text: "Question one?" }],
      qrels: "query-id\tcorpus-id\tscore\nq-1\tdoc-1\t1",
    });
    const outputDirectory = createTemporaryDirectory();

    const prepared = prepareBeirDataset("beir-nfcorpus", inputDirectory, outputDirectory);
    const metadata = JSON.parse(
      readFileSync(join(outputDirectory, "dataset.json"), "utf8"),
    ) as Record<string, unknown>;

    expect(metadata).toMatchObject({
      id: "beir-nfcorpus",
      track: "static-kb",
      source: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
      version: prepared.provenance.version,
    });
    expect(readFileSync(join(outputDirectory, "corpus.jsonl"), "utf8")).toContain('"id":"doc-1"');
    expect(readFileSync(join(outputDirectory, "queries.jsonl"), "utf8")).toContain(
      '"referenceAnswer":null',
    );

    writeFileSync(
      join(inputDirectory, "queries.jsonl"),
      `${JSON.stringify({ _id: "q-1", text: "Changed question?" })}\n`,
    );
    const changed = loadExtractedBeirDataset("beir-nfcorpus", inputDirectory);
    expect(changed.provenance.version).not.toBe(prepared.provenance.version);
  });

  it("fails closed when a positive qrel references a missing document", () => {
    const inputDirectory = createExtractedDataset({
      corpus: [{ _id: "doc-1", title: "One", text: "Evidence one." }],
      queries: [{ _id: "q-1", text: "Question one?" }],
      qrels: "query-id\tcorpus-id\tscore\nq-1\tmissing\t1",
    });

    expect(() => loadExtractedBeirDataset("beir-scifact", inputDirectory)).toThrow(
      "Positive qrel for q-1 references unknown document missing",
    );
  });

  it("requires every query in test qrels to have positive evidence", () => {
    const inputDirectory = createExtractedDataset({
      corpus: [{ _id: "doc-1", title: "One", text: "Evidence one." }],
      queries: [{ _id: "q-1", text: "Question one?" }],
      qrels: "query-id\tcorpus-id\tscore\nq-1\tdoc-1\t0",
    });

    expect(() => loadExtractedBeirDataset("beir-scifact", inputDirectory)).toThrow(
      "Test qrel query q-1 has no positive evidence",
    );
  });
});

function createExtractedDataset(fixture: {
  readonly corpus: readonly unknown[];
  readonly queries: readonly unknown[];
  readonly qrels: string;
}): string {
  const directory = createTemporaryDirectory();
  mkdirSync(join(directory, "qrels"));
  writeFileSync(
    join(directory, "corpus.jsonl"),
    `${fixture.corpus.map((record) => JSON.stringify(record)).join("\n")}\n`,
  );
  writeFileSync(
    join(directory, "queries.jsonl"),
    `${fixture.queries.map((record) => JSON.stringify(record)).join("\n")}\n`,
  );
  writeFileSync(join(directory, "qrels", "test.tsv"), `${fixture.qrels}\n`);
  return directory;
}

function createTemporaryDirectory(): string {
  const directory = mkdtempSync(join(tmpdir(), "prepare-beir-"));
  temporaryDirectories.push(directory);
  return directory;
}
