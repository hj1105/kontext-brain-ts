import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import type { BenchmarkQuery, CorpusDocument, DatasetBundle } from "./contracts.js";
import { writePreparedDataset } from "./datasets.js";
import { readJsonLines } from "./jsonl.js";

export type BeirDatasetId = "beir-scifact" | "beir-nfcorpus";

export interface PreparedBeirDataset extends Omit<DatasetBundle, "id"> {
  readonly id: BeirDatasetId;
}

interface ExtractedBeirDocument {
  readonly _id?: unknown;
  readonly title?: unknown;
  readonly text?: unknown;
}

interface ExtractedBeirQuery {
  readonly _id?: unknown;
  readonly text?: unknown;
}

interface BeirDatasetSource {
  readonly archiveUrl: string;
  readonly license: string;
}

interface BeirTestQrels {
  readonly queryIds: ReadonlySet<string>;
  readonly positiveByQuery: ReadonlyMap<string, ReadonlySet<string>>;
}

const BEIR_SOURCES: Readonly<Record<BeirDatasetId, BeirDatasetSource>> = {
  "beir-scifact": {
    archiveUrl: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    license: "CC BY-NC 2.0 (verify upstream SciFact terms before redistribution)",
  },
  "beir-nfcorpus": {
    archiveUrl: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    license: "Dataset-specific upstream terms (BEIR has no unified redistribution license)",
  },
};

const UPSTREAM_FILES = ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"] as const;

export function loadExtractedBeirDataset(
  datasetId: BeirDatasetId,
  inputDirectory: string,
): PreparedBeirDataset {
  const documents = loadDocuments(datasetId, join(inputDirectory, "corpus.jsonl"));
  const documentById = uniqueById(documents, "document");
  const extractedQueries = readJsonLines<ExtractedBeirQuery>(join(inputDirectory, "queries.jsonl"));
  const queryIds = new Set(
    extractedQueries.map((query, index) =>
      requiredString(query._id, `queries.jsonl record ${index + 1} _id`),
    ),
  );
  if (queryIds.size !== extractedQueries.length) throw new Error("Duplicate BEIR query id");
  const testQrels = loadTestQrels(join(inputDirectory, "qrels", "test.tsv"));

  for (const queryId of testQrels.queryIds) {
    if (!queryIds.has(queryId)) {
      throw new Error(`Test qrel references unknown query ${queryId}`);
    }
    const documentIds = testQrels.positiveByQuery.get(queryId);
    if (!documentIds || documentIds.size === 0) {
      throw new Error(`Test qrel query ${queryId} has no positive evidence`);
    }
    for (const documentId of documentIds) {
      if (!documentById.has(documentId)) {
        throw new Error(`Positive qrel for ${queryId} references unknown document ${documentId}`);
      }
    }
  }

  const queries = extractedQueries
    .filter((query, index) =>
      testQrels.queryIds.has(requiredString(query._id, `queries.jsonl record ${index + 1} _id`)),
    )
    .map((query, index) => toBenchmarkQuery(datasetId, query, index, testQrels.positiveByQuery))
    .sort(compareById);
  const source = BEIR_SOURCES[datasetId];
  return {
    id: datasetId,
    track: "static-kb",
    documents: [...documents].sort(compareById),
    queries,
    provenance: {
      source: source.archiveUrl,
      version: upstreamVersion(inputDirectory),
      license: source.license,
    },
  };
}

export function prepareBeirDataset(
  datasetId: BeirDatasetId,
  inputDirectory: string,
  outputDirectory: string,
): PreparedBeirDataset {
  const bundle = loadExtractedBeirDataset(datasetId, inputDirectory);
  writePreparedDataset(outputDirectory, {
    ...bundle,
    id: bundle.id as DatasetBundle["id"],
  });
  return bundle;
}

function loadDocuments(datasetId: BeirDatasetId, path: string): CorpusDocument[] {
  return readJsonLines<ExtractedBeirDocument>(path).map((record, index) => {
    const id = requiredString(record._id, `corpus.jsonl record ${index + 1} _id`);
    const text = requiredString(record.text, `corpus.jsonl record ${index + 1} text`);
    const title =
      typeof record.title === "string" && record.title.trim() ? record.title.trim() : id;
    return {
      id,
      sourceId: id,
      title,
      text,
      metadata: { beirDataset: datasetId, upstreamId: id },
    };
  });
}

function toBenchmarkQuery(
  datasetId: BeirDatasetId,
  query: ExtractedBeirQuery,
  index: number,
  positiveQrels: ReadonlyMap<string, ReadonlySet<string>>,
): BenchmarkQuery {
  const id = requiredString(query._id, `queries.jsonl record ${index + 1} _id`);
  const text = requiredString(query.text, `queries.jsonl record ${index + 1} text`);
  const goldEvidenceIds = [...(positiveQrels.get(id) ?? [])].sort(compareStrings);
  return {
    id,
    text,
    referenceAnswer: null,
    goldEvidenceIds,
    goldEvidenceText: [],
    answerable: goldEvidenceIds.length > 0,
    category: "retrieval-only",
    metadata: {
      beirDataset: datasetId,
      split: "test",
      positiveQrels: goldEvidenceIds.length,
    },
  };
}

function loadTestQrels(path: string): BeirTestQrels {
  const lines = readFileSync(path, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim());
  const header = lines.shift()?.split("\t");
  if (!header) throw new Error(`Empty BEIR qrels file ${path}`);
  const queryIndex = header.indexOf("query-id");
  const documentIndex = header.indexOf("corpus-id");
  const scoreIndex = header.indexOf("score");
  if (queryIndex < 0 || documentIndex < 0 || scoreIndex < 0) {
    throw new Error(`BEIR qrels header must contain query-id, corpus-id, and score: ${path}`);
  }
  const queryIds = new Set<string>();
  const positiveByQuery = new Map<string, Set<string>>();
  for (const [index, line] of lines.entries()) {
    const columns = line.split("\t");
    const queryId = requiredString(columns[queryIndex], `${path}:${index + 2} query-id`);
    const documentId = requiredString(columns[documentIndex], `${path}:${index + 2} corpus-id`);
    const score = Number(columns[scoreIndex]);
    if (!Number.isFinite(score)) throw new Error(`Invalid qrel score at ${path}:${index + 2}`);
    queryIds.add(queryId);
    if (score <= 0) continue;
    const documentIds = positiveByQuery.get(queryId) ?? new Set<string>();
    documentIds.add(documentId);
    positiveByQuery.set(queryId, documentIds);
  }
  return { queryIds, positiveByQuery };
}

function upstreamVersion(inputDirectory: string): string {
  const hash = createHash("sha256");
  for (const relativePath of UPSTREAM_FILES) {
    const contents = readFileSync(join(inputDirectory, relativePath));
    hash.update(relativePath).update("\0").update(contents).update("\0");
  }
  return `sha256:${hash.digest("hex")}`;
}

function uniqueById(
  documents: readonly CorpusDocument[],
  recordType: string,
): Map<string, CorpusDocument> {
  const records = new Map<string, CorpusDocument>();
  for (const document of documents) {
    if (records.has(document.id)) throw new Error(`Duplicate BEIR ${recordType} id ${document.id}`);
    records.set(document.id, document);
  }
  return records;
}

function requiredString(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`Missing BEIR ${field}`);
  return value.trim();
}

function compareById(left: { readonly id: string }, right: { readonly id: string }): number {
  return compareStrings(left.id, right.id);
}

function compareStrings(left: string, right: string): number {
  if (left < right) return -1;
  if (left > right) return 1;
  return 0;
}

function runCli(argv: readonly string[]): void {
  const [datasetId, inputDirectory, outputDirectory] = argv;
  if (
    (datasetId !== "beir-scifact" && datasetId !== "beir-nfcorpus") ||
    !inputDirectory ||
    !outputDirectory
  ) {
    throw new Error(
      "Usage: prepare-beir.ts <beir-scifact|beir-nfcorpus> <extracted-directory> <output-directory>",
    );
  }
  const bundle = prepareBeirDataset(datasetId, inputDirectory, outputDirectory);
  process.stdout.write(
    `${JSON.stringify({
      datasetId: bundle.id,
      documents: bundle.documents.length,
      queries: bundle.queries.length,
      outputDirectory: resolve(outputDirectory),
      version: bundle.provenance.version,
    })}\n`,
  );
}

const entryPoint = process.argv[1];
if (entryPoint && import.meta.url === pathToFileURL(resolve(entryPoint)).href) {
  runCli(process.argv.slice(2));
}
