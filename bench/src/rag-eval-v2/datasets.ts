import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type {
  BenchmarkQuery,
  CorpusDocument,
  DatasetBundle,
  DatasetDoctorResult,
  DatasetId,
  DatasetTrack,
} from "./contracts.js";
import { readJsonLines, writeJsonAtomic, writeJsonLines } from "./jsonl.js";
import type { DatasetManifest, RagEvalManifest } from "./manifest.js";

interface GraphRagQuestion {
  readonly id: string;
  readonly source: string;
  readonly question: string;
  readonly answer: string;
  readonly question_type: string;
  readonly evidence: string;
  readonly evidence_relations?: string;
  readonly evidence_triple?: string;
}

interface GraphRagCorpusRecord {
  readonly corpus_name: string;
  readonly context: string;
}

interface CanonicalDatasetMetadata {
  readonly id: DatasetId;
  readonly track: DatasetTrack;
  readonly source: string;
  readonly version: string;
  readonly license: string;
}

export interface DatasetLoadOptions {
  readonly limit?: number;
  readonly categories?: readonly string[];
  readonly chunkCharacters?: number;
  readonly chunkOverlapCharacters?: number;
}

export interface DatasetPaths {
  readonly repositoryDataDirectory: string;
  readonly externalDataRoot: string;
}

const GRAPH_RAG_LICENSE = "Apache-2.0 (verify upstream dataset terms before redistribution)";

const DATASET_PREPARATION_NOTES: Partial<Record<DatasetId, string>> = {
  "beir-scifact": "The official BEIR SciFact archive must be converted with prepare-beir.ts",
  "beir-nfcorpus": "The official BEIR NFCorpus archive must be converted with prepare-beir.ts",
  garage:
    "The published GaRAGe benchmark depends on private grounding sources; a rights-bearing complete corpus is required",
  "uaeval-kontext":
    "A versioned, human-reviewed UAEval4RAG-style set for the actual indexed corpus is required",
  "stable-rag":
    "The versioned query set and deterministic retrieval-order perturbation suite are required",
  crag: "The official query-specific web pages and mock-KG resources must be provisioned for the dynamic-API track",
  "trec-rag":
    "The official large MS MARCO v2 passage corpus must be provisioned on a large-corpus host",
  ragtime:
    "The registered/live multilingual news task resources and report interface must be provisioned",
};

export function defaultDatasetPaths(repositoryRoot: string): DatasetPaths {
  return {
    repositoryDataDirectory: resolve(repositoryRoot, "bench/data"),
    externalDataRoot: resolve(
      process.env.RAG_EVAL_DATA_ROOT ?? join(repositoryRoot, "bench/data/rag-eval-v2"),
    ),
  };
}

export function doctorDatasets(
  manifest: RagEvalManifest,
  paths: DatasetPaths,
): DatasetDoctorResult[] {
  return manifest.datasets.map((dataset) => doctorDataset(dataset, paths));
}

function doctorDataset(dataset: DatasetManifest, paths: DatasetPaths): DatasetDoctorResult {
  if (dataset.id === "graphrag-bench-medical" || dataset.id === "graphrag-bench-novel") {
    const suffix = dataset.id.endsWith("medical") ? "medical" : "novel";
    const required = [
      join(paths.repositoryDataDirectory, `gb-${suffix}.json`),
      join(paths.repositoryDataDirectory, `gb-${suffix}-questions.json`),
    ];
    const missing = required.filter((path) => !existsSync(path));
    return missing.length === 0
      ? { datasetId: dataset.id, status: "ready", detail: required.join(", ") }
      : { datasetId: dataset.id, status: "blocked", detail: `Missing ${missing.join(", ")}` };
  }

  const datasetDirectory = join(paths.externalDataRoot, dataset.requiredDataPath ?? dataset.id);
  const required = ["dataset.json", "corpus.jsonl", "queries.jsonl"].map((name) =>
    join(datasetDirectory, name),
  );
  const missing = required.filter((path) => !existsSync(path));
  if (missing.length > 0) {
    return {
      datasetId: dataset.id,
      status: "blocked",
      detail: `${DATASET_PREPARATION_NOTES[dataset.id] ?? "Canonical adapter data not prepared"}. Missing: ${missing.join(", ")}`,
    };
  }
  return { datasetId: dataset.id, status: "ready", detail: datasetDirectory };
}

export function loadDataset(
  id: DatasetId,
  paths: DatasetPaths,
  options: DatasetLoadOptions = {},
): DatasetBundle {
  if (id === "graphrag-bench-medical") return loadGraphRagMedical(paths, options);
  if (id === "graphrag-bench-novel") return loadGraphRagNovel(paths, options);
  return loadCanonicalDataset(id, paths, options);
}

function loadGraphRagMedical(paths: DatasetPaths, options: DatasetLoadOptions): DatasetBundle {
  const corpusPath = join(paths.repositoryDataDirectory, "gb-medical.json");
  const questionPath = join(paths.repositoryDataDirectory, "gb-medical-questions.json");
  const corpus = JSON.parse(readFileSync(corpusPath, "utf8")) as GraphRagCorpusRecord;
  const source = toSourceDocument(corpus);
  const documents = [source];
  const questions = selectGraphRagQuestions(
    JSON.parse(readFileSync(questionPath, "utf8")) as GraphRagQuestion[],
    options,
  );
  return {
    id: "graphrag-bench-medical",
    track: "static-kb",
    documents,
    queries: toGraphRagQueries(questions, documents),
    provenance: {
      source: "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench",
      version: fileVersion(corpusPath, questionPath),
      license: GRAPH_RAG_LICENSE,
    },
  };
}

function loadGraphRagNovel(paths: DatasetPaths, options: DatasetLoadOptions): DatasetBundle {
  const corpusPath = join(paths.repositoryDataDirectory, "gb-novel.json");
  const questionPath = join(paths.repositoryDataDirectory, "gb-novel-questions.json");
  const corpora = JSON.parse(readFileSync(corpusPath, "utf8")) as GraphRagCorpusRecord[];
  const documents = corpora.map(toSourceDocument);
  const questions = selectGraphRagQuestions(
    JSON.parse(readFileSync(questionPath, "utf8")) as GraphRagQuestion[],
    options,
  );
  return {
    id: "graphrag-bench-novel",
    track: "static-kb",
    documents,
    queries: toGraphRagQueries(questions, documents),
    provenance: {
      source: "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench",
      version: fileVersion(corpusPath, questionPath),
      license: GRAPH_RAG_LICENSE,
    },
  };
}

function toSourceDocument(record: GraphRagCorpusRecord): CorpusDocument {
  return {
    id: record.corpus_name,
    sourceId: record.corpus_name,
    title: record.corpus_name,
    text: record.context,
    metadata: { original: true },
  };
}

export function chunkDocument(
  source: CorpusDocument,
  chunkCharacters: number,
  overlap: number,
): CorpusDocument[] {
  const chunks: CorpusDocument[] = [];
  let start = 0;
  let index = 0;
  while (start < source.text.length) {
    const hardEnd = Math.min(source.text.length, start + chunkCharacters);
    let end = hardEnd;
    if (hardEnd < source.text.length) {
      const boundary = source.text.lastIndexOf(" ", hardEnd);
      if (boundary > start + Math.floor(chunkCharacters * 0.7)) end = boundary;
    }
    const text = source.text.slice(start, end).trim();
    if (text) {
      chunks.push({
        id: `${source.id}::chunk-${String(index).padStart(6, "0")}`,
        sourceId: source.sourceId,
        title: source.title,
        text,
        metadata: { original: false, chunkIndex: index, start, end },
      });
      index += 1;
    }
    if (end >= source.text.length) break;
    start = Math.max(start + 1, end - overlap);
  }
  return chunks;
}

function toGraphRagQueries(
  questions: readonly GraphRagQuestion[],
  documents: readonly CorpusDocument[],
): BenchmarkQuery[] {
  return questions.map((question) => {
    const evidenceText = splitEvidence(question.evidence);
    return {
      id: question.id,
      text: question.question,
      referenceAnswer: question.answer || null,
      goldEvidenceIds: matchEvidenceDocuments(evidenceText, documents, question.source),
      goldEvidenceText: evidenceText,
      answerable: true,
      category: question.question_type,
      metadata: {
        source: question.source,
        evidenceRelations: question.evidence_relations ?? null,
        evidenceTriple: question.evidence_triple ?? null,
      },
    };
  });
}

function splitEvidence(evidence: string): string[] {
  const trimmed = evidence.trim();
  if (!trimmed) return [];
  return trimmed
    .split(/\n{2,}|\s*\|\|\s*/)
    .map((part) => part.trim())
    .filter(Boolean);
}

function normalizeText(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").trim();
}

const normalizedDocumentCache = new WeakMap<CorpusDocument, string>();
const documentTokenCache = new WeakMap<CorpusDocument, ReadonlySet<string>>();

function matchEvidenceDocuments(
  evidenceText: readonly string[],
  documents: readonly CorpusDocument[],
  sourceId: string,
): string[] {
  const candidates = documents.filter((document) => document.sourceId === sourceId);
  const matched = new Set<string>();
  for (const evidence of evidenceText) {
    const normalizedEvidence = normalizeText(evidence);
    if (!normalizedEvidence) continue;
    for (const document of candidates) {
      let normalizedDocument = normalizedDocumentCache.get(document);
      if (normalizedDocument === undefined) {
        normalizedDocument = normalizeText(document.text);
        normalizedDocumentCache.set(document, normalizedDocument);
      }
      let documentTokens = documentTokenCache.get(document);
      if (documentTokens === undefined) {
        documentTokens = normalizedTokens(normalizedDocument);
        documentTokenCache.set(document, documentTokens);
      }
      if (
        normalizedDocument.includes(normalizedEvidence) ||
        tokenRecall(normalizedEvidence, documentTokens) >= 0.8
      ) {
        matched.add(document.sourceId);
      }
    }
  }
  if (matched.size === 0 && candidates.length > 0) matched.add(sourceId);
  return [...matched].sort();
}

function tokenRecall(needle: string, haystackTokens: ReadonlySet<string>): number {
  const tokens = new Set(needle.split(/[^a-z0-9]+/).filter((token) => token.length >= 3));
  if (tokens.size === 0) return 0;
  let found = 0;
  for (const token of tokens) if (haystackTokens.has(token)) found += 1;
  return found / tokens.size;
}

function normalizedTokens(value: string): ReadonlySet<string> {
  return new Set(value.split(/[^a-z0-9]+/).filter(Boolean));
}

function selectQueries(
  queries: readonly BenchmarkQuery[],
  options: DatasetLoadOptions,
): BenchmarkQuery[] {
  const categories = options.categories ? new Set(options.categories) : null;
  const selected = categories
    ? queries.filter((query) => categories.has(query.category))
    : [...queries];
  return options.limit === undefined ? selected : selected.slice(0, options.limit);
}

function selectGraphRagQuestions(
  questions: readonly GraphRagQuestion[],
  options: DatasetLoadOptions,
): GraphRagQuestion[] {
  const categories = options.categories ? new Set(options.categories) : null;
  const selected = categories
    ? questions.filter((question) => categories.has(question.question_type))
    : [...questions];
  return options.limit === undefined ? selected : selected.slice(0, options.limit);
}

function loadCanonicalDataset(
  id: DatasetId,
  paths: DatasetPaths,
  options: DatasetLoadOptions,
): DatasetBundle {
  const directory = join(paths.externalDataRoot, id);
  const metadataPath = join(directory, "dataset.json");
  const corpusPath = join(directory, "corpus.jsonl");
  const queryPath = join(directory, "queries.jsonl");
  if (!existsSync(metadataPath) || !existsSync(corpusPath) || !existsSync(queryPath)) {
    throw new Error(`Canonical dataset ${id} is not prepared under ${directory}`);
  }
  const metadata = JSON.parse(readFileSync(metadataPath, "utf8")) as CanonicalDatasetMetadata;
  if (metadata.id !== id) throw new Error(`Expected dataset ${id}, found ${metadata.id}`);
  const documents = readJsonLines<CorpusDocument>(corpusPath);
  const queries = selectQueries(readJsonLines<BenchmarkQuery>(queryPath), options);
  validateCanonicalRecords(documents, queries, directory);
  return {
    id,
    track: metadata.track,
    documents,
    queries,
    provenance: {
      source: metadata.source,
      version: metadata.version,
      license: metadata.license,
    },
  };
}

function validateCanonicalRecords(
  documents: readonly CorpusDocument[],
  queries: readonly BenchmarkQuery[],
  directory: string,
): void {
  const documentIds = new Set<string>();
  for (const document of documents) {
    if (!document.id || !document.text || !document.sourceId) {
      throw new Error(`Invalid corpus record in ${directory}`);
    }
    if (documentIds.has(document.id)) throw new Error(`Duplicate document id ${document.id}`);
    documentIds.add(document.id);
  }
  const queryIds = new Set<string>();
  for (const query of queries) {
    if (!query.id || !query.text) throw new Error(`Invalid query record in ${directory}`);
    if (queryIds.has(query.id)) throw new Error(`Duplicate query id ${query.id}`);
    queryIds.add(query.id);
  }
}

export function writePreparedDataset(directory: string, bundle: DatasetBundle): void {
  writeJsonLines(join(directory, "corpus.jsonl"), bundle.documents);
  writeJsonLines(join(directory, "queries.jsonl"), bundle.queries);
  const metadata: CanonicalDatasetMetadata = {
    id: bundle.id,
    track: bundle.track,
    source: bundle.provenance.source,
    version: bundle.provenance.version,
    license: bundle.provenance.license,
  };
  writeJsonAtomic(join(directory, "dataset.json"), metadata);
}

function fileVersion(...paths: readonly string[]): string {
  const hash = createHash("sha256");
  for (const path of paths) hash.update(readFileSync(path)).update("\0");
  return `sha256:${hash.digest("hex")}`;
}
