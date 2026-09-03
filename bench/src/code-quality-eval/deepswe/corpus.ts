import { createHash } from "node:crypto";
import { readFile, realpath } from "node:fs/promises";
import path from "node:path";
import type {
  DeepSweArm,
  DeepSweContextBundle,
  DeepSweContextCorpus,
  DeepSweSourceDocument,
} from "./contracts.js";

const forbiddenSegments = new Set([
  ".git",
  "agent",
  "artifacts",
  "jobs",
  "result",
  "results",
  "solution",
  "tests",
  "trajectories",
  "verifier",
]);

export async function loadDeepSweCorpus(input: {
  readonly corpusRoot: string;
  readonly taskId: string;
  readonly taskPath: string;
  readonly datasetTasksPath: string;
}): Promise<DeepSweContextCorpus> {
  const [corpusRoot, taskPath, datasetTasksPath] = await Promise.all([
    realpath(input.corpusRoot),
    realpath(input.taskPath),
    realpath(input.datasetTasksPath),
  ]);
  if (isWithin(corpusRoot, taskPath) || isWithin(corpusRoot, datasetTasksPath)) {
    throw new Error("DeepSWE context corpus must live outside the task and dataset trees");
  }
  const candidates = [
    path.join(corpusRoot, `${input.taskId}.json`),
    path.join(corpusRoot, input.taskId, "corpus.json"),
  ];
  let corpusPath: string | undefined;
  for (const candidate of candidates) {
    try {
      corpusPath = await realpath(candidate);
      break;
    } catch {
      // Try the next supported layout.
    }
  }
  if (!corpusPath) throw new Error(`Missing context corpus for DeepSWE task ${input.taskId}`);
  if (!isWithin(corpusPath, corpusRoot)) {
    throw new Error(`Context corpus escapes its root: ${corpusPath}`);
  }
  const parsed = JSON.parse(await readFile(corpusPath, "utf8")) as DeepSweContextCorpus;
  validateCorpus(parsed, input.taskId, taskPath);
  return parsed;
}

export function validateCorpus(
  corpus: DeepSweContextCorpus,
  expectedTaskId: string,
  taskPath: string,
): void {
  if (!corpus || typeof corpus !== "object") throw new Error("Corpus must be a JSON object");
  if (corpus.schemaVersion !== 1) throw new Error("Unsupported DeepSWE corpus schema");
  if (corpus.taskId !== expectedTaskId) {
    throw new Error(`Corpus task mismatch: ${corpus.taskId} != ${expectedTaskId}`);
  }
  if (corpus.generator?.name !== "kontext-brain" || !corpus.generator.revision?.trim()) {
    throw new Error("Corpus must identify the kontext-brain generator revision");
  }
  if (!Array.isArray(corpus.documents) || !Array.isArray(corpus.normativeRecords)) {
    throw new Error("Corpus documents and normativeRecords must be arrays");
  }
  const snapshot = Date.parse(corpus.snapshotAt);
  if (!Number.isFinite(snapshot))
    throw new Error(`Invalid corpus snapshotAt: ${corpus.snapshotAt}`);
  const documentIds = new Set<string>();
  for (const document of corpus.documents) {
    validateDocument(document, snapshot, taskPath);
    if (documentIds.has(document.documentId)) {
      throw new Error(`Duplicate corpus document id: ${document.documentId}`);
    }
    documentIds.add(document.documentId);
  }
  const recordIds = new Set<string>();
  for (const record of corpus.normativeRecords) {
    if (!["decision", "domain_term", "invariant"].includes(record.kind)) {
      throw new Error(`Unsupported normative record kind: ${String(record.kind)}`);
    }
    if (!record.recordId?.trim() || !record.revisionId?.trim() || !record.text?.trim()) {
      throw new Error("Normative records require recordId, revisionId, and text");
    }
    if (!Array.isArray(record.evidenceIds) || !Array.isArray(record.ontologyNodeIds)) {
      throw new Error(`Normative record ${record.recordId} has invalid evidence or ontology ids`);
    }
    if (recordIds.has(record.recordId)) {
      throw new Error(`Duplicate normative record id: ${record.recordId}`);
    }
    recordIds.add(record.recordId);
    if (record.evidenceIds.length === 0) {
      throw new Error(`Normative record ${record.recordId} has no provenance evidence`);
    }
    for (const evidenceId of record.evidenceIds) {
      if (!documentIds.has(evidenceId)) {
        throw new Error(`Normative record ${record.recordId} has unknown evidence ${evidenceId}`);
      }
    }
  }
}

export function buildContextBundle(
  arm: DeepSweArm,
  corpus: DeepSweContextCorpus,
): DeepSweContextBundle {
  const corpusSha256 = sha256(stableJson(corpus.documents));
  const documents = arm === "baseline" ? [] : corpus.documents;
  const normativeRecords = arm === "kontext" ? corpus.normativeRecords : [];
  const projection = {
    arm,
    taskId: corpus.taskId,
    documents,
    normativeRecords,
  };
  return {
    schemaVersion: 1,
    arm,
    taskId: corpus.taskId,
    snapshotAt: corpus.snapshotAt,
    corpusSha256,
    projectionSha256: sha256(stableJson(projection)),
    generator: corpus.generator,
    documents,
    normativeRecords,
  };
}

export function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Readonly<Record<string, unknown>>).sort(([a], [b]) =>
      a.localeCompare(b),
    );
    return `{${entries.map(([key, entry]) => `${JSON.stringify(key)}:${stableJson(entry)}`).join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

export function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function validateDocument(
  document: DeepSweSourceDocument,
  snapshot: number,
  taskPath: string,
): void {
  if (!document.documentId?.trim() || !document.title?.trim() || !document.body?.trim()) {
    throw new Error("Corpus documents require documentId, title, and body");
  }
  if (!document.sourceUri?.trim() || !Array.isArray(document.ontologyNodeIds)) {
    throw new Error(`Document ${document.documentId} has invalid provenance metadata`);
  }
  const observedAt = Date.parse(document.observedAt);
  if (!Number.isFinite(observedAt) || observedAt > snapshot) {
    throw new Error(`Document ${document.documentId} is invalid or newer than the snapshot`);
  }
  const expected = document.contentSha256.replace(/^sha256:/, "");
  if (expected !== sha256(document.body)) {
    throw new Error(`Document ${document.documentId} content hash does not match`);
  }
  const sourcePath = fileSourcePath(document.sourceUri);
  const segments = provenanceSegments(document.sourceUri);
  if (segments.some((segment) => forbiddenSegments.has(segment.toLowerCase()))) {
    throw new Error(`Forbidden benchmark artifact in corpus provenance: ${document.sourceUri}`);
  }
  if (!sourcePath) return;
  if (isWithin(path.resolve(sourcePath), path.resolve(taskPath))) {
    throw new Error(`Task artifact cannot be a corpus source: ${document.sourceUri}`);
  }
}

function provenanceSegments(uri: string): readonly string[] {
  try {
    const parsed = new URL(uri);
    return decodeURIComponent(parsed.pathname)
      .split(/[\\/]+/)
      .filter(Boolean);
  } catch {
    return uri.split(/[\\/]+/).filter(Boolean);
  }
}

function fileSourcePath(uri: string): string | undefined {
  if (uri.startsWith("file://")) return decodeURIComponent(new URL(uri).pathname);
  return path.isAbsolute(uri) ? uri : undefined;
}

function isWithin(candidate: string, parent: string): boolean {
  const relative = path.relative(parent, candidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}
