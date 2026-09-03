import { createHash } from "node:crypto";
import { readFile, realpath } from "node:fs/promises";
import path from "node:path";
import { decodeNormativeManifest } from "@kontext-brain/spec";
import type {
  DeepSweArm,
  DeepSweContextBundle,
  DeepSweContextCorpus,
  DeepSweEvidenceSnapshot,
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
  if (
    !corpus.organizationId?.trim() ||
    !corpus.baseCodeRevision?.trim() ||
    !corpus.contextDigest?.trim() ||
    !corpus.sourceFreshnessDigest?.trim()
  ) {
    throw new Error("Corpus requires Organization and Task Context Snapshot identity");
  }
  if (!Array.isArray(corpus.evidence) || !Array.isArray(corpus.normativeRecords)) {
    throw new Error("Corpus evidence and normativeRecords must be arrays");
  }
  const snapshot = Date.parse(corpus.snapshotAt);
  if (!Number.isFinite(snapshot))
    throw new Error(`Invalid corpus snapshotAt: ${corpus.snapshotAt}`);
  const evidenceIds = new Set<string>();
  for (const evidence of corpus.evidence) {
    validateEvidence(evidence, snapshot, taskPath);
    if (evidenceIds.has(evidence.evidenceId)) {
      throw new Error(`Duplicate corpus Evidence id: ${evidence.evidenceId}`);
    }
    evidenceIds.add(evidence.evidenceId);
  }
  try {
    decodeNormativeManifest(
      JSON.stringify({
        schemaVersion: 1,
        organizationId: corpus.organizationId,
        revisions: corpus.normativeRecords.map((record) => record.revision),
        activations: [],
      }),
    );
  } catch (error) {
    throw new Error("Corpus contains invalid Normative Revisions", { cause: error });
  }
  const recordIds = new Set<string>();
  for (const record of corpus.normativeRecords) {
    const revision = record.revision;
    if (!revision || !["decision", "domain_term", "invariant"].includes(revision.kind)) {
      throw new Error(`Unsupported normative record kind: ${String(revision?.kind)}`);
    }
    if (
      revision.organizationId !== corpus.organizationId ||
      !revision.recordId?.trim() ||
      !revision.revisionId?.trim() ||
      !revision.authoredBy?.trim() ||
      !Number.isFinite(Date.parse(revision.authoredAt)) ||
      !Array.isArray(revision.evidence) ||
      !Array.isArray(revision.egress?.allowedRuntimeProviders) ||
      revision.egress.allowedRuntimeProviders.length === 0
    ) {
      throw new Error("Normative revisions require canonical identity and provenance");
    }
    if (recordIds.has(revision.recordId)) {
      throw new Error(`Duplicate normative record id: ${revision.recordId}`);
    }
    recordIds.add(revision.recordId);
    if (revision.evidence.length === 0) {
      throw new Error(`Normative record ${revision.recordId} has no provenance Evidence`);
    }
    for (const reference of revision.evidence) {
      if (!evidenceIds.has(reference.evidenceId)) {
        throw new Error(
          `Normative record ${revision.recordId} has unknown Evidence ${reference.evidenceId}`,
        );
      }
    }
  }
}

export function buildContextBundle(
  arm: DeepSweArm,
  corpus: DeepSweContextCorpus,
): DeepSweContextBundle {
  const corpusSha256 = sha256(stableJson(corpus));
  const evidence = arm === "baseline" ? [] : corpus.evidence;
  const normativeRecords = arm === "kontext" ? corpus.normativeRecords : [];
  const projection = {
    arm,
    taskId: corpus.taskId,
    evidence,
    normativeRecords,
  };
  return {
    schemaVersion: 1,
    arm,
    taskId: corpus.taskId,
    organizationId: corpus.organizationId,
    baseCodeRevision: corpus.baseCodeRevision,
    contextDigest: corpus.contextDigest,
    sourceFreshnessDigest: corpus.sourceFreshnessDigest,
    snapshotAt: corpus.snapshotAt,
    corpusSha256,
    projectionSha256: sha256(stableJson(projection)),
    generator: corpus.generator,
    evidence,
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

function validateEvidence(
  evidence: DeepSweEvidenceSnapshot,
  snapshot: number,
  taskPath: string,
): void {
  if (
    !evidence.evidenceId?.trim() ||
    !evidence.resourceId?.trim() ||
    !evidence.chunkId?.trim() ||
    !evidence.title?.trim() ||
    !evidence.text?.trim()
  ) {
    throw new Error("Corpus Evidence requires Evidence, Resource, Chunk, title, and text");
  }
  if (!evidence.sourceUri?.trim() || !Array.isArray(evidence.ontologyNodeIds)) {
    throw new Error(`Evidence ${evidence.evidenceId} has invalid provenance metadata`);
  }
  const observedAt = Date.parse(evidence.observedAt);
  if (!Number.isFinite(observedAt) || observedAt > snapshot) {
    throw new Error(`Evidence ${evidence.evidenceId} is invalid or newer than the snapshot`);
  }
  const expected = evidence.contentSha256.replace(/^sha256:/, "");
  if (expected !== sha256(evidence.text)) {
    throw new Error(`Evidence ${evidence.evidenceId} content hash does not match`);
  }
  const sourcePath = fileSourcePath(evidence.sourceUri);
  const segments = provenanceSegments(evidence.sourceUri);
  if (segments.some((segment) => forbiddenSegments.has(segment.toLowerCase()))) {
    throw new Error(`Forbidden benchmark artifact in corpus provenance: ${evidence.sourceUri}`);
  }
  if (!sourcePath) return;
  if (isWithin(path.resolve(sourcePath), path.resolve(taskPath))) {
    throw new Error(`Task artifact cannot be a corpus source: ${evidence.sourceUri}`);
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
