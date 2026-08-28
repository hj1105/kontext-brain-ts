// @ts-nocheck -- standalone benchmark artifact inspector for evolving JSONL schemas
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defaultDatasetPaths, loadDataset } from "./datasets.js";
import { bootstrapMean95Ci, contextPrecisionForQuery, evidenceRecallForQuery } from "./metrics.js";

const [datasetId, baselinePath, candidatePath, outputPath] = process.argv.slice(2);
if (!datasetId || !baselinePath || !candidatePath) {
  throw new Error(
    "Usage: compare-retrieval-runs.ts <dataset-id> <baseline.jsonl> <candidate.jsonl> [output.json]",
  );
}
const repositoryRoot = resolve(fileURLToPath(import.meta.url), "../../../..");
const bundle = loadDataset(datasetId, defaultDatasetPaths(repositoryRoot));
const baseline = readJsonl(resolve(baselinePath));
const candidate = readJsonl(resolve(candidatePath));

const result = {
  generatedAt: new Date().toISOString(),
  datasetId,
  comparison: compareRuns(bundle, baseline, candidate),
  baseline: summarizeRun(bundle, baseline),
  candidate: summarizeRun(bundle, candidate),
};

if (outputPath) writeFileSync(resolve(outputPath), `${JSON.stringify(result, null, 2)}\n`);
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);

function compareRuns(bundle, baselineRows, candidateRows) {
  const baseline = new Map(baselineRows.map((row) => [row.queryId, row]));
  const candidate = new Map(candidateRows.map((row) => [row.queryId, row]));
  const pairs = bundle.queries.flatMap((query) => {
    const before = baseline.get(query.id);
    const after = candidate.get(query.id);
    if (before?.status !== "ok" || after?.status !== "ok") return [];
    const beforeRecall = evidenceRecallForQuery(query, before.evidence);
    const afterRecall = evidenceRecallForQuery(query, after.evidence);
    const beforePrecision = contextPrecisionForQuery(query, before.evidence);
    const afterPrecision = contextPrecisionForQuery(query, after.evidence);
    if (
      beforeRecall === null ||
      afterRecall === null ||
      beforePrecision === null ||
      afterPrecision === null
    )
      return [];
    const beforeIds = new Set(before.evidence.map((item) => item.id));
    const afterIds = new Set(after.evidence.map((item) => item.id));
    return [
      {
        query,
        beforeRecall,
        afterRecall,
        beforePrecision,
        afterPrecision,
        recallDelta: afterRecall - beforeRecall,
        precisionDelta: afterPrecision - beforePrecision,
        overlap:
          [...beforeIds].filter((id) => afterIds.has(id)).length /
          Math.max(1, Math.min(10, beforeIds.size, afterIds.size)),
        beforeDirect: directRatio(before),
        afterDirect: directRatio(after),
      },
    ];
  });
  return {
    pairedQueries: pairs.length,
    all: pairSummary(pairs),
    development: pairSummary(pairs.filter(({ query }) => !isHoldout(query.id))),
    holdout: pairSummary(pairs.filter(({ query }) => isHoldout(query.id))),
    byCategory: Object.fromEntries(
      [...new Set(pairs.map(({ query }) => query.category))]
        .sort()
        .map((category) => [
          category,
          pairSummary(pairs.filter(({ query }) => query.category === category)),
        ]),
    ),
  };
}

function pairSummary(pairs) {
  const recallDeltas = pairs.map((pair) => pair.recallDelta);
  const precisionDeltas = pairs.map((pair) => pair.precisionDelta);
  return {
    queries: pairs.length,
    baselineRecall: mean(pairs.map((pair) => pair.beforeRecall)),
    candidateRecall: mean(pairs.map((pair) => pair.afterRecall)),
    recallDelta: mean(recallDeltas),
    recallDelta95Ci: recallDeltas.length ? bootstrapMean95Ci(recallDeltas, 10_000, 20260824) : null,
    baselinePrecision: mean(pairs.map((pair) => pair.beforePrecision)),
    candidatePrecision: mean(pairs.map((pair) => pair.afterPrecision)),
    precisionDelta: mean(precisionDeltas),
    precisionDelta95Ci: precisionDeltas.length
      ? bootstrapMean95Ci(precisionDeltas, 10_000, 20260824)
      : null,
    recallWins: recallDeltas.filter((value) => value > 0).length,
    recallTies: recallDeltas.filter((value) => value === 0).length,
    recallLosses: recallDeltas.filter((value) => value < 0).length,
    meanTop10Overlap: mean(pairs.map((pair) => pair.overlap)),
    baselineDirectEvidenceRatio: mean(pairs.map((pair) => pair.beforeDirect)),
    candidateDirectEvidenceRatio: mean(pairs.map((pair) => pair.afterDirect)),
  };
}

function summarizeRun(bundle, rows) {
  const byQuery = new Map(rows.map((row) => [row.queryId, row]));
  const values = bundle.queries.flatMap((query) => {
    const row = byQuery.get(query.id);
    if (row?.status !== "ok") return [];
    const recall = evidenceRecallForQuery(query, row.evidence);
    const precision = contextPrecisionForQuery(query, row.evidence);
    return recall === null || precision === null ? [] : [{ query, row, recall, precision }];
  });
  const stoppedBy = countBy(rows, (row) => row.evidence[0]?.metadata?.stoppedBy ?? "missing");
  const selectedEvidence = values.flatMap(({ row }) => row.evidence);
  const selectedByPathShape = new Map();
  const selectedBySeedProviders = new Map();
  for (const { query, row } of values) {
    for (const evidence of row.evidence) {
      const shape = pathShape(evidence);
      const entries = selectedByPathShape.get(shape) ?? [];
      entries.push(contextPrecisionForQuery(query, [evidence]) ?? 0);
      selectedByPathShape.set(shape, entries);
      const providers = seedProviderProfile(evidence);
      const providerEntries = selectedBySeedProviders.get(providers) ?? [];
      providerEntries.push(contextPrecisionForQuery(query, [evidence]) ?? 0);
      selectedBySeedProviders.set(providers, providerEntries);
    }
  }
  const selectedMissingSignals = new Map();
  for (const evidence of selectedEvidence) {
    const breakdown = scoreBreakdown(evidence);
    if (!breakdown) continue;
    for (const signal of [
      ...breakdown.seed.missingSignals,
      ...breakdown.edges.flatMap((edge) => edge.missingSignals),
      ...breakdown.evidence.missingSignals,
    ]) {
      selectedMissingSignals.set(signal, (selectedMissingSignals.get(signal) ?? 0) + 1);
    }
  }
  const missingSignals = new Map();
  for (const row of rows) {
    const unique = new Set();
    for (const evidence of row.evidence) {
      const encoded = evidence.metadata?.missingSignals;
      if (typeof encoded !== "string") continue;
      for (const signal of encoded.split(",").filter(Boolean)) unique.add(signal);
    }
    for (const signal of unique) missingSignals.set(signal, (missingSignals.get(signal) ?? 0) + 1);
  }
  return {
    queries: values.length,
    recall: mean(values.map(({ recall }) => recall)),
    precision: mean(values.map(({ precision }) => precision)),
    development: metricSummary(values.filter(({ query }) => !isHoldout(query.id))),
    holdout: metricSummary(values.filter(({ query }) => isHoldout(query.id))),
    directEvidenceRatio: mean(values.map(({ row }) => directRatio(row))),
    averageEvidence: mean(values.map(({ row }) => row.evidence.length)),
    averagePathLength: mean(values.flatMap(({ row }) => row.evidence.map(pathLength))),
    selectedPathShapes: countBy(selectedEvidence, pathShape),
    precisionByPathShape: Object.fromEntries(
      [...selectedByPathShape.entries()].map(([shape, precisions]) => [shape, mean(precisions)]),
    ),
    precisionBySeedProviders: Object.fromEntries(
      [...selectedBySeedProviders.entries()].map(([providers, precisions]) => [
        providers,
        { selected: precisions.length, precision: mean(precisions) },
      ]),
    ),
    meanSeedScoreByPathShape: Object.fromEntries(
      [...new Set(selectedEvidence.map(pathShape))].map((shape) => [
        shape,
        mean(
          selectedEvidence
            .filter((evidence) => pathShape(evidence) === shape)
            .flatMap((evidence) => {
              const breakdown = scoreBreakdown(evidence);
              return breakdown ? [breakdown.seed.score] : [];
            }),
        ),
      ]),
    ),
    stoppedBy,
    queriesWithMissingSignal: Object.fromEntries(
      [...missingSignals.entries()].sort((left, right) => right[1] - left[1]),
    ),
    selectedEvidenceMissingSignals: Object.fromEntries(
      [...selectedMissingSignals.entries()].sort((left, right) => right[1] - left[1]),
    ),
    scoringProfile: rows.find((row) => row.scoringProfile)?.scoringProfile ?? null,
    featureSchemaVersion:
      rows.find((row) => row.featureSchemaVersion)?.featureSchemaVersion ?? null,
  };
}

function metricSummary(values) {
  return {
    queries: values.length,
    recall: mean(values.map(({ recall }) => recall)),
    precision: mean(values.map(({ precision }) => precision)),
  };
}

function directRatio(row) {
  if (row.evidence.length === 0) return 0;
  return row.evidence.filter((item) => !item.metadata?.path).length / row.evidence.length;
}

function pathLength(evidence) {
  const path = evidence.metadata?.path;
  return typeof path === "string" && path ? path.split(" | ").length : 0;
}

function pathShape(evidence) {
  const path = evidence.metadata?.path;
  if (typeof path !== "string" || !path) return "direct-chunk-seed";
  const edges = path.split(" | ");
  const first = edges[0] ?? "unknown";
  if (first.startsWith("entity:")) return "entity-seed-path";
  if (first.startsWith("chunk:") && first.includes("->resource:")) return "chunk-resource-path";
  if (first.startsWith("chunk:")) return "chunk-graph-path";
  return "other-path";
}

function scoreBreakdown(evidence) {
  const encoded = evidence.metadata?.scoreBreakdown;
  if (typeof encoded !== "string") return null;
  try {
    return JSON.parse(encoded);
  } catch {
    return null;
  }
}

function seedProviderProfile(evidence) {
  const breakdown = scoreBreakdown(evidence);
  const providers = breakdown?.seed?.observations?.providers;
  return typeof providers === "string" && providers ? providers : "missing";
}

function isHoldout(queryId) {
  return createHash("sha256").update(queryId).digest().readUInt32BE(0) % 100 < 20;
}

function countBy(values, key) {
  const counts = new Map();
  for (const value of values) {
    const item = key(value);
    counts.set(item, (counts.get(item) ?? 0) + 1);
  }
  return Object.fromEntries([...counts.entries()].sort((left, right) => right[1] - left[1]));
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function readJsonl(path) {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}
