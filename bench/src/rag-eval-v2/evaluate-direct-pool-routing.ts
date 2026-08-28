// @ts-nocheck -- standalone offline policy evaluator for evolving benchmark traces
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defaultDatasetPaths, loadDataset } from "./datasets.js";
import { contextPrecisionForQuery, evidenceRecallForQuery } from "./metrics.js";

const args = process.argv.slice(2);
if (args.length < 6 || (args.length % 3 !== 0 && args.length % 3 !== 1)) {
  throw new Error(
    "Usage: evaluate-direct-pool-routing.ts <dataset-id> <baseline.jsonl> <expanded.jsonl> " +
      "[<dataset-id> <baseline.jsonl> <expanded.jsonl> ...] [output.json]",
  );
}
const outputPath = args.length % 3 === 1 ? args.pop() : undefined;
const repositoryRoot = resolve(fileURLToPath(import.meta.url), "../../../..");
const cases = [];
for (let index = 0; index < args.length; index += 3) {
  const datasetId = args[index];
  const bundle = loadDataset(datasetId, defaultDatasetPaths(repositoryRoot));
  const baseline = new Map(readJsonl(resolve(args[index + 1])).map((row) => [row.queryId, row]));
  const expanded = new Map(readJsonl(resolve(args[index + 2])).map((row) => [row.queryId, row]));
  for (const query of bundle.queries) {
    const before = baseline.get(query.id);
    const after = expanded.get(query.id);
    if (before?.status !== "ok" || after?.status !== "ok") continue;
    const beforeMetrics = metrics(query, before.evidence);
    const afterMetrics = metrics(query, after.evidence);
    cases.push({
      datasetId,
      queryId: query.id,
      split: splitFor(query.id, query.category, query.answerable),
      beforeMetrics,
      afterMetrics,
      features: routingFeatures(query.text, before.evidence, after.evidence),
    });
  }
}

const baseline = evaluatePolicy({ id: "baseline", alwaysBaseline: true }, cases);
const evaluated = policyGrid().map((policy) => evaluatePolicy(policy, cases));
const development = evaluated
  .filter((result) => result.splits.development.expandedQueries > 0)
  .filter((result) => nonRegresses(result, baseline, "development"))
  .sort((left, right) => compareSplit(right.splits.development, left.splits.development))
  .slice(0, 200);
const validation = development
  .filter((result) => nonRegresses(result, baseline, "validation"))
  .sort((left, right) => {
    const validationOrder = compareSplit(right.splits.validation, left.splits.validation);
    return validationOrder !== 0
      ? validationOrder
      : compareSplit(right.splits.development, left.splits.development);
  });
const selected = validation[0] ?? null;
const result = {
  generatedAt: new Date().toISOString(),
  protocol: {
    splitSeed: "direct-pool-routing-v1",
    policiesEvaluated: evaluated.length,
    datasetIdentityFeature: false,
    runtimeFeaturesOnly: true,
    developmentEligible: development.length,
    validationEligible: validation.length,
    selection:
      "Aggregate and per-dataset recall/precision must not regress on development and validation; holdout is read only after selection.",
  },
  baseline,
  selected,
  topDevelopment: development.slice(0, 10),
  topValidation: validation.slice(0, 10),
  outcomeDiagnostics: outcomeDiagnostics(cases),
};
if (outputPath) writeFileSync(resolve(outputPath), `${JSON.stringify(result, null, 2)}\n`);
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);

function policyGrid() {
  const policies = [];
  for (const minExpandedDual of [0, 2, 4, 6, 8]) {
    for (const minDualGain of [-5, -2, 0, 1, 2, 3, 4]) {
      for (const minMeanConsensus of [0, 0.25, 0.35, 0.45, 0.55, 0.65]) {
        for (const maxBaselineDual of [2, 4, 6, 8, 10]) {
          for (const minOverlap of [0, 0.8, 0.9]) {
            const policy = {
              minExpandedDual,
              minDualGain,
              minMeanConsensus,
              maxBaselineDual,
              minOverlap,
            };
            policies.push({ ...policy, id: policyId(policy) });
          }
        }
      }
    }
  }
  return policies;
}

function evaluatePolicy(policy, values) {
  const decisions = values.map((item) => {
    const expanded = !policy.alwaysBaseline && useExpanded(policy, item.features);
    return {
      datasetId: item.datasetId,
      split: item.split,
      expanded,
      metrics: expanded ? item.afterMetrics : item.beforeMetrics,
    };
  });
  return {
    policy,
    splits: Object.fromEntries(
      ["development", "validation", "holdout", "all"].map((split) => [
        split,
        summarize(
          split === "all" ? decisions : decisions.filter((decision) => decision.split === split),
        ),
      ]),
    ),
  };
}

function useExpanded(policy, features) {
  return (
    features.expandedDual >= policy.minExpandedDual &&
    features.dualGain >= policy.minDualGain &&
    features.meanConsensus >= policy.minMeanConsensus &&
    features.baselineDual <= policy.maxBaselineDual &&
    features.overlap >= policy.minOverlap
  );
}

function routingFeatures(question, baseline, expanded) {
  const baselineIds = new Set(baseline.map((evidence) => evidence.id));
  const expandedIds = new Set(expanded.map((evidence) => evidence.id));
  const baselineDual = baseline.filter((evidence) => providerCount(evidence) >= 2).length;
  const expandedDual = expanded.filter((evidence) => providerCount(evidence) >= 2).length;
  const consensus = expanded.map(consensusScore);
  return {
    baselineDual,
    expandedDual,
    dualGain: expandedDual - baselineDual,
    meanConsensus: mean(consensus),
    minimumConsensus: Math.min(...consensus),
    overlap:
      [...baselineIds].filter((id) => expandedIds.has(id)).length /
      Math.max(1, Math.min(10, baselineIds.size, expandedIds.size)),
    queryTerms: question.split(/[^\p{L}\p{N}]+/u).filter(Boolean).length,
  };
}

function providerCount(evidence) {
  const providers = scoreBreakdown(evidence)?.seed?.observations?.providers;
  return typeof providers === "string" && providers ? providers.split(",").length : 0;
}

function consensusScore(evidence) {
  return scoreBreakdown(evidence)?.seed?.observations?.reranker ?? 0;
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

function metrics(query, evidence) {
  return {
    recall: evidenceRecallForQuery(query, evidence) ?? 0,
    precision: contextPrecisionForQuery(query, evidence) ?? 0,
  };
}

function summarize(values) {
  const byDataset = {};
  for (const datasetId of new Set(values.map((value) => value.datasetId))) {
    byDataset[datasetId] = metricSummary(values.filter((value) => value.datasetId === datasetId));
  }
  return { ...metricSummary(values), byDataset };
}

function metricSummary(values) {
  return {
    queries: values.length,
    recall: mean(values.map((value) => value.metrics.recall)),
    precision: mean(values.map((value) => value.metrics.precision)),
    expandedQueries: values.filter((value) => value.expanded).length,
  };
}

function nonRegresses(result, baseline, split) {
  const candidate = result.splits[split];
  const control = baseline.splits[split];
  if (candidate.recall < control.recall || candidate.precision < control.precision) return false;
  return Object.keys(control.byDataset).every((datasetId) => {
    const left = candidate.byDataset[datasetId];
    const right = control.byDataset[datasetId];
    return left && left.recall >= right.recall && left.precision >= right.precision;
  });
}

function compareSplit(left, right) {
  return (
    left.recall - right.recall ||
    left.precision - right.precision ||
    right.expandedQueries - left.expandedQueries
  );
}

function outcomeDiagnostics(values) {
  const rows = values.map((item) => ({
    split: item.split,
    recallDelta: item.afterMetrics.recall - item.beforeMetrics.recall,
    precisionDelta: item.afterMetrics.precision - item.beforeMetrics.precision,
    ...item.features,
  }));
  return Object.fromEntries(
    ["development", "validation", "holdout", "all"].map((split) => {
      const subset = split === "all" ? rows : rows.filter((row) => row.split === split);
      return [
        split,
        {
          positive: numericSummary(
            subset.filter((row) => row.recallDelta > 0 || row.precisionDelta > 0),
          ),
          neutral: numericSummary(
            subset.filter((row) => row.recallDelta === 0 && row.precisionDelta === 0),
          ),
          negative: numericSummary(
            subset.filter((row) => row.recallDelta < 0 || row.precisionDelta < 0),
          ),
        },
      ];
    }),
  );
}

function numericSummary(rows) {
  return {
    queries: rows.length,
    means: Object.fromEntries(
      [
        "recallDelta",
        "precisionDelta",
        "baselineDual",
        "expandedDual",
        "dualGain",
        "meanConsensus",
        "minimumConsensus",
        "overlap",
        "queryTerms",
      ].map((key) => [key, mean(rows.map((row) => row[key]))]),
    ),
  };
}

function splitFor(queryId, category, answerable) {
  const bucket =
    createHash("sha256")
      .update(`direct-pool-routing-v1\0${category}\0${answerable}\0${queryId}`)
      .digest()
      .readUInt32BE(0) % 100;
  return bucket < 60 ? "development" : bucket < 80 ? "validation" : "holdout";
}

function policyId(policy) {
  return `e${policy.minExpandedDual}-g${policy.minDualGain}-c${policy.minMeanConsensus}-b${policy.maxBaselineDual}-o${policy.minOverlap}`;
}

function mean(values) {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function readJsonl(path) {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}
