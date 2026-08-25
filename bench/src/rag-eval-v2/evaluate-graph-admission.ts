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
    "Usage: evaluate-graph-admission.ts <dataset-id> <direct.jsonl> <graph.jsonl> " +
      "[<dataset-id> <direct.jsonl> <graph.jsonl> ...] [output.json]",
  );
}
const hasOutput = args.length % 3 === 1;
const outputPath = hasOutput ? args.pop() : undefined;
const repositoryRoot = resolve(fileURLToPath(import.meta.url), "../../../..");
const queryCases = [];
for (let index = 0; index < args.length; index += 3) {
  const datasetId = args[index];
  const directPath = args[index + 1];
  const graphPath = args[index + 2];
  const bundle = loadDataset(datasetId, defaultDatasetPaths(repositoryRoot));
  const directRows = new Map(readJsonl(resolve(directPath)).map((row) => [row.queryId, row]));
  const graphRows = new Map(readJsonl(resolve(graphPath)).map((row) => [row.queryId, row]));
  for (const query of bundle.queries) {
    const direct = directRows.get(query.id);
    const graph = graphRows.get(query.id);
    if (direct?.status !== "ok" || graph?.status !== "ok") continue;
    const directEvidence = [...direct.evidence].sort(byScore).slice(0, 10);
    const directIds = new Set(directEvidence.map((evidence) => evidence.id));
    queryCases.push({
      datasetId,
      query,
      split: splitFor(query.id, query.category, query.answerable),
      directEvidence,
      directBoundary: directBoundaryFeatures(
        directEvidence[directEvidence.length - 1],
        directEvidence[0]?.score ?? 0,
      ),
      graphCandidates: graph.evidence
        .filter((evidence) => evidence.metadata?.path && !directIds.has(evidence.id))
        .map((evidence) => ({ evidence, features: featuresFor(evidence) }))
        .sort((left, right) => byScore(left.evidence, right.evidence)),
      metricCache: new Map(),
    });
  }
}

const policies = policyGrid();
const directPolicy = {
  id: "direct-only",
  maxGraph: 0,
  scoreMargin: Number.POSITIVE_INFINITY,
  minRouteGate: 0,
  minQueryEvidence: 0,
  minSeedGate: 0,
  minSupport: 0,
  maxPathLength: 0,
  shape: "none",
};
const baseline = evaluatePolicy(directPolicy, queryCases);
const evaluated = policies.map((policy) => evaluatePolicy(policy, queryCases));
const developmentShortlist = evaluated
  .filter((result) => result.splits.development.graphSelected > 0)
  .filter((result) => nonRegresses(result, baseline, "development"))
  .sort((left, right) => compareSplit(right.splits.development, left.splits.development))
  .slice(0, 100);
const validationEligible = developmentShortlist.filter((result) =>
  nonRegresses(result, baseline, "validation"),
);
const selected = [...validationEligible].sort((left, right) => {
  const validation = compareSplit(right.splits.validation, left.splits.validation);
  return validation !== 0
    ? validation
    : compareSplit(right.splits.development, left.splits.development);
})[0];

const result = {
  generatedAt: new Date().toISOString(),
  protocol: {
    splitSeed: "graph-admission-v2",
    selection:
      "Require aggregate and per-dataset non-regression on development, then the same gate and lexicographic recall/precision selection on validation; holdout is reported only after selection.",
    runtimeFeaturesOnly: true,
    datasetIdentityFeature: false,
    policiesEvaluated: policies.length,
    developmentShortlist: developmentShortlist.length,
    validationEligible: validationEligible.length,
  },
  baseline,
  selected: selected ?? null,
  topDevelopment: developmentShortlist.slice(0, 10),
  topValidation: [...validationEligible]
    .sort((left, right) => compareSplit(right.splits.validation, left.splits.validation))
    .slice(0, 10),
  graphCandidateDiagnostics: graphCandidateDiagnostics(queryCases),
  swapOutcomeDiagnostics: swapOutcomeDiagnostics(queryCases),
};
if (outputPath) writeFileSync(resolve(outputPath), `${JSON.stringify(result, null, 2)}\n`);
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);

function policyGrid() {
  const policies = [];
  for (const scoreMargin of [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7]) {
    for (const minRouteGate of [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]) {
      for (const minQueryEvidence of [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]) {
        for (const minDirectWorstScore of [0, 0.1, 0.15, 0.18, 0.2]) {
          for (const maxDirectWorstProviderCount of [1, 2]) {
            const policy = {
              maxGraph: 1,
              scoreMargin,
              minRouteGate,
              minQueryEvidence,
              minSeedGate: 0,
              minSupport: 0,
              maxPathLength: 1,
              shape: "entity",
              minDirectWorstScore,
              maxDirectWorstProviderCount,
            };
            policies.push({ ...policy, id: policyId(policy) });
          }
        }
      }
    }
  }
  return policies;
}

function evaluatePolicy(policy, cases) {
  const values = cases.map((item) => evaluateCase(policy, item));
  const splits = Object.fromEntries(
    ["development", "validation", "holdout", "all"].map((split) => [
      split,
      summarize(split === "all" ? values : values.filter((value) => value.split === split)),
    ]),
  );
  return { policy, splits };
}

function evaluateCase(policy, item) {
  const selected = applyPolicy(policy, item);
  const signature = selected.map((evidence) => evidence.id).join("\0");
  let metrics = item.metricCache.get(signature);
  if (!metrics) {
    metrics = {
      recall: evidenceRecallForQuery(item.query, selected) ?? 0,
      precision: contextPrecisionForQuery(item.query, selected) ?? 0,
    };
    item.metricCache.set(signature, metrics);
  }
  return {
    datasetId: item.datasetId,
    queryId: item.query.id,
    split: item.split,
    recall: metrics.recall,
    precision: metrics.precision,
    graphSelected: selected.filter((evidence) => evidence.metadata?.path).length,
  };
}

function applyPolicy(policy, item) {
  const selected = [...item.directEvidence];
  if (policy.maxGraph === 0) return selected;
  const eligible = item.graphCandidates
    .filter(({ features }) => eligibleByFeatures(policy, features, item.directBoundary))
    .sort((left, right) => byScore(left.evidence, right.evidence));
  let inserted = 0;
  for (const { evidence } of eligible) {
    if (inserted >= policy.maxGraph || selected.some((candidate) => candidate.id === evidence.id)) {
      continue;
    }
    selected.sort(byScore);
    const displaced = selected[selected.length - 1];
    if (!displaced || evidence.score < displaced.score * policy.scoreMargin) continue;
    selected[selected.length - 1] = evidence;
    inserted++;
  }
  return selected.sort(byScore).slice(0, 10);
}

function eligibleByFeatures(policy, features, boundary) {
  if (features.pathLength > policy.maxPathLength) return false;
  if (policy.shape !== "any" && features.shape !== policy.shape) return false;
  return (
    features.minRouteGate >= policy.minRouteGate &&
    features.minQueryEvidence >= policy.minQueryEvidence &&
    features.seedGate >= policy.minSeedGate &&
    features.support >= policy.minSupport &&
    boundary.directWorstScore >= (policy.minDirectWorstScore ?? 0) &&
    boundary.directWorstProviderCount <=
      (policy.maxDirectWorstProviderCount ?? Number.POSITIVE_INFINITY)
  );
}

function featuresFor(evidence) {
  const breakdown = scoreBreakdown(evidence);
  const edges = breakdown?.edges ?? [];
  return {
    score: evidence.score,
    shape: pathShape(evidence),
    pathLength: pathLength(evidence),
    minRouteGate: minimum(
      edges.map((edge) => edge.factors?.adaptiveRouteGate),
      1,
    ),
    minQueryEvidence: minimum(
      edges.map((edge) => edge.factors?.routeQueryEvidence),
      1,
    ),
    minQueryRank: minimum(
      edges.map((edge) => edge.factors?.routeQueryRank),
      1,
    ),
    minSelectivity: minimum(
      edges.map((edge) => edge.factors?.fanoutSelectivity),
      1,
    ),
    seedScore: breakdown?.seed?.score ?? evidence.score,
    seedQuery: breakdown?.seed?.factors?.query ?? 1,
    seedGate: breakdown?.seed?.factors?.adaptiveSeedGate ?? 1,
    seedProviderCount: breakdown?.seed?.observations?.providerCount ?? 0,
    support: breakdown?.evidence?.factors?.support ?? 1,
  };
}

function directBoundaryFeatures(evidence, topScore) {
  const breakdown = scoreBreakdown(evidence);
  return {
    directWorstScore: evidence.score,
    directWorstToTop: topScore === 0 ? 1 : evidence.score / topScore,
    directWorstSeedScore: breakdown?.seed?.score ?? evidence.score,
    directWorstSeedQuery: breakdown?.seed?.factors?.query ?? 1,
    directWorstProviderCount: breakdown?.seed?.observations?.providerCount ?? 0,
    directWorstSupport: breakdown?.evidence?.factors?.support ?? 1,
  };
}

function graphCandidateDiagnostics(cases) {
  const values = [];
  for (const item of cases) {
    for (const { evidence, features } of item.graphCandidates) {
      values.push({
        split: item.split,
        relevant: (contextPrecisionForQuery(item.query, [evidence]) ?? 0) > 0,
        ...features,
      });
    }
  }
  return Object.fromEntries(
    ["development", "validation", "holdout", "all"].map((split) => {
      const subset = split === "all" ? values : values.filter((value) => value.split === split);
      return [
        split,
        {
          candidates: subset.length,
          relevant: subset.filter((value) => value.relevant).length,
          precision:
            subset.length === 0
              ? 0
              : subset.filter((value) => value.relevant).length / subset.length,
          byShape: Object.fromEntries(
            ["entity", "resource", "other"].map((shape) => {
              const shaped = subset.filter((value) => value.shape === shape);
              return [
                shape,
                {
                  candidates: shaped.length,
                  relevant: shaped.filter((value) => value.relevant).length,
                },
              ];
            }),
          ),
        },
      ];
    }),
  );
}

function swapOutcomeDiagnostics(cases) {
  const records = [];
  for (const item of cases) {
    const worst = item.directEvidence[item.directEvidence.length - 1];
    const top = item.directEvidence[0];
    if (!worst || !top) continue;
    const baselineRecall = evidenceRecallForQuery(item.query, item.directEvidence) ?? 0;
    const baselinePrecision = contextPrecisionForQuery(item.query, item.directEvidence) ?? 0;
    const boundary = directBoundaryFeatures(worst, top.score);
    for (const { evidence, features } of item.graphCandidates) {
      const selected = [...item.directEvidence.slice(0, -1), evidence].sort(byScore);
      const recall = evidenceRecallForQuery(item.query, selected) ?? 0;
      const precision = contextPrecisionForQuery(item.query, selected) ?? 0;
      records.push({
        split: item.split,
        recallDelta: recall - baselineRecall,
        precisionDelta: precision - baselinePrecision,
        graphToDirectWorst: worst.score === 0 ? 1 : evidence.score / worst.score,
        ...boundary,
        ...features,
      });
    }
  }
  return Object.fromEntries(
    ["development", "validation", "holdout", "all"].map((split) => {
      const subset = split === "all" ? records : records.filter((record) => record.split === split);
      return [
        split,
        {
          positive: numericSummary(
            subset.filter((record) => record.recallDelta > 0 || record.precisionDelta > 0),
          ),
          neutral: numericSummary(
            subset.filter((record) => record.recallDelta === 0 && record.precisionDelta === 0),
          ),
          negative: numericSummary(
            subset.filter((record) => record.recallDelta < 0 || record.precisionDelta < 0),
          ),
        },
      ];
    }),
  );
}

function numericSummary(records) {
  const keys = [
    "recallDelta",
    "precisionDelta",
    "graphToDirectWorst",
    "score",
    "pathLength",
    "minRouteGate",
    "minQueryEvidence",
    "minQueryRank",
    "minSelectivity",
    "seedScore",
    "seedQuery",
    "seedGate",
    "seedProviderCount",
    "support",
    "directWorstScore",
    "directWorstToTop",
    "directWorstSeedScore",
    "directWorstSeedQuery",
    "directWorstProviderCount",
    "directWorstSupport",
  ];
  return {
    candidates: records.length,
    features: Object.fromEntries(
      keys.map((key) => {
        const values = records
          .map((record) => record[key])
          .filter((value) => typeof value === "number" && Number.isFinite(value))
          .sort((left, right) => left - right);
        return [
          key,
          {
            mean: mean(values),
            p25: quantile(values, 0.25),
            p50: quantile(values, 0.5),
            p75: quantile(values, 0.75),
          },
        ];
      }),
    ),
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
    recall: mean(values.map((value) => value.recall)),
    precision: mean(values.map((value) => value.precision)),
    graphSelected: values.reduce((sum, value) => sum + value.graphSelected, 0),
  };
}

function compareSplit(left, right) {
  return (
    left.recall - right.recall ||
    left.precision - right.precision ||
    right.graphSelected - left.graphSelected
  );
}

function nonRegresses(result, baseline, split) {
  const candidate = result.splits[split];
  const control = baseline.splits[split];
  if (candidate.recall < control.recall || candidate.precision < control.precision) return false;
  return Object.keys(control.byDataset).every((datasetId) => {
    const candidateDataset = candidate.byDataset[datasetId];
    const controlDataset = control.byDataset[datasetId];
    return (
      candidateDataset &&
      candidateDataset.recall >= controlDataset.recall &&
      candidateDataset.precision >= controlDataset.precision
    );
  });
}

function splitFor(queryId, category, answerable) {
  const value = createHash("sha256")
    .update(`graph-admission-v2\0${category}\0${answerable}\0${queryId}`)
    .digest()
    .readUInt32BE(0);
  const bucket = value % 100;
  return bucket < 60 ? "development" : bucket < 80 ? "validation" : "holdout";
}

function policyId(policy) {
  return [
    `g${policy.maxGraph}`,
    `m${policy.scoreMargin}`,
    `r${policy.minRouteGate}`,
    `q${policy.minQueryEvidence}`,
    `s${policy.minSeedGate}`,
    `u${policy.minSupport}`,
    `h${policy.maxPathLength}`,
    policy.shape,
    `d${policy.minDirectWorstScore ?? 0}`,
    `p${policy.maxDirectWorstProviderCount ?? "any"}`,
  ].join("-");
}

function pathLength(evidence) {
  const path = evidence.metadata?.path;
  return typeof path === "string" && path ? path.split(" | ").length : 0;
}

function pathShape(evidence) {
  const path = evidence.metadata?.path;
  if (typeof path !== "string" || !path) return "direct";
  const first = path.split(" | ")[0] ?? "";
  if (first.startsWith("entity:")) return "entity";
  if (first.startsWith("chunk:") && first.includes("->resource:")) return "resource";
  return "other";
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

function minimum(values, fallback) {
  const numbers = values.filter((value) => typeof value === "number" && Number.isFinite(value));
  return numbers.length === 0 ? fallback : Math.min(...numbers);
}

function byScore(left, right) {
  return right.score - left.score || left.id.localeCompare(right.id);
}

function mean(values) {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function quantile(values, fraction) {
  if (values.length === 0) return 0;
  return values[Math.min(values.length - 1, Math.floor(values.length * fraction))] ?? 0;
}

function readJsonl(path) {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}
