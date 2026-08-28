import type {
  AnswerResult,
  BenchmarkQuery,
  DatasetBundle,
  FrameworkId,
  JudgeContract,
  JudgeResult,
  RetrievalResult,
  RetrievedEvidence,
} from "./contracts.js";

export interface ConfidenceInterval {
  readonly low: number;
  readonly high: number;
  readonly samples: number;
}

export interface DatasetFrameworkScore {
  readonly datasetId: DatasetBundle["id"];
  readonly frameworkId: FrameworkId;
  readonly retrievalQueries: number;
  readonly retrievalCompleted: number;
  readonly retrievalBlocked: number;
  readonly retrievalUnsupported: number;
  readonly retrievalErrors: number;
  readonly queries: number;
  readonly completed: number;
  readonly blocked: number;
  readonly unsupported: number;
  readonly errors: number;
  readonly evidenceRecallAtK: number | null;
  readonly ndcgAtK: number | null;
  readonly contextPrecision: number | null;
  readonly answerCorrectness: number | null;
  readonly answerCorrectness95Ci: ConfidenceInterval | null;
  readonly claimPrecision: number | null;
  readonly claimRecall: number | null;
  readonly claimSupportPrecision: number | null;
  readonly claimF1: number | null;
  readonly completeness: number | null;
  readonly strictFaithfulness: number | null;
  readonly citationPrecision: number | null;
  readonly citationRecall: number | null;
  readonly citationF1: number | null;
  readonly acceptableAbstention: number | null;
  readonly answerabilityJointAccuracy: number | null;
  readonly robustnessDrop: number | null;
  readonly clarity: number | null;
  readonly conciseness: number | null;
  readonly fluency: number | null;
  readonly cragTruthfulness: number | null;
  readonly retrievalLatencyP95Ms: number | null;
  readonly queryToAnswerLatencyP95Ms: number | null;
  readonly endToEndLatencyP95Ms: number | null;
  readonly averageInputTokens: number | null;
  readonly estimatedCostUsd: null;
}

export interface PairedFrameworkComparison {
  readonly leftFrameworkId: FrameworkId;
  readonly rightFrameworkId: FrameworkId;
  readonly metric:
    | "evidence-recall-at-k"
    | "ndcg-at-k"
    | "answer-correctness"
    | "claim-recall"
    | "claim-support-precision"
    | "strict-faithfulness"
    | "citation-precision"
    | "citation-recall"
    | "clarity"
    | "conciseness"
    | "fluency"
    | "citation-f1";
  readonly pairedQueries: number;
  readonly meanDifferenceLeftMinusRight: number;
  readonly difference95Ci: ConfidenceInterval;
}

export function scoreDatasetFramework(
  bundle: DatasetBundle,
  frameworkId: FrameworkId,
  retrievals: readonly RetrievalResult[],
  answers: readonly AnswerResult[],
  judgements: readonly JudgeResult[],
  evaluationQueries: readonly BenchmarkQuery[] = bundle.queries,
): DatasetFrameworkScore {
  const retrievalByQuery = new Map(retrievals.map((result) => [result.queryId, result]));
  const answerByQuery = new Map(answers.map((result) => [result.queryId, result]));
  const judgeByQuery = new Map(judgements.map((result) => [result.queryId, result]));
  const queryById = new Map(bundle.queries.map((query) => [query.id, query]));
  const retrievalStatuses = bundle.queries.map(
    (query) => retrievalByQuery.get(query.id)?.status ?? "blocked",
  );
  const statuses = evaluationQueries.map((query) =>
    combinedStatus(
      retrievalByQuery.get(query.id),
      answerByQuery.get(query.id),
      judgeByQuery.get(query.id),
    ),
  );

  const evidenceRecall = bundle.queries.flatMap((query) => {
    const retrieval = retrievalByQuery.get(query.id);
    const score = evidenceRecallForQuery(
      query,
      retrieval?.status === "ok" ? retrieval.evidence : [],
    );
    return score === null ? [] : [score];
  });
  const contextPrecision = bundle.queries.flatMap((query) => {
    const retrieval = retrievalByQuery.get(query.id);
    const score = contextPrecisionForQuery(
      query,
      retrieval?.status === "ok" ? retrieval.evidence : [],
    );
    return score === null ? [] : [score];
  });
  const ndcg = bundle.queries.flatMap((query) => {
    const retrieval = retrievalByQuery.get(query.id);
    if (!retrieval || retrieval.status !== "ok") return [];
    const score = ndcgForQuery(query, retrieval.evidence);
    return score === null ? [] : [score];
  });
  const validJudgements = judgements.filter(hasJudgeOutput);
  const correctness = validJudgements.map((result) => result.output.answerCorrectness);
  const claimPrecision = validJudgements.flatMap((result) =>
    result.output.claims.length > 0
      ? [binaryFraction(result.output.claims.map((claim) => claim.correct))]
      : [],
  );
  const claimSupportPrecision = validJudgements.flatMap((result) =>
    result.output.claims.length > 0
      ? [binaryFraction(result.output.claims.map((claim) => claim.supported))]
      : [],
  );
  const completeness = validJudgements.map((result) => result.output.completeness);
  const claimF1 = validJudgements.flatMap((result) => {
    if (result.output.claims.length === 0) return [];
    const precision = binaryFraction(result.output.claims.map((claim) => claim.correct));
    return [harmonicMean(precision, result.output.completeness)];
  });
  const faithfulness = validJudgements.map((result) => result.output.strictFaithfulness);
  const citationPrecision = validJudgements.map((result) => result.output.citationPrecision);
  const citationRecall = validJudgements.map((result) => result.output.citationRecall);
  const citationF1 = validJudgements.map((result) =>
    harmonicMean(result.output.citationPrecision, result.output.citationRecall),
  );
  const acceptableAbstention = validJudgements
    .filter((result) => queryById.get(result.queryId)?.answerable === false)
    .map((result) => (result.output.acceptableAbstention ? 1 : 0));
  const answerabilityJointAccuracy = answerabilityJointAccuracyForResults(
    evaluationQueries,
    answerByQuery,
    judgeByQuery,
  );
  const robustnessDrop = robustnessDropForQueries(evaluationQueries, judgeByQuery);
  const clarity = validJudgements.flatMap((result) =>
    result.output.clarity === undefined ? [] : [result.output.clarity],
  );
  const conciseness = validJudgements.flatMap((result) =>
    result.output.conciseness === undefined ? [] : [result.output.conciseness],
  );
  const fluency = validJudgements.flatMap((result) =>
    result.output.fluency === undefined ? [] : [result.output.fluency],
  );
  const cragTruthfulness =
    bundle.id === "crag"
      ? bundle.queries.flatMap((query) => {
          const answer = answerByQuery.get(query.id);
          const judge = judgeByQuery.get(query.id);
          if (!answer || answer.status !== "ok" || !judge?.output) return [];
          if (answer.output.abstained) return [0];
          return [judge.output.answerCorrectness >= 0.5 ? 1 : -1];
        })
      : [];
  const retrievalLatencies = retrievals
    .filter((result) => result.status === "ok")
    .map((result) => result.latencyMs);
  const queryToAnswerLatencies = evaluationQueries.flatMap((query) => {
    const retrieval = retrievalByQuery.get(query.id);
    const answer = answerByQuery.get(query.id);
    if (!retrieval || !answer || retrieval.status !== "ok" || answer.status !== "ok") return [];
    return [retrieval.latencyMs + answer.latencyMs];
  });
  const endToEndLatencies = evaluationQueries.flatMap((query) => {
    const retrieval = retrievalByQuery.get(query.id);
    const answer = answerByQuery.get(query.id);
    const judge = judgeByQuery.get(query.id);
    if (!retrieval || !answer || !judge || combinedStatus(retrieval, answer, judge) !== "ok")
      return [];
    return [retrieval.latencyMs + answer.latencyMs + judge.latencyMs];
  });
  const inputTokens = [
    ...retrievals.map((result) => result.inputTokens),
    ...answers.map((result) => result.inputTokens),
    ...judgements.map((result) => result.inputTokens),
  ].filter((value): value is number => value !== null);

  return {
    datasetId: bundle.id,
    frameworkId,
    retrievalQueries: bundle.queries.length,
    retrievalCompleted: retrievalStatuses.filter((status) => status === "ok").length,
    retrievalBlocked: retrievalStatuses.filter((status) => status === "blocked").length,
    retrievalUnsupported: retrievalStatuses.filter((status) => status === "unsupported").length,
    retrievalErrors: retrievalStatuses.filter((status) => status === "error").length,
    queries: evaluationQueries.length,
    completed: statuses.filter((status) => status === "ok").length,
    blocked: statuses.filter((status) => status === "blocked").length,
    unsupported: statuses.filter((status) => status === "unsupported").length,
    errors: statuses.filter((status) => status === "error").length,
    evidenceRecallAtK: meanOrNull(evidenceRecall),
    ndcgAtK: meanOrNull(ndcg),
    contextPrecision: meanOrNull(contextPrecision),
    answerCorrectness: meanOrNull(correctness),
    answerCorrectness95Ci:
      correctness.length > 0 ? bootstrapMean95Ci(correctness, 2_000, 42) : null,
    claimPrecision: meanOrNull(claimPrecision),
    claimRecall: meanOrNull(completeness),
    claimSupportPrecision: meanOrNull(claimSupportPrecision),
    claimF1: meanOrNull(claimF1),
    completeness: meanOrNull(completeness),
    strictFaithfulness: meanOrNull(faithfulness),
    citationPrecision: meanOrNull(citationPrecision),
    citationRecall: meanOrNull(citationRecall),
    citationF1: meanOrNull(citationF1),
    acceptableAbstention: meanOrNull(acceptableAbstention),
    answerabilityJointAccuracy,
    robustnessDrop,
    clarity: meanOrNull(clarity),
    conciseness: meanOrNull(conciseness),
    fluency: meanOrNull(fluency),
    cragTruthfulness: meanOrNull(cragTruthfulness),
    retrievalLatencyP95Ms: nearestRankPercentileOrNull(retrievalLatencies, 0.95),
    queryToAnswerLatencyP95Ms: nearestRankPercentileOrNull(queryToAnswerLatencies, 0.95),
    endToEndLatencyP95Ms: nearestRankPercentileOrNull(endToEndLatencies, 0.95),
    averageInputTokens: meanOrNull(inputTokens),
    estimatedCostUsd: null,
  };
}

export function compareFrameworkPairs(
  bundle: DatasetBundle,
  frameworkIds: readonly FrameworkId[],
  retrievals: readonly RetrievalResult[],
  judgements: readonly JudgeResult[],
): PairedFrameworkComparison[] {
  const retrievalByFramework = groupByFrameworkAndQuery(retrievals);
  const judgementByFramework = groupByFrameworkAndQuery(judgements);
  const output: PairedFrameworkComparison[] = [];
  for (const [leftIndex, leftFrameworkId] of frameworkIds.entries()) {
    for (const rightFrameworkId of frameworkIds.slice(leftIndex + 1)) {
      const metricPairs = new Map<PairedFrameworkComparison["metric"], [number, number][]>();
      for (const query of bundle.queries) {
        const leftRetrieval = retrievalByFramework.get(leftFrameworkId)?.get(query.id);
        const rightRetrieval = retrievalByFramework.get(rightFrameworkId)?.get(query.id);
        if (leftRetrieval?.status === "ok" && rightRetrieval?.status === "ok") {
          const leftRecall = evidenceRecallForQuery(query, leftRetrieval.evidence);
          const rightRecall = evidenceRecallForQuery(query, rightRetrieval.evidence);
          if (leftRecall !== null && rightRecall !== null) {
            addPair(metricPairs, "evidence-recall-at-k", leftRecall, rightRecall);
          }
          const leftNdcg = ndcgForQuery(query, leftRetrieval.evidence);
          const rightNdcg = ndcgForQuery(query, rightRetrieval.evidence);
          if (leftNdcg !== null && rightNdcg !== null) {
            addPair(metricPairs, "ndcg-at-k", leftNdcg, rightNdcg);
          }
        }
        const leftJudge = judgementByFramework.get(leftFrameworkId)?.get(query.id);
        const rightJudge = judgementByFramework.get(rightFrameworkId)?.get(query.id);
        if (
          leftJudge?.status !== "ok" ||
          !leftJudge.output ||
          rightJudge?.status !== "ok" ||
          !rightJudge.output
        ) {
          continue;
        }
        addPair(
          metricPairs,
          "answer-correctness",
          leftJudge.output.answerCorrectness,
          rightJudge.output.answerCorrectness,
        );
        addPair(
          metricPairs,
          "strict-faithfulness",
          leftJudge.output.strictFaithfulness,
          rightJudge.output.strictFaithfulness,
        );
        addPair(
          metricPairs,
          "claim-recall",
          leftJudge.output.completeness,
          rightJudge.output.completeness,
        );
        if (leftJudge.output.claims.length > 0 && rightJudge.output.claims.length > 0) {
          addPair(
            metricPairs,
            "claim-support-precision",
            binaryFraction(leftJudge.output.claims.map((claim) => claim.supported)),
            binaryFraction(rightJudge.output.claims.map((claim) => claim.supported)),
          );
        }
        addPair(
          metricPairs,
          "citation-precision",
          leftJudge.output.citationPrecision,
          rightJudge.output.citationPrecision,
        );
        addPair(
          metricPairs,
          "citation-recall",
          leftJudge.output.citationRecall,
          rightJudge.output.citationRecall,
        );
        addPair(
          metricPairs,
          "citation-f1",
          harmonicMean(leftJudge.output.citationPrecision, leftJudge.output.citationRecall),
          harmonicMean(rightJudge.output.citationPrecision, rightJudge.output.citationRecall),
        );
        addOptionalPair(
          metricPairs,
          "clarity",
          leftJudge.output.clarity,
          rightJudge.output.clarity,
        );
        addOptionalPair(
          metricPairs,
          "conciseness",
          leftJudge.output.conciseness,
          rightJudge.output.conciseness,
        );
        addOptionalPair(
          metricPairs,
          "fluency",
          leftJudge.output.fluency,
          rightJudge.output.fluency,
        );
      }
      for (const [metric, pairs] of metricPairs) {
        if (pairs.length === 0) continue;
        const differences = pairs.map(([left, right]) => left - right);
        const meanDifference = meanOrNull(differences);
        if (meanDifference === null) continue;
        output.push({
          leftFrameworkId,
          rightFrameworkId,
          metric,
          pairedQueries: pairs.length,
          meanDifferenceLeftMinusRight: meanDifference,
          difference95Ci: bootstrapMean95Ci(differences, 2_000, 42),
        });
      }
    }
  }
  return output;
}

function groupByFrameworkAndQuery<T extends { frameworkId: FrameworkId; queryId: string }>(
  records: readonly T[],
): Map<FrameworkId, Map<string, T>> {
  const output = new Map<FrameworkId, Map<string, T>>();
  for (const record of records) {
    let byQuery = output.get(record.frameworkId);
    if (!byQuery) {
      byQuery = new Map();
      output.set(record.frameworkId, byQuery);
    }
    byQuery.set(record.queryId, record);
  }
  return output;
}

function addPair(
  output: Map<PairedFrameworkComparison["metric"], [number, number][]>,
  metric: PairedFrameworkComparison["metric"],
  left: number,
  right: number,
): void {
  const pairs = output.get(metric) ?? [];
  pairs.push([left, right]);
  output.set(metric, pairs);
}

function addOptionalPair(
  output: Map<PairedFrameworkComparison["metric"], [number, number][]>,
  metric: PairedFrameworkComparison["metric"],
  left: number | undefined,
  right: number | undefined,
): void {
  if (left === undefined || right === undefined) return;
  addPair(output, metric, left, right);
}

type CombinedStatus = "ok" | "blocked" | "unsupported" | "error";

type JudgeResultWithOutput = JudgeResult & { readonly output: NonNullable<JudgeResult["output"]> };

function hasJudgeOutput(result: JudgeResult): result is JudgeResultWithOutput {
  return result.status === "ok" && result.output !== null;
}

function combinedStatus(
  retrieval: RetrievalResult | undefined,
  answer: AnswerResult | undefined,
  judge: JudgeResult | undefined,
): CombinedStatus {
  const statuses = [retrieval?.status, answer?.status, judge?.status].filter(Boolean);
  if (statuses.includes("error")) return "error";
  if (statuses.includes("unsupported")) return "unsupported";
  if (statuses.includes("blocked") || statuses.length < 3) return "blocked";
  return "ok";
}

export function evidenceRecallForQuery(
  query: BenchmarkQuery,
  evidence: readonly RetrievedEvidence[],
): number | null {
  if (query.goldEvidenceText.length > 0) {
    const context = evidence.map((item) => normalize(item.text)).join(" ");
    const covered = query.goldEvidenceText.filter(
      (gold) => textCoverage(gold, context) >= 0.8,
    ).length;
    return covered / query.goldEvidenceText.length;
  }
  if (query.goldEvidenceIds.length > 0) {
    const retrieved = new Set(
      evidence.flatMap((item) => [item.id, item.sourceId, ...(item.sourceIds ?? [])]),
    );
    return (
      query.goldEvidenceIds.filter((id) => retrieved.has(id)).length / query.goldEvidenceIds.length
    );
  }
  return null;
}

export function contextPrecisionForQuery(
  query: BenchmarkQuery,
  evidence: readonly RetrievedEvidence[],
): number | null {
  if (evidence.length === 0) return 0;
  if (query.goldEvidenceText.length > 0) {
    const relevant = evidence.filter((item) =>
      query.goldEvidenceText.some((gold) => textCoverage(gold, normalize(item.text)) >= 0.5),
    ).length;
    return relevant / evidence.length;
  }
  if (query.goldEvidenceIds.length > 0) {
    const gold = new Set(query.goldEvidenceIds);
    return (
      evidence.filter(
        (item) =>
          gold.has(item.id) ||
          gold.has(item.sourceId) ||
          (item.sourceIds ?? []).some((sourceId) => gold.has(sourceId)),
      ).length / evidence.length
    );
  }
  return null;
}

/**
 * Binary-relevance nDCG over the returned ranking. A retrieved item is relevant
 * when its provenance matches a gold evidence ID, or when its text covers a
 * gold evidence unit using the same threshold as context precision.
 */
export function ndcgForQuery(
  query: BenchmarkQuery,
  evidence: readonly RetrievedEvidence[],
): number | null {
  const relevantCount =
    query.goldEvidenceText.length > 0
      ? query.goldEvidenceText.length
      : new Set(query.goldEvidenceIds).size;
  if (relevantCount === 0) return null;
  if (evidence.length === 0) return 0;

  const ranked = [...evidence].sort(
    (left, right) => left.rank - right.rank || left.id.localeCompare(right.id),
  );
  const coveredGold = new Set<string>();
  const dcg = ranked.reduce((total, item, index) => {
    const newlyCovered = matchingGoldUnits(query, item).filter((gold) => !coveredGold.has(gold));
    for (const gold of newlyCovered) coveredGold.add(gold);
    return total + newlyCovered.length / Math.log2(index + 2);
  }, 0);
  const idealRelevant = Math.min(relevantCount, ranked.length);
  let idealDcg = 0;
  for (let index = 0; index < idealRelevant; index += 1) idealDcg += 1 / Math.log2(index + 2);
  return idealDcg === 0 ? 0 : Math.min(1, dcg / idealDcg);
}

/**
 * Class-balanced handling accuracy. Answerable cases require a non-abstained,
 * at-least-half-correct answer; unanswerable cases require an acceptable
 * abstention. The metric is null unless both classes are represented.
 */
export function answerabilityJointAccuracyForResults(
  queries: readonly BenchmarkQuery[],
  answers: ReadonlyMap<string, AnswerResult>,
  judgements: ReadonlyMap<string, JudgeResult>,
): number | null {
  const answerable: number[] = [];
  const unanswerable: number[] = [];
  for (const query of queries) {
    const answer = answers.get(query.id);
    const judgement = judgements.get(query.id);
    if (answer?.status !== "ok" || judgement?.status !== "ok" || !judgement.output) continue;
    if (query.answerable) {
      answerable.push(
        !answer.output.abstained && judgement.output.answerCorrectness >= 0.5 ? 1 : 0,
      );
    } else {
      unanswerable.push(answer.output.abstained && judgement.output.acceptableAbstention ? 1 : 0);
    }
  }
  if (answerable.length === 0 || unanswerable.length === 0) return null;
  return (mean(answerable) + mean(unanswerable)) / 2;
}

/**
 * Mean non-negative correctness drop for explicitly paired perturbations.
 * Queries opt in with metadata.robustnessGroupId and metadata.robustnessRole
 * (`baseline` or `perturbed`). Untagged datasets return null.
 */
export function robustnessDropForQueries(
  queries: readonly BenchmarkQuery[],
  judgements: ReadonlyMap<string, JudgeResult>,
): number | null {
  const groups = new Map<string, { baseline: number[]; perturbed: number[] }>();
  for (const query of queries) {
    const groupId = query.metadata.robustnessGroupId;
    const role = query.metadata.robustnessRole;
    if (typeof groupId !== "string" || (role !== "baseline" && role !== "perturbed")) continue;
    const judgement = judgements.get(query.id);
    if (judgement?.status !== "ok" || !judgement.output) continue;
    const group = groups.get(groupId) ?? { baseline: [], perturbed: [] };
    group[role].push(judgement.output.answerCorrectness);
    groups.set(groupId, group);
  }

  const drops: number[] = [];
  for (const [groupId, group] of groups) {
    if (group.baseline.length !== 1) {
      throw new Error(`Robustness group ${groupId} must have exactly one completed baseline`);
    }
    if (group.perturbed.length === 0) {
      throw new Error(`Robustness group ${groupId} must have at least one completed perturbation`);
    }
    const baseline = requiredValue(group.baseline[0], `baseline for robustness group ${groupId}`);
    for (const perturbed of group.perturbed) drops.push(Math.max(0, baseline - perturbed));
  }
  return meanOrNull(drops);
}

function matchingGoldUnits(query: BenchmarkQuery, item: RetrievedEvidence): string[] {
  if (query.goldEvidenceText.length > 0) {
    return query.goldEvidenceText
      .map((gold, index) => ({ gold, key: `text:${index}` }))
      .filter(({ gold }) => textCoverage(gold, normalize(item.text)) >= 0.5)
      .map(({ key }) => key);
  }
  const gold = new Set(query.goldEvidenceIds);
  return [
    ...new Set([item.id, item.sourceId, ...(item.sourceIds ?? [])].filter((id) => gold.has(id))),
  ];
}

function normalize(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").trim();
}

function textCoverage(gold: string, candidate: string): number {
  const tokens = new Set(
    normalize(gold)
      .split(/[^\p{L}\p{N}]+/u)
      .filter((token) => token.length >= 3),
  );
  if (tokens.size === 0) return 0;
  const candidateTokens = new Set(candidate.split(/[^\p{L}\p{N}]+/u).filter(Boolean));
  let matches = 0;
  for (const token of tokens) if (candidateTokens.has(token)) matches += 1;
  return matches / tokens.size;
}

function harmonicMean(left: number, right: number): number {
  return left + right === 0 ? 0 : (2 * left * right) / (left + right);
}

function binaryFraction(values: readonly boolean[]): number {
  if (values.length === 0) throw new Error("Cannot score an empty boolean set");
  return values.filter(Boolean).length / values.length;
}

function meanOrNull(values: readonly number[]): number | null {
  return values.length === 0 ? null : mean(values);
}

function mean(values: readonly number[]): number {
  if (values.length === 0) throw new Error("Cannot average an empty set");
  return values.reduce((total, value) => total + value, 0) / values.length;
}

/**
 * Nearest-rank percentile: sort ascending and select ceil(N * p), using a
 * one-based rank. This is the frozen latency statistic for RAG evaluation.
 */
export function nearestRankPercentileOrNull(
  values: readonly number[],
  percentile: number,
): number | null {
  if (values.length === 0) return null;
  if (!Number.isFinite(percentile) || percentile <= 0 || percentile > 1)
    throw new Error("percentile must be in (0, 1]");
  const sorted = [...values].sort((left, right) => left - right);
  return requiredAt(sorted, Math.max(0, Math.ceil(sorted.length * percentile) - 1), "percentile");
}

export function bootstrapMean95Ci(
  values: readonly number[],
  samples: number,
  seed: number,
): ConfidenceInterval {
  if (values.length === 0) throw new Error("Cannot bootstrap an empty sample");
  if (!Number.isInteger(samples) || samples <= 0) {
    throw new Error("Bootstrap sample count must be a positive integer");
  }
  const random = mulberry32(seed);
  const means: number[] = [];
  for (let sample = 0; sample < samples; sample += 1) {
    let total = 0;
    for (let index = 0; index < values.length; index += 1) {
      total += requiredAt(values, Math.floor(random() * values.length), "bootstrap source");
    }
    means.push(total / values.length);
  }
  means.sort((left, right) => left - right);
  return {
    low: requiredAt(means, Math.floor(samples * 0.025), "bootstrap lower bound"),
    high: requiredAt(
      means,
      Math.min(samples - 1, Math.ceil(samples * 0.975) - 1),
      "bootstrap upper bound",
    ),
    samples,
  };
}

function requiredAt<T>(values: readonly T[], index: number, description: string): T {
  const value = values[index];
  if (value === undefined) {
    throw new RangeError(
      `${description} index ${index} is outside a collection of ${values.length}`,
    );
  }
  return value;
}

function mulberry32(seed: number): () => number {
  let value = seed;
  return () => {
    value = (value + 0x6d2b79f5) | 0;
    let output = Math.imul(value ^ (value >>> 15), 1 | value);
    output = (output + Math.imul(output ^ (output >>> 7), 61 | output)) ^ output;
    return ((output ^ (output >>> 14)) >>> 0) / 4294967296;
  };
}

function requiredValue<T>(value: T | undefined, name: string): T {
  if (value === undefined) throw new Error(`Missing ${name}`);
  return value;
}
