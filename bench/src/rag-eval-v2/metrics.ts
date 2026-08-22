import type {
  AnswerResult,
  BenchmarkQuery,
  DatasetBundle,
  FrameworkId,
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
  readonly contextPrecision: number | null;
  readonly answerCorrectness: number | null;
  readonly answerCorrectness95Ci: ConfidenceInterval | null;
  readonly claimPrecision: number | null;
  readonly claimRecall: number | null;
  readonly claimF1: number | null;
  readonly completeness: number | null;
  readonly strictFaithfulness: number | null;
  readonly citationF1: number | null;
  readonly acceptableAbstention: number | null;
  readonly cragTruthfulness: number | null;
  readonly retrievalLatencyP95Ms: number | null;
  readonly endToEndLatencyP95Ms: number | null;
  readonly averageInputTokens: number | null;
  readonly estimatedCostUsd: null;
}

export interface PairedFrameworkComparison {
  readonly leftFrameworkId: FrameworkId;
  readonly rightFrameworkId: FrameworkId;
  readonly metric: "evidence-recall-at-k" | "answer-correctness" | "strict-faithfulness" | "citation-f1";
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
  const retrievalStatuses = bundle.queries.map((query) => retrievalByQuery.get(query.id)?.status ?? "blocked");
  const statuses = evaluationQueries.map((query) => combinedStatus(
    retrievalByQuery.get(query.id),
    answerByQuery.get(query.id),
    judgeByQuery.get(query.id),
  ));

  const evidenceRecall = bundle.queries.flatMap((query) => {
    const retrieval = retrievalByQuery.get(query.id);
    if (!retrieval || retrieval.status !== "ok") return [];
    const score = evidenceRecallForQuery(query, retrieval.evidence);
    return score === null ? [] : [score];
  });
  const contextPrecision = bundle.queries.flatMap((query) => {
    const retrieval = retrievalByQuery.get(query.id);
    if (!retrieval || retrieval.status !== "ok") return [];
    const score = contextPrecisionForQuery(query, retrieval.evidence);
    return score === null ? [] : [score];
  });
  const validJudgements = judgements.filter((result) => result.status === "ok" && result.output);
  const correctness = validJudgements.map((result) => result.output!.answerCorrectness);
  const claimPrecision = validJudgements.flatMap((result) => result.output!.claims.length > 0
    ? [meanOrNull(result.output!.claims.map((claim) => (claim.correct ? 1 : 0)))!]
    : []);
  const completeness = validJudgements.map((result) => result.output!.completeness);
  const claimF1 = validJudgements.flatMap((result) => {
    if (result.output!.claims.length === 0) return [];
    const precision = meanOrNull(result.output!.claims.map((claim) => (claim.correct ? 1 : 0)))!;
    return [harmonicMean(precision, result.output!.completeness)];
  });
  const faithfulness = validJudgements.map((result) => result.output!.strictFaithfulness);
  const citationF1 = validJudgements.map((result) => harmonicMean(
    result.output!.citationPrecision,
    result.output!.citationRecall,
  ));
  const acceptableAbstention = validJudgements
    .filter((result) => queryById.get(result.queryId)?.answerable === false)
    .map((result) => (result.output!.acceptableAbstention ? 1 : 0));
  const cragTruthfulness = bundle.id === "crag"
    ? bundle.queries.flatMap((query) => {
        const answer = answerByQuery.get(query.id);
        const judge = judgeByQuery.get(query.id);
        if (!answer || answer.status !== "ok" || !judge?.output) return [];
        if (answer.output.abstained) return [0];
        return [judge.output.answerCorrectness >= 0.5 ? 1 : -1];
      })
    : [];
  const retrievalLatencies = retrievals.filter((result) => result.status === "ok").map((result) => result.latencyMs);
  const endToEndLatencies = evaluationQueries.flatMap((query) => {
    const retrieval = retrievalByQuery.get(query.id);
    const answer = answerByQuery.get(query.id);
    const judge = judgeByQuery.get(query.id);
    if (!retrieval || !answer || !judge || combinedStatus(retrieval, answer, judge) !== "ok") return [];
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
    contextPrecision: meanOrNull(contextPrecision),
    answerCorrectness: meanOrNull(correctness),
    answerCorrectness95Ci: correctness.length > 0 ? bootstrapMean95Ci(correctness, 2_000, 42) : null,
    claimPrecision: meanOrNull(claimPrecision),
    claimRecall: meanOrNull(completeness),
    claimF1: meanOrNull(claimF1),
    completeness: meanOrNull(completeness),
    strictFaithfulness: meanOrNull(faithfulness),
    citationF1: meanOrNull(citationF1),
    acceptableAbstention: meanOrNull(acceptableAbstention),
    cragTruthfulness: meanOrNull(cragTruthfulness),
    retrievalLatencyP95Ms: percentileOrNull(retrievalLatencies, 0.95),
    endToEndLatencyP95Ms: percentileOrNull(endToEndLatencies, 0.95),
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
  for (let leftIndex = 0; leftIndex < frameworkIds.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < frameworkIds.length; rightIndex += 1) {
      const leftFrameworkId = frameworkIds[leftIndex]!;
      const rightFrameworkId = frameworkIds[rightIndex]!;
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
        }
        const leftJudge = judgementByFramework.get(leftFrameworkId)?.get(query.id);
        const rightJudge = judgementByFramework.get(rightFrameworkId)?.get(query.id);
        if (leftJudge?.status !== "ok" || !leftJudge.output || rightJudge?.status !== "ok" || !rightJudge.output) {
          continue;
        }
        addPair(metricPairs, "answer-correctness", leftJudge.output.answerCorrectness, rightJudge.output.answerCorrectness);
        addPair(metricPairs, "strict-faithfulness", leftJudge.output.strictFaithfulness, rightJudge.output.strictFaithfulness);
        addPair(
          metricPairs,
          "citation-f1",
          harmonicMean(leftJudge.output.citationPrecision, leftJudge.output.citationRecall),
          harmonicMean(rightJudge.output.citationPrecision, rightJudge.output.citationRecall),
        );
      }
      for (const [metric, pairs] of metricPairs) {
        if (pairs.length === 0) continue;
        const differences = pairs.map(([left, right]) => left - right);
        output.push({
          leftFrameworkId,
          rightFrameworkId,
          metric,
          pairedQueries: pairs.length,
          meanDifferenceLeftMinusRight: meanOrNull(differences)!,
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

type CombinedStatus = "ok" | "blocked" | "unsupported" | "error";

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
    const covered = query.goldEvidenceText.filter((gold) => textCoverage(gold, context) >= 0.8).length;
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

function normalize(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").trim();
}

function textCoverage(gold: string, candidate: string): number {
  const tokens = new Set(normalize(gold).split(/[^\p{L}\p{N}]+/u).filter((token) => token.length >= 3));
  if (tokens.size === 0) return 0;
  const candidateTokens = new Set(candidate.split(/[^\p{L}\p{N}]+/u).filter(Boolean));
  let matches = 0;
  for (const token of tokens) if (candidateTokens.has(token)) matches += 1;
  return matches / tokens.size;
}

function harmonicMean(left: number, right: number): number {
  return left + right === 0 ? 0 : (2 * left * right) / (left + right);
}

function meanOrNull(values: readonly number[]): number | null {
  return values.length === 0 ? null : values.reduce((total, value) => total + value, 0) / values.length;
}

function percentileOrNull(values: readonly number[], percentile: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.max(0, Math.ceil(sorted.length * percentile) - 1)]!;
}

export function bootstrapMean95Ci(
  values: readonly number[],
  samples: number,
  seed: number,
): ConfidenceInterval {
  if (values.length === 0) throw new Error("Cannot bootstrap an empty sample");
  const random = mulberry32(seed);
  const means: number[] = [];
  for (let sample = 0; sample < samples; sample += 1) {
    let total = 0;
    for (let index = 0; index < values.length; index += 1) {
      total += values[Math.floor(random() * values.length)]!;
    }
    means.push(total / values.length);
  }
  means.sort((left, right) => left - right);
  return {
    low: means[Math.floor(samples * 0.025)]!,
    high: means[Math.min(samples - 1, Math.ceil(samples * 0.975) - 1)]!,
    samples,
  };
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
