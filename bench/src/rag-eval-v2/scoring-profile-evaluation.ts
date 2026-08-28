import {
  CalibratedTraversalScorePolicy,
  type EdgeScoreInput,
  type EvidenceScoreInput,
  type SeedScoreInput,
  type TraversalScoringProfile,
  scoringProfileDigest,
} from "@kontext-brain/core";

export type ScoringSplit = "development" | "validation" | "holdout";

export interface ScoringFeatureCandidate {
  readonly queryId: string;
  readonly category: string;
  readonly answerable: boolean;
  readonly candidateId: string;
  readonly relevant: boolean;
  readonly seed: SeedScoreInput;
  readonly edges: readonly EdgeScoreInput[];
  readonly evidence: EvidenceScoreInput;
}

export interface ScoringProfileEvaluation {
  readonly profileId: string;
  readonly profileDigest: string;
  readonly queries: number;
  readonly recallAtK: number;
  readonly ndcgAtK: number;
  readonly missingSignalRatio: number;
  readonly perQuery: Readonly<Record<string, { readonly recall: number; readonly ndcg: number }>>;
}

export interface ScoringProfileComparison {
  readonly candidate: ScoringProfileEvaluation;
  readonly recallDifference: BootstrapInterval;
  readonly ndcgDifference: BootstrapInterval;
}

export interface BootstrapInterval {
  readonly mean: number;
  readonly lower95: number;
  readonly upper95: number;
}

/** Deterministic stratification keeps tuning and final holdout queries separate. */
export function assignScoringSplits(
  candidates: readonly ScoringFeatureCandidate[],
  seed = "kontext-scoring-v1",
): ReadonlyMap<string, ScoringSplit> {
  const queryMetadata = new Map<string, Pick<ScoringFeatureCandidate, "category" | "answerable">>();
  for (const candidate of candidates) {
    queryMetadata.set(candidate.queryId, {
      category: candidate.category,
      answerable: candidate.answerable,
    });
  }
  const strata = new Map<string, string[]>();
  for (const [queryId, metadata] of queryMetadata) {
    const key = `${metadata.category}:${metadata.answerable}`;
    const values = strata.get(key) ?? [];
    values.push(queryId);
    strata.set(key, values);
  }
  const assignments = new Map<string, ScoringSplit>();
  for (const queryIds of strata.values()) {
    queryIds.sort((left, right) => stableHash(`${seed}:${left}`) - stableHash(`${seed}:${right}`));
    const developmentCount = Math.max(1, Math.floor(queryIds.length * 0.6));
    const validationCount =
      queryIds.length < 2 ? 0 : Math.max(1, Math.floor(queryIds.length * 0.2));
    queryIds.forEach((queryId, index) => {
      assignments.set(
        queryId,
        index < developmentCount
          ? "development"
          : index < developmentCount + validationCount
            ? "validation"
            : "holdout",
      );
    });
  }
  return assignments;
}

export function evaluateScoringProfile(
  profile: TraversalScoringProfile,
  candidates: readonly ScoringFeatureCandidate[],
  k = 10,
): ScoringProfileEvaluation {
  const policy = new CalibratedTraversalScorePolicy(profile);
  const byQuery = new Map<string, ScoredCandidate[]>();
  let missing = 0;
  let possibleSignals = 0;
  for (const candidate of candidates) {
    const seed = policy.seedScore(candidate.seed);
    let pathScore = seed.score;
    const computations = [seed];
    candidate.edges.forEach((edge, index) => {
      const scored = policy.edgeScore(pathScore, edge, index + 1);
      pathScore = scored.score;
      computations.push(scored);
    });
    const evidence = policy.evidenceScore(pathScore, candidate.evidence);
    computations.push(evidence);
    missing += computations.reduce((sum, item) => sum + item.missingSignals.length, 0);
    possibleSignals += 1 + candidate.edges.length * 3 + 4;
    const values = byQuery.get(candidate.queryId) ?? [];
    values.push({ id: candidate.candidateId, relevant: candidate.relevant, score: evidence.score });
    byQuery.set(candidate.queryId, values);
  }

  const perQuery: Record<string, { recall: number; ndcg: number }> = {};
  for (const [queryId, values] of byQuery) {
    values.sort((left, right) => right.score - left.score || left.id.localeCompare(right.id));
    const relevantTotal = values.filter((value) => value.relevant).length;
    const top = values.slice(0, k);
    const relevantRetrieved = top.filter((value) => value.relevant).length;
    const recall = relevantTotal === 0 ? 1 : relevantRetrieved / relevantTotal;
    const dcg = top.reduce(
      (sum, value, index) => sum + (value.relevant ? 1 / Math.log2(index + 2) : 0),
      0,
    );
    const idealCount = Math.min(k, relevantTotal);
    let idealDcg = 0;
    for (let index = 0; index < idealCount; index++) idealDcg += 1 / Math.log2(index + 2);
    perQuery[queryId] = { recall, ndcg: idealDcg === 0 ? 1 : dcg / idealDcg };
  }
  const queryMetrics = Object.values(perQuery);
  return {
    profileId: profile.id,
    profileDigest: scoringProfileDigest(profile),
    queries: queryMetrics.length,
    recallAtK: mean(queryMetrics.map((item) => item.recall)),
    ndcgAtK: mean(queryMetrics.map((item) => item.ndcg)),
    missingSignalRatio: possibleSignals === 0 ? 0 : missing / possibleSignals,
    perQuery,
  };
}

export function compareScoringProfiles(
  baseline: ScoringProfileEvaluation,
  candidate: ScoringProfileEvaluation,
  samples = 2_000,
  seed = 20_260_824,
): ScoringProfileComparison {
  const queryIds = Object.keys(baseline.perQuery).filter((id) => candidate.perQuery[id]);
  return {
    candidate,
    recallDifference: pairedBootstrap(
      queryIds.map(
        (id) => (candidate.perQuery[id]?.recall ?? 0) - (baseline.perQuery[id]?.recall ?? 0),
      ),
      samples,
      seed,
    ),
    ndcgDifference: pairedBootstrap(
      queryIds.map(
        (id) => (candidate.perQuery[id]?.ndcg ?? 0) - (baseline.perQuery[id]?.ndcg ?? 0),
      ),
      samples,
      seed + 1,
    ),
  };
}

export function selectProfileByValidation(
  evaluations: readonly ScoringProfileEvaluation[],
  baseline: ScoringProfileEvaluation,
): ScoringProfileEvaluation | null {
  return (
    [...evaluations]
      .filter((evaluation) => evaluation.recallAtK >= baseline.recallAtK - 0.01)
      .sort(
        (left, right) =>
          right.ndcgAtK - left.ndcgAtK ||
          right.recallAtK - left.recallAtK ||
          left.profileDigest.localeCompare(right.profileDigest),
      )[0] ?? null
  );
}

interface ScoredCandidate {
  readonly id: string;
  readonly relevant: boolean;
  readonly score: number;
}

function pairedBootstrap(
  differences: readonly number[],
  samples: number,
  seed: number,
): BootstrapInterval {
  if (differences.length === 0) return { mean: 0, lower95: 0, upper95: 0 };
  const random = mulberry32(seed);
  const estimates: number[] = [];
  for (let sample = 0; sample < samples; sample++) {
    let total = 0;
    for (let index = 0; index < differences.length; index++) {
      total += differences[Math.floor(random() * differences.length)] ?? 0;
    }
    estimates.push(total / differences.length);
  }
  estimates.sort((left, right) => left - right);
  return {
    mean: mean(differences),
    lower95: percentile(estimates, 0.025),
    upper95: percentile(estimates, 0.975),
  };
}

function percentile(values: readonly number[], quantile: number): number {
  const index = Math.min(values.length - 1, Math.max(0, Math.floor(quantile * values.length)));
  return values[index] ?? 0;
}

function mean(values: readonly number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function stableHash(value: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}
