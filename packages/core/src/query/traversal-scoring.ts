import { createHash } from "node:crypto";

/**
 * Raw observations used by traversal scoring. Adapters should report what they
 * observed here instead of turning observations into opaque, hand-tuned scores.
 */
export interface RankedRetrievalObservation {
  readonly rank: number;
  readonly candidateCount: number;
  /** Optional native score, normalized to [0, 1] by the producing adapter. */
  readonly normalizedScore?: number;
}

export interface FanoutObservation {
  /** Number of neighbors returned to the traversal after its bounded cut. */
  readonly returnedCount: number;
  /** Total visible neighbors before the bounded cut. */
  readonly candidateCount: number;
}

export type SignalApplicability = "applicable" | "not-applicable";

export interface QueryMatchObservation {
  readonly exactMatch?: boolean;
  readonly aliasMatch?: boolean;
  readonly lexical?: RankedRetrievalObservation;
  readonly vector?: RankedRetrievalObservation;
  readonly rerankerScore?: number;
}

export type StructuralObservation =
  | { readonly kind: "deterministic" }
  | { readonly kind: "declared"; readonly weight?: number }
  | {
      readonly kind: "extracted";
      readonly confidence?: number;
      readonly extractorVersion?: string;
    }
  | { readonly kind: "inferred"; readonly confidence?: number };

export interface EvidenceSupportObservation {
  readonly activeEvidenceCount?: number;
  readonly curatedEvidenceCount?: number;
  readonly derivedEvidenceCount?: number;
  readonly distinctResourceCount?: number;
  readonly conflictCount?: number;
  readonly staleEvidenceCount?: number;
}

export interface SearchSeedObservations {
  readonly query?: QueryMatchObservation;
  readonly fallback?: boolean;
  readonly providers?: readonly string[];
}

export interface SearchEdgeObservations {
  readonly structural?: StructuralObservation;
  readonly query?: QueryMatchObservation;
  readonly support?: EvidenceSupportObservation;
  readonly fanout?: FanoutObservation;
  readonly queryApplicability?: SignalApplicability;
  readonly supportApplicability?: SignalApplicability;
}

export interface EvidenceHitObservations {
  readonly support?: EvidenceSupportObservation;
  readonly origin?: "curated" | "derived";
  readonly freshnessDays?: number;
  readonly confidence?: number;
  readonly supportApplicability?: SignalApplicability;
  readonly confidenceApplicability?: SignalApplicability;
  readonly freshnessApplicability?: SignalApplicability;
}

export type ScoreStage = "seed" | "edge" | "evidence";

export interface ScoreComputation {
  readonly stage: ScoreStage;
  readonly inputScore?: number;
  readonly score: number;
  readonly factors: Readonly<Record<string, number>>;
  readonly observations: Readonly<Record<string, number | boolean | string>>;
  readonly missingSignals: readonly string[];
}

export interface TraversalScoreBreakdown {
  readonly profileId: string;
  readonly profileVersion: number;
  readonly profileDigest: string;
  readonly seed: ScoreComputation;
  readonly edges: readonly ScoreComputation[];
  readonly evidence: ScoreComputation;
  readonly finalScore: number;
}

export interface TraversalScoringProfile {
  readonly id: string;
  readonly version: number;
  readonly featureSchemaVersion: string;
  readonly description?: string;
  readonly seed: {
    readonly exactMatchScore: number;
    readonly aliasMatchScore: number;
    readonly fallbackScore: number;
    readonly missingQueryScore: number;
    readonly observedQueryFloor: number;
    readonly lexicalWeight: number;
    readonly vectorWeight: number;
    readonly rerankerWeight: number;
  };
  readonly edge: {
    readonly hopFactor: number;
    readonly kgExpansionFactor: number;
    readonly queryFloor: number;
    readonly supportFloor: number;
    readonly missingQueryFactor: number;
    readonly missingSupportFactor: number;
    readonly structuralPriors: {
      readonly deterministic: number;
      readonly declared: number;
      readonly extracted: number;
      readonly inferred: number;
      readonly missing: number;
    };
  };
  readonly supportEncoding: {
    readonly reliabilityWeight: number;
    readonly diversityWeight: number;
    readonly volumeWeight: number;
    readonly derivedEvidenceWeight: number;
    readonly staleEvidenceWeight: number;
    readonly diversitySaturation: number;
    readonly volumeSaturation: number;
  };
  readonly evidence: {
    readonly activeFactFactor: number;
    readonly conflictFactFactor: number;
    readonly noFactFactor: number;
    readonly curatedOriginFactor: number;
    readonly derivedOriginFactor: number;
    readonly missingOriginFactor: number;
    readonly supportFloor: number;
    readonly missingSupportFactor: number;
    readonly confidenceFloor: number;
    readonly missingConfidenceFactor: number;
    readonly freshnessHalfLifeDays: number;
    readonly minimumFreshnessFactor: number;
  };
}

export interface ScorePolicyDescriptor {
  readonly profileId: string;
  readonly profileVersion: number;
  readonly profileDigest: string;
  readonly featureSchemaVersion: string;
}

export const TRAVERSAL_FEATURE_SCHEMA_VERSION = "n-layer-observations-v1" as const;

export interface SeedScoreInput {
  readonly legacyScore?: number;
  readonly observations?: SearchSeedObservations;
  readonly nodeKind?: "ontology" | "resource" | "chunk" | "entity" | "fact";
}

export interface EdgeScoreInput {
  readonly operation: "lift" | "expand" | "ground";
  readonly fromKind: "ontology" | "resource" | "chunk" | "entity" | "fact";
  readonly toKind: "ontology" | "resource" | "chunk" | "entity" | "fact";
  readonly legacyConfidence?: number;
  readonly legacyQueryRelevance?: number;
  readonly legacyEvidenceSupport?: number;
  readonly observations?: SearchEdgeObservations;
}

export interface EvidenceScoreInput {
  readonly legacyScore?: number;
  readonly factStatus?: "active" | "conflict";
  readonly observations?: EvidenceHitObservations;
}

export interface ExplainableTraversalScorePolicy {
  readonly explainable: true;
  readonly descriptor: ScorePolicyDescriptor;
  seedScore(input: SeedScoreInput): ScoreComputation;
  edgeScore(parentScore: number, input: EdgeScoreInput, hop: number): ScoreComputation;
  evidenceScore(pathScore: number, input: EvidenceScoreInput): ScoreComputation;
}

/**
 * A reusable policy can bind query-local intent once, keeping query analysis
 * inside the scoring module instead of duplicating it across graph adapters.
 */
export interface QueryAdaptiveTraversalScorePolicy {
  readonly queryAdaptive: true;
  readonly descriptor: ScorePolicyDescriptor;
  bind(question: string): ExplainableTraversalScorePolicy;
}

export const DEFAULT_CALIBRATED_SCORING_PROFILE: TraversalScoringProfile = {
  id: "calibrated-v2",
  version: 1,
  featureSchemaVersion: TRAVERSAL_FEATURE_SCHEMA_VERSION,
  description: "Observation-based monotonic traversal scoring with explicit missing signals",
  seed: {
    exactMatchScore: 1,
    aliasMatchScore: 0.94,
    fallbackScore: 0.08,
    missingQueryScore: 0,
    observedQueryFloor: 0.05,
    lexicalWeight: 1,
    vectorWeight: 1,
    rerankerWeight: 1.25,
  },
  edge: {
    hopFactor: 0.92,
    kgExpansionFactor: 0.96,
    queryFloor: 0.55,
    supportFloor: 0.65,
    missingQueryFactor: 1,
    missingSupportFactor: 1,
    structuralPriors: {
      deterministic: 1,
      declared: 0.95,
      extracted: 0.9,
      inferred: 0.78,
      missing: 0.72,
    },
  },
  supportEncoding: {
    reliabilityWeight: 0.55,
    diversityWeight: 0.25,
    volumeWeight: 0.2,
    derivedEvidenceWeight: 0.75,
    staleEvidenceWeight: 0.5,
    diversitySaturation: 2,
    volumeSaturation: 3,
  },
  evidence: {
    activeFactFactor: 1,
    conflictFactFactor: 0.55,
    noFactFactor: 0.82,
    curatedOriginFactor: 1,
    derivedOriginFactor: 0.92,
    missingOriginFactor: 1,
    supportFloor: 0.7,
    missingSupportFactor: 1,
    confidenceFloor: 0.75,
    missingConfidenceFactor: 1,
    freshnessHalfLifeDays: 180,
    minimumFreshnessFactor: 0.75,
  },
};

export class CalibratedTraversalScorePolicy implements ExplainableTraversalScorePolicy {
  readonly explainable = true as const;
  readonly descriptor: ScorePolicyDescriptor;

  constructor(readonly profile: TraversalScoringProfile = DEFAULT_CALIBRATED_SCORING_PROFILE) {
    validateTraversalScoringProfile(profile);
    this.descriptor = {
      profileId: profile.id,
      profileVersion: profile.version,
      profileDigest: scoringProfileDigest(profile),
      featureSchemaVersion: profile.featureSchemaVersion,
    };
  }

  seedScore(input: SeedScoreInput): ScoreComputation {
    const missingSignals: string[] = [];
    const observations: Record<string, number | boolean | string> = {};
    const query = input.observations?.query;
    let score: number;

    if (query?.exactMatch) {
      observations.exactMatch = true;
      score = this.profile.seed.exactMatchScore;
    } else if (query?.aliasMatch) {
      observations.aliasMatch = true;
      score = this.profile.seed.aliasMatchScore;
    } else {
      const combined = combineQueryObservations(query, this.profile.seed, observations);
      if (combined === undefined) {
        missingSignals.push("seed.query");
        score = input.observations?.fallback
          ? this.profile.seed.fallbackScore
          : this.profile.seed.missingQueryScore;
      } else {
        score = interpolate(this.profile.seed.observedQueryFloor, combined);
      }
    }

    if (input.observations?.fallback) observations.fallback = true;
    return computation("seed", undefined, score, { query: score }, observations, missingSignals);
  }

  edgeScore(parentScore: number, input: EdgeScoreInput, _hop: number): ScoreComputation {
    const missingSignals: string[] = [];
    const observations: Record<string, number | boolean | string> = {};
    const structural = structuralFactor(
      input.observations?.structural,
      this.profile.edge.structuralPriors,
      observations,
      missingSignals,
    );
    const queryMatch = combineQueryObservations(
      input.observations?.query,
      this.profile.seed,
      observations,
    );
    const queryNotApplicable = input.observations?.queryApplicability === "not-applicable";
    const query =
      queryMatch === undefined
        ? queryNotApplicable
          ? 1
          : this.profile.edge.missingQueryFactor
        : interpolate(this.profile.edge.queryFloor, queryMatch);
    if (queryMatch === undefined && !queryNotApplicable) missingSignals.push("edge.query");
    if (queryNotApplicable) observations.queryApplicability = "not-applicable";

    const observedSupport = supportStrength(
      input.observations?.support,
      observations,
      this.profile.supportEncoding,
    );
    const supportNotApplicable = input.observations?.supportApplicability === "not-applicable";
    const support =
      observedSupport === undefined
        ? supportNotApplicable
          ? 1
          : this.profile.edge.missingSupportFactor
        : interpolate(this.profile.edge.supportFloor, observedSupport);
    if (observedSupport === undefined && !supportNotApplicable) missingSignals.push("edge.support");
    if (supportNotApplicable) observations.supportApplicability = "not-applicable";

    const kgExpansion = isKgExpansion(input) ? this.profile.edge.kgExpansionFactor : 1;
    const factors = {
      structural,
      query,
      support,
      hop: this.profile.edge.hopFactor,
      kgExpansion,
    };
    const score = multiplyInLogSpace(parentScore, Object.values(factors));
    return computation("edge", parentScore, score, factors, observations, missingSignals);
  }

  evidenceScore(pathScore: number, input: EvidenceScoreInput): ScoreComputation {
    const missingSignals: string[] = [];
    const observations: Record<string, number | boolean | string> = {};
    const fact =
      input.factStatus === "active"
        ? this.profile.evidence.activeFactFactor
        : input.factStatus === "conflict"
          ? this.profile.evidence.conflictFactFactor
          : this.profile.evidence.noFactFactor;
    observations.factStatus = input.factStatus ?? "none";

    const origin = input.observations?.origin;
    const originFactor =
      origin === "curated"
        ? this.profile.evidence.curatedOriginFactor
        : origin === "derived"
          ? this.profile.evidence.derivedOriginFactor
          : this.profile.evidence.missingOriginFactor;
    if (origin === undefined) missingSignals.push("evidence.origin");
    else observations.origin = origin;

    const observedSupport = supportStrength(
      input.observations?.support,
      observations,
      this.profile.supportEncoding,
    );
    const supportNotApplicable = input.observations?.supportApplicability === "not-applicable";
    const support =
      observedSupport === undefined
        ? supportNotApplicable
          ? 1
          : this.profile.evidence.missingSupportFactor
        : interpolate(this.profile.evidence.supportFloor, observedSupport);
    if (observedSupport === undefined && !supportNotApplicable) {
      missingSignals.push("evidence.support");
    }
    if (supportNotApplicable) observations.supportApplicability = "not-applicable";

    const observedConfidence = input.observations?.confidence;
    const confidenceNotApplicable =
      input.observations?.confidenceApplicability === "not-applicable";
    const confidence =
      observedConfidence === undefined || !Number.isFinite(observedConfidence)
        ? confidenceNotApplicable
          ? 1
          : this.profile.evidence.missingConfidenceFactor
        : interpolate(this.profile.evidence.confidenceFloor, observedConfidence);
    if (
      (observedConfidence === undefined || !Number.isFinite(observedConfidence)) &&
      !confidenceNotApplicable
    ) {
      missingSignals.push("evidence.confidence");
    } else {
      if (confidenceNotApplicable) observations.confidenceApplicability = "not-applicable";
      else observations.confidence = clampScore(observedConfidence ?? 0);
    }

    const freshnessDays = input.observations?.freshnessDays;
    let freshness = 1;
    const freshnessNotApplicable = input.observations?.freshnessApplicability === "not-applicable";
    if (
      (freshnessDays === undefined || !Number.isFinite(freshnessDays)) &&
      !freshnessNotApplicable
    ) {
      missingSignals.push("evidence.freshnessDays");
    } else if (freshnessNotApplicable) {
      observations.freshnessApplicability = "not-applicable";
    } else {
      observations.freshnessDays = Math.max(0, freshnessDays ?? 0);
      freshness = Math.max(
        this.profile.evidence.minimumFreshnessFactor,
        2 ** (-Math.max(0, freshnessDays ?? 0) / this.profile.evidence.freshnessHalfLifeDays),
      );
    }

    const factors = { fact, origin: originFactor, support, confidence, freshness };
    const score = multiplyInLogSpace(pathScore, Object.values(factors));
    return computation("evidence", pathScore, score, factors, observations, missingSignals);
  }
}

export function validateTraversalScoringProfile(profile: TraversalScoringProfile): void {
  if (typeof profile?.id !== "string" || profile.id.trim().length === 0) {
    throw new Error("Scoring profile id must not be empty");
  }
  if (!Number.isInteger(profile.version) || profile.version < 1) {
    throw new Error("Scoring profile version must be a positive integer");
  }
  if (
    typeof profile.featureSchemaVersion !== "string" ||
    profile.featureSchemaVersion.trim().length === 0
  ) {
    throw new Error("Scoring profile featureSchemaVersion must not be empty");
  }
  assertExactKeys(profile, "profile", [
    "id",
    "version",
    "featureSchemaVersion",
    "description",
    "seed",
    "edge",
    "supportEncoding",
    "evidence",
  ]);
  assertExactKeys(profile.seed, "profile.seed", [
    "exactMatchScore",
    "aliasMatchScore",
    "fallbackScore",
    "missingQueryScore",
    "observedQueryFloor",
    "lexicalWeight",
    "vectorWeight",
    "rerankerWeight",
  ]);
  assertExactKeys(profile.edge, "profile.edge", [
    "hopFactor",
    "kgExpansionFactor",
    "queryFloor",
    "supportFloor",
    "missingQueryFactor",
    "missingSupportFactor",
    "structuralPriors",
  ]);
  assertExactKeys(profile.edge?.structuralPriors, "profile.edge.structuralPriors", [
    "deterministic",
    "declared",
    "extracted",
    "inferred",
    "missing",
  ]);
  assertExactKeys(profile.supportEncoding, "profile.supportEncoding", [
    "reliabilityWeight",
    "diversityWeight",
    "volumeWeight",
    "derivedEvidenceWeight",
    "staleEvidenceWeight",
    "diversitySaturation",
    "volumeSaturation",
  ]);
  assertExactKeys(profile.evidence, "profile.evidence", [
    "activeFactFactor",
    "conflictFactFactor",
    "noFactFactor",
    "curatedOriginFactor",
    "derivedOriginFactor",
    "missingOriginFactor",
    "supportFloor",
    "missingSupportFactor",
    "confidenceFloor",
    "missingConfidenceFactor",
    "freshnessHalfLifeDays",
    "minimumFreshnessFactor",
  ]);
  const requiredNumbers: ReadonlyArray<readonly [string, unknown]> = [
    ["profile.seed.exactMatchScore", profile.seed?.exactMatchScore],
    ["profile.seed.aliasMatchScore", profile.seed?.aliasMatchScore],
    ["profile.seed.fallbackScore", profile.seed?.fallbackScore],
    ["profile.seed.missingQueryScore", profile.seed?.missingQueryScore],
    ["profile.seed.observedQueryFloor", profile.seed?.observedQueryFloor],
    ["profile.seed.lexicalWeight", profile.seed?.lexicalWeight],
    ["profile.seed.vectorWeight", profile.seed?.vectorWeight],
    ["profile.seed.rerankerWeight", profile.seed?.rerankerWeight],
    ["profile.edge.hopFactor", profile.edge?.hopFactor],
    ["profile.edge.kgExpansionFactor", profile.edge?.kgExpansionFactor],
    ["profile.edge.queryFloor", profile.edge?.queryFloor],
    ["profile.edge.supportFloor", profile.edge?.supportFloor],
    ["profile.edge.missingQueryFactor", profile.edge?.missingQueryFactor],
    ["profile.edge.missingSupportFactor", profile.edge?.missingSupportFactor],
    ["profile.edge.structuralPriors.deterministic", profile.edge?.structuralPriors?.deterministic],
    ["profile.edge.structuralPriors.declared", profile.edge?.structuralPriors?.declared],
    ["profile.edge.structuralPriors.extracted", profile.edge?.structuralPriors?.extracted],
    ["profile.edge.structuralPriors.inferred", profile.edge?.structuralPriors?.inferred],
    ["profile.edge.structuralPriors.missing", profile.edge?.structuralPriors?.missing],
    ["profile.supportEncoding.reliabilityWeight", profile.supportEncoding?.reliabilityWeight],
    ["profile.supportEncoding.diversityWeight", profile.supportEncoding?.diversityWeight],
    ["profile.supportEncoding.volumeWeight", profile.supportEncoding?.volumeWeight],
    [
      "profile.supportEncoding.derivedEvidenceWeight",
      profile.supportEncoding?.derivedEvidenceWeight,
    ],
    ["profile.supportEncoding.staleEvidenceWeight", profile.supportEncoding?.staleEvidenceWeight],
    ["profile.supportEncoding.diversitySaturation", profile.supportEncoding?.diversitySaturation],
    ["profile.supportEncoding.volumeSaturation", profile.supportEncoding?.volumeSaturation],
    ["profile.evidence.activeFactFactor", profile.evidence?.activeFactFactor],
    ["profile.evidence.conflictFactFactor", profile.evidence?.conflictFactFactor],
    ["profile.evidence.noFactFactor", profile.evidence?.noFactFactor],
    ["profile.evidence.curatedOriginFactor", profile.evidence?.curatedOriginFactor],
    ["profile.evidence.derivedOriginFactor", profile.evidence?.derivedOriginFactor],
    ["profile.evidence.missingOriginFactor", profile.evidence?.missingOriginFactor],
    ["profile.evidence.supportFloor", profile.evidence?.supportFloor],
    ["profile.evidence.missingSupportFactor", profile.evidence?.missingSupportFactor],
    ["profile.evidence.confidenceFloor", profile.evidence?.confidenceFloor],
    ["profile.evidence.missingConfidenceFactor", profile.evidence?.missingConfidenceFactor],
    ["profile.evidence.freshnessHalfLifeDays", profile.evidence?.freshnessHalfLifeDays],
    ["profile.evidence.minimumFreshnessFactor", profile.evidence?.minimumFreshnessFactor],
  ];
  for (const [path, value] of requiredNumbers) {
    if (typeof value !== "number") throw new Error(`${path} is required and must be a number`);
  }
  const supportWeightSum =
    profile.supportEncoding.reliabilityWeight +
    profile.supportEncoding.diversityWeight +
    profile.supportEncoding.volumeWeight;
  if (Math.abs(supportWeightSum - 1) > 1e-9) {
    throw new Error("Support encoding reliability, diversity, and volume weights must sum to one");
  }
  for (const [path, value] of [
    [
      "profile.supportEncoding.derivedEvidenceWeight",
      profile.supportEncoding.derivedEvidenceWeight,
    ],
    ["profile.supportEncoding.staleEvidenceWeight", profile.supportEncoding.staleEvidenceWeight],
  ] as const) {
    if (value < 0 || value > 1) throw new Error(`${path} must be between zero and one`);
  }
  visitNumbers(profile, "profile", (path, value) => {
    if (!Number.isFinite(value)) throw new Error(`${path} must be finite`);
    if (path === "profile.version") return;
    if (path.endsWith("Weight")) {
      if (value <= 0) throw new Error(`${path} must be greater than zero`);
      return;
    }
    if (path.endsWith("freshnessHalfLifeDays")) {
      if (value <= 0) throw new Error(`${path} must be greater than zero`);
      return;
    }
    if (path.endsWith("Saturation")) {
      if (value <= 0) throw new Error(`${path} must be greater than zero`);
      return;
    }
    if (value < 0 || value > 1) throw new Error(`${path} must be between zero and one`);
  });
}

export function scoringProfileDigest(profile: TraversalScoringProfile): string {
  return `sha256:${createHash("sha256").update(canonicalJson(profile)).digest("hex")}`;
}

function combineQueryObservations(
  query: QueryMatchObservation | undefined,
  weights: TraversalScoringProfile["seed"],
  output: Record<string, number | boolean | string>,
): number | undefined {
  if (!query) return undefined;
  if (query.exactMatch) {
    output.exactMatch = true;
    return weights.exactMatchScore;
  }
  if (query.aliasMatch) {
    output.aliasMatch = true;
    return weights.aliasMatchScore;
  }
  const values: Array<{ value: number; weight: number }> = [];
  const lexical = rankedObservationScore(query.lexical);
  if (lexical !== undefined) {
    output.lexical = lexical;
    recordRankedObservation("lexical", query.lexical, output);
    values.push({ value: lexical, weight: weights.lexicalWeight });
  }
  const vector = rankedObservationScore(query.vector);
  if (vector !== undefined) {
    output.vector = vector;
    recordRankedObservation("vector", query.vector, output);
    values.push({ value: vector, weight: weights.vectorWeight });
  }
  if (query.rerankerScore !== undefined) {
    const reranker = clampScore(query.rerankerScore);
    output.reranker = reranker;
    values.push({ value: reranker, weight: weights.rerankerWeight });
  }
  if (values.length === 0) return undefined;
  const totalWeight = values.reduce((sum, item) => sum + item.weight, 0);
  return values.reduce((sum, item) => sum + item.value * item.weight, 0) / totalWeight;
}

function recordRankedObservation(
  prefix: "lexical" | "vector",
  observation: RankedRetrievalObservation | undefined,
  output: Record<string, number | boolean | string>,
): void {
  if (!observation) return;
  output[`${prefix}Rank`] = finitePositiveInteger(observation.rank);
  output[`${prefix}CandidateCount`] = finitePositiveInteger(observation.candidateCount);
  if (observation.normalizedScore !== undefined) {
    output[`${prefix}NormalizedScore`] = clampScore(observation.normalizedScore);
  }
}

function rankedObservationScore(
  observation: RankedRetrievalObservation | undefined,
): number | undefined {
  if (!observation) return undefined;
  const candidateCount = finitePositiveInteger(observation.candidateCount);
  const rank = Math.min(candidateCount, finitePositiveInteger(observation.rank));
  // Multi-provider fusion uses ordinal position, not incomparable provider
  // score scales. A normalized score is only used when there is no ranking
  // distribution (for example, token overlap on a single graph edge).
  if (candidateCount === 1 && observation.normalizedScore !== undefined) {
    return clampScore(observation.normalizedScore);
  }
  return candidateCount === 1 ? 1 : (candidateCount - rank) / (candidateCount - 1);
}

function finitePositiveInteger(value: number): number {
  return Number.isFinite(value) ? Math.max(1, Math.floor(value)) : 1;
}

function structuralFactor(
  structural: StructuralObservation | undefined,
  priors: TraversalScoringProfile["edge"]["structuralPriors"],
  output: Record<string, number | boolean | string>,
  missingSignals: string[],
): number {
  if (!structural) {
    missingSignals.push("edge.structural");
    return priors.missing;
  }
  output.structuralKind = structural.kind;
  if (structural.kind === "deterministic") return priors.deterministic;
  if (structural.kind === "declared") {
    if (structural.weight === undefined) {
      missingSignals.push("edge.structural.weight");
      return priors.declared;
    }
    output.structuralWeight = clampScore(structural.weight);
    return priors.declared * clampScore(structural.weight);
  }
  const confidence = structural.confidence;
  if (confidence === undefined) {
    missingSignals.push("edge.structural.confidence");
    return priors[structural.kind];
  }
  output.structuralConfidence = clampScore(confidence);
  if (structural.kind === "extracted" && structural.extractorVersion) {
    output.extractorVersion = structural.extractorVersion;
  }
  return priors[structural.kind] * clampScore(confidence);
}

function supportStrength(
  support: EvidenceSupportObservation | undefined,
  output: Record<string, number | boolean | string>,
  encoding: TraversalScoringProfile["supportEncoding"],
): number | undefined {
  if (!support) return undefined;
  const active = nonNegative(support.activeEvidenceCount);
  const curated = nonNegative(support.curatedEvidenceCount);
  const derived = nonNegative(support.derivedEvidenceCount);
  const resources = nonNegative(support.distinctResourceCount);
  const conflicts = nonNegative(support.conflictCount);
  const stale = nonNegative(support.staleEvidenceCount);
  const present = [active, curated, derived, resources, conflicts, stale].some(
    (value) => value !== undefined,
  );
  if (!present) return undefined;

  const hasOriginCounts = curated !== undefined || derived !== undefined;
  const positiveCount = hasOriginCounts
    ? (curated ?? 0) + encoding.derivedEvidenceWeight * (derived ?? 0)
    : (active ?? 0);
  const negativeCount = (conflicts ?? 0) + encoding.staleEvidenceWeight * (stale ?? 0);
  const reliability =
    positiveCount + negativeCount === 0 ? 0 : positiveCount / (positiveCount + negativeCount);
  const diversity = 1 - Math.exp(-(resources ?? 0) / encoding.diversitySaturation);
  const volume = 1 - Math.exp(-positiveCount / encoding.volumeSaturation);
  const score =
    encoding.reliabilityWeight * reliability +
    encoding.diversityWeight * diversity +
    encoding.volumeWeight * volume;

  if (active !== undefined) output.activeEvidenceCount = active;
  if (curated !== undefined) output.curatedEvidenceCount = curated;
  if (derived !== undefined) output.derivedEvidenceCount = derived;
  if (resources !== undefined) output.distinctResourceCount = resources;
  if (conflicts !== undefined) output.conflictCount = conflicts;
  if (stale !== undefined) output.staleEvidenceCount = stale;
  return clampScore(score);
}

function computation(
  stage: ScoreStage,
  inputScore: number | undefined,
  score: number,
  factors: Record<string, number>,
  observations: Record<string, number | boolean | string>,
  missingSignals: readonly string[],
): ScoreComputation {
  return {
    stage,
    ...(inputScore === undefined ? {} : { inputScore: clampScore(inputScore) }),
    score: clampScore(score),
    factors,
    observations,
    missingSignals,
  };
}

function isKgExpansion(input: EdgeScoreInput): boolean {
  return (
    input.operation === "expand" &&
    ([input.fromKind, input.toKind].includes("entity") ||
      [input.fromKind, input.toKind].includes("fact"))
  );
}

function interpolate(floor: number, value: number): number {
  return floor + (1 - floor) * clampScore(value);
}

function nonNegative(value: number | undefined): number | undefined {
  if (value === undefined || !Number.isFinite(value)) return undefined;
  return Math.max(0, value);
}

function clampScore(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function multiplyInLogSpace(inputScore: number, factors: readonly number[]): number {
  const values = [clampScore(inputScore), ...factors.map(clampScore)];
  if (values.some((value) => value === 0)) return 0;
  return clampScore(Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0)));
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value)
      .filter(([, item]) => item !== undefined)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function visitNumbers(
  value: unknown,
  path: string,
  visitor: (path: string, value: number) => void,
): void {
  if (typeof value === "number") {
    visitor(path, value);
    return;
  }
  if (!value || typeof value !== "object") return;
  for (const [key, child] of Object.entries(value)) visitNumbers(child, `${path}.${key}`, visitor);
}

function assertExactKeys(value: unknown, path: string, allowed: readonly string[]): void {
  if (!value || typeof value !== "object" || Array.isArray(value)) return;
  const allowedKeys = new Set(allowed);
  for (const key of Object.keys(value)) {
    if (!allowedKeys.has(key)) throw new Error(`${path}.${key} is not a recognized profile field`);
  }
}
