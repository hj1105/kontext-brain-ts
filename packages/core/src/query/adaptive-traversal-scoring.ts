import { createHash } from "node:crypto";
import {
  CalibratedTraversalScorePolicy,
  DEFAULT_CALIBRATED_SCORING_PROFILE,
  type EdgeScoreInput,
  type EvidenceScoreInput,
  type ExplainableTraversalScorePolicy,
  type QueryAdaptiveTraversalScorePolicy,
  type QueryMatchObservation,
  type RankedRetrievalObservation,
  type ScoreComputation,
  type ScorePolicyDescriptor,
  type SeedScoreInput,
  type TraversalScoringProfile,
} from "./traversal-scoring.js";

export type AdaptiveQueryIntent = "summary" | "comparison" | "lookup";

const ADAPTIVE_ROUTE_ALGORITHM_VERSION = "adaptive-route-gate-v2";

/**
 * Query-local route adaptation behind the existing scoring interface.
 *
 * It never branches on a dataset or organization name. Broad neighbor lists
 * receive less authority than selective lists, lower list ranks decay by
 * reciprocal rank, and summary questions may intentionally traverse a broad
 * resource because broad coverage is the requested behavior.
 */
export class AdaptiveRouteTraversalScorePolicy implements QueryAdaptiveTraversalScorePolicy {
  readonly queryAdaptive = true as const;
  readonly descriptor: ScorePolicyDescriptor;
  private readonly base: CalibratedTraversalScorePolicy;

  constructor(profile: TraversalScoringProfile = DEFAULT_CALIBRATED_SCORING_PROFILE) {
    this.base = new CalibratedTraversalScorePolicy(profile);
    const profileDigest = `sha256:${createHash("sha256")
      .update(ADAPTIVE_ROUTE_ALGORITHM_VERSION)
      .update("\0")
      .update(this.base.descriptor.profileDigest)
      .digest("hex")}`;
    this.descriptor = {
      profileId: "adaptive-route-v3",
      profileVersion: 2,
      profileDigest,
      featureSchemaVersion: "n-layer-adaptive-routing-v1",
    };
  }

  bind(question: string): ExplainableTraversalScorePolicy {
    return new BoundAdaptiveRoutePolicy(this.base, this.descriptor, classifyIntent(question));
  }
}

class BoundAdaptiveRoutePolicy implements ExplainableTraversalScorePolicy {
  readonly explainable = true as const;

  constructor(
    private readonly base: CalibratedTraversalScorePolicy,
    readonly descriptor: ScorePolicyDescriptor,
    private readonly intent: AdaptiveQueryIntent,
  ) {}

  seedScore(input: SeedScoreInput): ScoreComputation {
    const base = this.base.seedScore(input);
    const normalizedMatch = bestNormalizedMatch(input.observations?.query);
    const semanticGate =
      input.nodeKind !== "chunk" && normalizedMatch !== undefined ? normalizedMatch : 1;
    return adaptComputation(
      base,
      semanticGate,
      { adaptiveSeedGate: semanticGate },
      {
        nodeKind: input.nodeKind ?? "unknown",
        providers: (input.observations?.providers ?? []).join(","),
        providerCount: input.observations?.providers?.length ?? 0,
        queryIntent: this.intent,
      },
    );
  }

  edgeScore(parentScore: number, input: EdgeScoreInput, hop: number): ScoreComputation {
    const base = this.base.edgeScore(parentScore, input, hop);
    const fanout = input.observations?.fanout;
    if (!fanout) {
      return adaptComputation(
        base,
        1,
        { adaptiveRouteGate: 1 },
        { route: routeName(input), queryIntent: this.intent },
      );
    }

    const selectivity = fanoutSelectivity(fanout.returnedCount, fanout.candidateCount);
    const queryRank = bestReciprocalRank(input.observations?.query) ?? 1;
    const queryEvidence = bestNormalizedMatch(input.observations?.query) ?? 1;
    const summaryCoverage =
      this.intent === "summary" && input.fromKind === "resource" && input.toKind === "chunk";
    // Rank only says which candidate won within this route. An absolute,
    // adapter-normalized query signal prevents a one-item or uniformly weak
    // route from receiving full authority merely because its best item is #1.
    // Missing native scores remain neutral so adapters can adopt the feature
    // incrementally without inventing pseudo-confidence.
    const routeGate = summaryCoverage ? queryRank : selectivity * queryRank * queryEvidence;
    return adaptComputation(
      base,
      routeGate,
      {
        adaptiveRouteGate: routeGate,
        fanoutSelectivity: selectivity,
        routeQueryRank: queryRank,
        routeQueryEvidence: queryEvidence,
        summaryCoverage: summaryCoverage ? 1 : 0,
      },
      {
        route: routeName(input),
        queryIntent: this.intent,
        fanoutReturnedCount: positiveInteger(fanout.returnedCount),
        fanoutCandidateCount: positiveInteger(fanout.candidateCount),
      },
    );
  }

  evidenceScore(pathScore: number, input: EvidenceScoreInput): ScoreComputation {
    return this.base.evidenceScore(pathScore, input);
  }
}

function adaptComputation(
  base: ScoreComputation,
  gate: number,
  factors: Readonly<Record<string, number>>,
  observations: Readonly<Record<string, number | boolean | string>>,
): ScoreComputation {
  return {
    ...base,
    score: clamp(base.score * gate),
    factors: { ...base.factors, ...factors },
    observations: { ...base.observations, ...observations },
  };
}

function fanoutSelectivity(returnedCount: number, candidateCount: number): number {
  const returned = Math.min(positiveInteger(returnedCount), positiveInteger(candidateCount));
  const candidates = Math.max(returned, positiveInteger(candidateCount));
  return Math.log2(1 + returned) / Math.log2(1 + candidates);
}

function bestReciprocalRank(query: QueryMatchObservation | undefined): number | undefined {
  if (query?.exactMatch || query?.aliasMatch) return 1;
  const ranks = [query?.lexical, query?.vector]
    .filter((value): value is RankedRetrievalObservation => value !== undefined)
    .map((value) => 1 / Math.log2(positiveInteger(value.rank) + 1));
  return ranks.length === 0 ? undefined : Math.max(...ranks);
}

function bestNormalizedMatch(query: QueryMatchObservation | undefined): number | undefined {
  if (query?.exactMatch || query?.aliasMatch) return 1;
  const values = [
    query?.lexical?.normalizedScore,
    query?.vector?.normalizedScore,
    query?.rerankerScore,
  ].filter((value): value is number => value !== undefined && Number.isFinite(value));
  return values.length === 0 ? undefined : clamp(Math.max(...values));
}

function classifyIntent(question: string): AdaptiveQueryIntent {
  const normalized = question.toLocaleLowerCase();
  if (
    /\b(summary|summarize|overview|main themes?|comprehensive account|outline)\b/.test(
      normalized,
    ) ||
    /(요약|개요|전체 내용|핵심 내용)/.test(normalized)
  ) {
    return "summary";
  }
  if (
    /\b(compare|comparison|contrast|difference|versus|vs\.?|relationship between)\b/.test(
      normalized,
    ) ||
    /(비교|차이|관계)/.test(normalized)
  ) {
    return "comparison";
  }
  return "lookup";
}

function routeName(input: EdgeScoreInput): string {
  return `${input.fromKind}:${input.operation}:${input.toKind}`;
}

function positiveInteger(value: number): number {
  return Number.isFinite(value) ? Math.max(1, Math.floor(value)) : 1;
}

function clamp(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}
