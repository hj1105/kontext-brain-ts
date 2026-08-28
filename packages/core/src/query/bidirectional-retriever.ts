import type { Principal } from "../knowledge/domain.js";
import type {
  EvidenceHitObservations,
  ExplainableTraversalScorePolicy,
  QueryAdaptiveTraversalScorePolicy,
  ScoreComputation,
  ScorePolicyDescriptor,
  SearchEdgeObservations,
  SearchSeedObservations,
  TraversalScoreBreakdown,
} from "./traversal-scoring.js";

export type SearchNodeKind = "ontology" | "resource" | "chunk" | "entity" | "fact";
export type SearchOperation = "lift" | "expand" | "ground";

export interface SearchNode {
  readonly kind: SearchNodeKind;
  readonly id: string;
}

export interface SearchSeed {
  readonly node: SearchNode;
  /** @deprecated Prefer raw `observations`; retained for legacy-v1 compatibility. */
  readonly score?: number;
  readonly observations?: SearchSeedObservations;
}

export interface SearchEdge {
  readonly from: SearchNode;
  readonly to: SearchNode;
  readonly operation: SearchOperation;
  /** @deprecated Prefer `observations.structural`. */
  readonly confidence?: number;
  /** @deprecated Prefer `observations.query`. */
  readonly queryRelevance?: number;
  /** @deprecated Prefer `observations.support`. */
  readonly evidenceSupport?: number;
  readonly observations?: SearchEdgeObservations;
}

export interface EvidenceHit {
  readonly evidenceId: string;
  readonly chunkId: string;
  readonly resourceId: string;
  readonly text: string;
  /** @deprecated Prefer raw `observations`; retained for legacy-v1 compatibility. */
  readonly score?: number;
  readonly factKey?: string;
  readonly factStatus?: "active" | "conflict";
  readonly observations?: EvidenceHitObservations;
}

export interface RankedEvidenceHit extends EvidenceHit {
  readonly score: number;
  readonly path: readonly SearchEdge[];
  readonly scoreBreakdown?: TraversalScoreBreakdown;
}

/**
 * A single traversal's shared resources (e.g. one pooled DB connection and
 * read-only transaction). Opened once per `retrieve()` and closed at the end,
 * so an implementation does not have to open a new transaction per visited node.
 */
export interface SearchGraphSession {
  close(): Promise<void>;
}

/**
 * ACL filtering is part of this port's contract: implementations must never
 * return a seed, edge, or Evidence item that `principal` cannot access.
 *
 * `openSession` is optional: when provided, the retriever opens one session for
 * the whole traversal and passes it back into every `seed`/`neighbors`/`evidence`
 * call so they can reuse a single connection/transaction. Implementations that
 * ignore the `session` argument keep working unchanged.
 */
export interface SearchGraphPort {
  openSession?(principal: Principal): Promise<SearchGraphSession>;
  seed(
    question: string,
    principal: Principal,
    session?: SearchGraphSession,
  ): Promise<readonly SearchSeed[]>;
  neighbors(
    node: SearchNode,
    question: string,
    principal: Principal,
    session?: SearchGraphSession,
  ): Promise<readonly SearchEdge[]>;
  evidence(
    node: SearchNode,
    principal: Principal,
    session?: SearchGraphSession,
  ): Promise<readonly EvidenceHit[]>;
}

export interface SearchBudget {
  readonly maxHops: number;
  readonly maxKgHops: number;
  readonly maxVisited: number;
  readonly maxCandidates: number;
  readonly timeBudgetMs: number;
  readonly minScore: number;
  readonly topK: number;
}

export type SearchStopReason =
  | "frontier_exhausted"
  | "time_budget"
  | "visited_budget"
  | "candidate_budget";

export interface SearchTrace {
  readonly visited: number;
  readonly candidates: number;
  readonly elapsedMs: number;
  readonly stoppedBy: SearchStopReason;
  readonly averageSelectedPathLength: number;
  readonly maxSelectedPathLength: number;
  readonly seedProviderCounts: Readonly<Record<string, number>>;
  readonly scoring: ScorePolicyDescriptor & {
    readonly missingSignals: readonly string[];
    readonly missingSignalCounts: Readonly<Record<string, number>>;
    readonly routing?: AdaptiveRoutingTrace;
    readonly shadow?: SearchShadowTrace;
  };
}

export interface AdaptiveRoutingTrace {
  readonly selectedRouteCounts: Readonly<Record<string, number>>;
  readonly averageRouteGate: number;
  readonly queryIntent?: string;
}

export interface SearchShadowTrace extends ScorePolicyDescriptor {
  readonly status: "completed" | "failed";
  readonly topKOverlapRatio?: number;
  readonly normalizedRankDisagreement?: number;
  readonly visited?: number;
  readonly candidates?: number;
  readonly elapsedMs?: number;
  readonly errorName?: string;
}

export interface BidirectionalRetrievalInput {
  readonly question: string;
  readonly principal: Principal;
  readonly budget?: Partial<SearchBudget>;
}

export interface BidirectionalRetrievalResult {
  readonly evidence: readonly RankedEvidenceHit[];
  readonly trace: SearchTrace;
}

export interface TraversalScorePolicy {
  seedScore(seed: SearchSeed): number;
  edgeScore(parentScore: number, edge: SearchEdge, hop: number): number;
  evidenceScore(pathScore: number, evidence: EvidenceHit): number;
}

type BoundTraversalScorePolicy = TraversalScorePolicy | ExplainableTraversalScorePolicy;

export type AnyTraversalScorePolicy = BoundTraversalScorePolicy | QueryAdaptiveTraversalScorePolicy;

export interface TraversalScorePolicyResolver {
  resolve(principal: Principal): Promise<AnyTraversalScorePolicy>;
  resolveShadow?(principal: Principal): Promise<AnyTraversalScorePolicy | null>;
}

/** @deprecated Use a versioned observation-based `CalibratedTraversalScorePolicy`. */
export class BalancedTraversalScorePolicy implements TraversalScorePolicy {
  constructor(private readonly hopPenalty = 0.92) {}

  seedScore(seed: SearchSeed): number {
    return clampScore(seed.score ?? 0);
  }

  edgeScore(parentScore: number, edge: SearchEdge, hop: number): number {
    const signals =
      0.5 +
      0.3 * clampScore(edge.queryRelevance ?? 0) +
      0.2 * clampScore(edge.evidenceSupport ?? 0);
    return (
      parentScore * clampScore(edge.confidence ?? 0) * signals * this.hopPenalty ** Math.max(1, hop)
    );
  }

  evidenceScore(pathScore: number, evidence: EvidenceHit): number {
    return pathScore * clampScore(evidence.score ?? 0);
  }
}

interface FrontierCandidate {
  readonly node: SearchNode;
  readonly score: number;
  readonly hops: number;
  readonly kgHops: number;
  readonly path: readonly SearchEdge[];
  readonly seedComputation: ScoreComputation;
  readonly edgeComputations: readonly ScoreComputation[];
}

const LEGACY_SCORE_DESCRIPTOR: ScorePolicyDescriptor = {
  profileId: "legacy-v1",
  profileVersion: 1,
  profileDigest: "builtin:legacy-v1",
  featureSchemaVersion: "legacy-score-fields-v1",
};

const DEFAULT_SEARCH_BUDGET: SearchBudget = {
  maxHops: 8,
  maxKgHops: 3,
  maxVisited: 200,
  maxCandidates: 500,
  timeBudgetMs: 1200,
  minScore: 0.02,
  topK: 12,
};

export class BidirectionalNLayerRetriever {
  constructor(
    private readonly graph: SearchGraphPort,
    private readonly scorePolicySource:
      | AnyTraversalScorePolicy
      | TraversalScorePolicyResolver = new BalancedTraversalScorePolicy(),
    private readonly now: () => number = () => performance.now(),
  ) {}

  async retrieve(input: BidirectionalRetrievalInput): Promise<BidirectionalRetrievalResult> {
    const budget = { ...DEFAULT_SEARCH_BUDGET, ...input.budget };
    const startedAt = this.now();
    const resolvedScorePolicy = isScorePolicyResolver(this.scorePolicySource)
      ? await this.scorePolicySource.resolve(input.principal)
      : this.scorePolicySource;
    const scorePolicy = bindQueryPolicy(resolvedScorePolicy, input.question);
    let shadowPolicy: BoundTraversalScorePolicy | null = null;
    let shadowResolutionError: unknown;
    if (isScorePolicyResolver(this.scorePolicySource)) {
      try {
        const resolvedShadow =
          (await this.scorePolicySource.resolveShadow?.(input.principal)) ?? null;
        shadowPolicy = resolvedShadow ? bindQueryPolicy(resolvedShadow, input.question) : null;
      } catch (error) {
        shadowResolutionError = error;
      }
    }
    // One session for the whole traversal, so adapters can share a single
    // connection/transaction instead of opening one per visited node.
    const session = await this.graph.openSession?.(input.principal);
    let traversal:
      | { readonly ok: true; readonly result: BidirectionalRetrievalResult }
      | { readonly ok: false; readonly error: unknown };
    try {
      const result = await this.traverse(input, budget, startedAt, session, scorePolicy);
      let shadow: SearchShadowTrace | undefined =
        shadowResolutionError === undefined
          ? undefined
          : {
              profileId: "unresolved-shadow",
              profileVersion: 0,
              profileDigest: "unresolved",
              featureSchemaVersion: "unresolved",
              status: "failed",
              errorName:
                shadowResolutionError instanceof Error
                  ? shadowResolutionError.name
                  : "UnknownError",
            };
      if (shadowPolicy) {
        const shadowStartedAt = this.now();
        try {
          const shadowResult = await this.traverse(
            input,
            budget,
            shadowStartedAt,
            session,
            shadowPolicy,
          );
          shadow = compareShadowResults(result, shadowResult);
        } catch (error) {
          shadow = {
            ...scoreDescriptor(shadowPolicy),
            status: "failed",
            errorName: error instanceof Error ? error.name : "UnknownError",
          };
        }
      }
      traversal = {
        ok: true,
        result:
          shadow === undefined
            ? result
            : {
                ...result,
                trace: {
                  ...result.trace,
                  scoring: { ...result.trace.scoring, shadow },
                },
              },
      };
    } catch (error) {
      traversal = { ok: false, error };
    }

    let cleanupError: unknown;
    let cleanupFailed = false;
    try {
      await session?.close();
    } catch (error) {
      cleanupFailed = true;
      cleanupError = error;
    }

    if (!traversal.ok) {
      if (cleanupFailed) {
        throw new AggregateError(
          [traversal.error, cleanupError],
          "Search traversal and session cleanup both failed",
        );
      }
      throw traversal.error;
    }
    if (cleanupFailed) throw cleanupError;
    return traversal.result;
  }

  private async traverse(
    input: BidirectionalRetrievalInput,
    budget: SearchBudget,
    startedAt: number,
    session: SearchGraphSession | undefined,
    scorePolicy: BoundTraversalScorePolicy,
  ): Promise<BidirectionalRetrievalResult> {
    const frontier = new MaxPriorityQueue<FrontierCandidate>((candidate) => candidate.score);
    const bestScores = new Map<string, number>();
    const evidenceById = new Map<string, RankedEvidenceHit>();
    const evidenceScoresByNode = new Map<string, number>();
    const missingSignalCounts = new Map<string, number>();
    let candidateCount = 0;
    let visited = 0;
    let stoppedBy: SearchStopReason = "frontier_exhausted";
    let candidateBudgetReached = false;

    const collectEvidence = async (
      node: SearchNode,
      pathScore: number,
      path: readonly SearchEdge[],
      seedComputation: ScoreComputation,
      edgeComputations: readonly ScoreComputation[],
    ): Promise<void> => {
      const nodeKey = searchNodeKey(node);
      if (pathScore <= (evidenceScoresByNode.get(nodeKey) ?? Number.NEGATIVE_INFINITY)) return;
      evidenceScoresByNode.set(nodeKey, pathScore);
      const hits = await this.graph.evidence(node, input.principal, session);
      for (const hit of hits) {
        const evidenceComputation = this.computeEvidenceScore(scorePolicy, pathScore, hit);
        recordMissingSignals(missingSignalCounts, evidenceComputation.missingSignals);
        const ranked: RankedEvidenceHit = {
          ...hit,
          score: evidenceComputation.score,
          path,
          scoreBreakdown: {
            ...this.scoreDescriptor(scorePolicy),
            seed: seedComputation,
            edges: edgeComputations,
            evidence: evidenceComputation,
            finalScore: evidenceComputation.score,
          },
        };
        const previous = evidenceById.get(hit.evidenceId);
        if (!previous || ranked.score > previous.score) evidenceById.set(hit.evidenceId, ranked);
      }
    };

    // Pool checkout and session setup are part of the caller's latency budget.
    // If they consume it, avoid starting more database work just to discover that
    // the traversal has already timed out.
    if (this.now() - startedAt >= budget.timeBudgetMs) {
      return {
        evidence: [],
        trace: {
          visited,
          candidates: candidateCount,
          elapsedMs: this.now() - startedAt,
          stoppedBy: "time_budget",
          averageSelectedPathLength: 0,
          maxSelectedPathLength: 0,
          seedProviderCounts: {},
          scoring: {
            ...this.scoreDescriptor(scorePolicy),
            missingSignals: [],
            missingSignalCounts: {},
          },
        },
      };
    }

    const seeds = await this.graph.seed(input.question, input.principal, session);
    const seedProviderCounts = countSeedProviders(seeds);
    for (const seed of seeds) {
      if (this.now() - startedAt >= budget.timeBudgetMs) {
        stoppedBy = "time_budget";
        break;
      }
      const seedComputation = this.computeSeedScore(scorePolicy, seed);
      recordMissingSignals(missingSignalCounts, seedComputation.missingSignals);
      const score = seedComputation.score;
      if (score < budget.minScore || candidateCount >= budget.maxCandidates) continue;
      const nodeKey = searchNodeKey(seed.node);
      if (score <= (bestScores.get(nodeKey) ?? Number.NEGATIVE_INFINITY)) continue;
      bestScores.set(nodeKey, score);
      frontier.push({
        node: seed.node,
        score,
        hops: 0,
        kgHops: 0,
        path: [],
        seedComputation,
        edgeComputations: [],
      });
      candidateCount++;
      // A direct seed is already a successful retrieval candidate. Collect its
      // evidence before graph expansion so a high-fanout seed cannot exhaust
      // the candidate budget and starve lower-ranked direct hits.
      await collectEvidence(seed.node, score, [], seedComputation, []);
      if (this.now() - startedAt >= budget.timeBudgetMs) {
        stoppedBy = "time_budget";
        break;
      }
    }

    while (frontier.size > 0) {
      if (this.now() - startedAt >= budget.timeBudgetMs) {
        stoppedBy = "time_budget";
        break;
      }
      if (visited >= budget.maxVisited) {
        stoppedBy = "visited_budget";
        break;
      }

      const current = frontier.pop();
      if (!current) break;
      const currentKey = searchNodeKey(current.node);
      if (current.score < (bestScores.get(currentKey) ?? Number.NEGATIVE_INFINITY)) continue;
      visited++;

      await collectEvidence(
        current.node,
        current.score,
        current.path,
        current.seedComputation,
        current.edgeComputations,
      );

      if (current.hops >= budget.maxHops || candidateBudgetReached) continue;
      const neighbors = await this.graph.neighbors(
        current.node,
        input.question,
        input.principal,
        session,
      );
      for (const edge of neighbors) {
        if (candidateCount >= budget.maxCandidates) {
          stoppedBy = "candidate_budget";
          candidateBudgetReached = true;
          break;
        }
        const kgHops = current.kgHops + (isKgExpansion(edge) ? 1 : 0);
        if (kgHops > budget.maxKgHops) continue;
        const hops = current.hops + 1;
        const edgeComputation = this.computeEdgeScore(scorePolicy, current.score, edge, hops);
        recordMissingSignals(missingSignalCounts, edgeComputation.missingSignals);
        const score = edgeComputation.score;
        if (score < budget.minScore) continue;
        const nextKey = searchNodeKey(edge.to);
        if (score <= (bestScores.get(nextKey) ?? Number.NEGATIVE_INFINITY)) continue;
        bestScores.set(nextKey, score);
        frontier.push({
          node: edge.to,
          score,
          hops,
          kgHops,
          path: [...current.path, edge],
          seedComputation: current.seedComputation,
          edgeComputations: [...current.edgeComputations, edgeComputation],
        });
        candidateCount++;
      }
    }

    const evidence = Array.from(evidenceById.values())
      .sort((left, right) => right.score - left.score)
      .slice(0, budget.topK);
    const pathLengths = evidence.map((hit) => hit.path.length);
    const routing = summarizeAdaptiveRouting(evidence);
    return {
      evidence,
      trace: {
        visited,
        candidates: candidateCount,
        elapsedMs: this.now() - startedAt,
        stoppedBy,
        averageSelectedPathLength:
          pathLengths.length === 0
            ? 0
            : pathLengths.reduce((sum, length) => sum + length, 0) / pathLengths.length,
        maxSelectedPathLength: pathLengths.length === 0 ? 0 : Math.max(...pathLengths),
        seedProviderCounts,
        scoring: {
          ...this.scoreDescriptor(scorePolicy),
          missingSignals: Array.from(missingSignalCounts.keys()).sort(),
          missingSignalCounts: Object.fromEntries(
            Array.from(missingSignalCounts.entries()).sort(([left], [right]) =>
              left.localeCompare(right),
            ),
          ),
          ...(routing === undefined ? {} : { routing }),
        },
      },
    };
  }

  private computeSeedScore(
    scorePolicy: BoundTraversalScorePolicy,
    seed: SearchSeed,
  ): ScoreComputation {
    if (isExplainablePolicy(scorePolicy)) {
      return scorePolicy.seedScore({
        legacyScore: seed.score,
        observations: seed.observations,
        nodeKind: seed.node.kind,
      });
    }
    const score = clampScore(scorePolicy.seedScore(seed));
    return legacyComputation("seed", score, { legacySeedScore: score });
  }

  private computeEdgeScore(
    scorePolicy: BoundTraversalScorePolicy,
    parentScore: number,
    edge: SearchEdge,
    hop: number,
  ): ScoreComputation {
    if (isExplainablePolicy(scorePolicy)) {
      return scorePolicy.edgeScore(
        parentScore,
        {
          operation: edge.operation,
          fromKind: edge.from.kind,
          toKind: edge.to.kind,
          legacyConfidence: edge.confidence,
          legacyQueryRelevance: edge.queryRelevance,
          legacyEvidenceSupport: edge.evidenceSupport,
          observations: edge.observations,
        },
        hop,
      );
    }
    const score = clampScore(scorePolicy.edgeScore(parentScore, edge, hop));
    return legacyComputation("edge", score, { legacyEdgeScore: score }, parentScore);
  }

  private computeEvidenceScore(
    scorePolicy: BoundTraversalScorePolicy,
    pathScore: number,
    evidence: EvidenceHit,
  ): ScoreComputation {
    if (isExplainablePolicy(scorePolicy)) {
      return scorePolicy.evidenceScore(pathScore, {
        legacyScore: evidence.score,
        factStatus: evidence.factStatus,
        observations: evidence.observations,
      });
    }
    const score = clampScore(scorePolicy.evidenceScore(pathScore, evidence));
    return legacyComputation("evidence", score, { legacyEvidenceScore: score }, pathScore);
  }

  private scoreDescriptor(scorePolicy: BoundTraversalScorePolicy): ScorePolicyDescriptor {
    return scoreDescriptor(scorePolicy);
  }
}

function scoreDescriptor(scorePolicy: BoundTraversalScorePolicy): ScorePolicyDescriptor {
  return isExplainablePolicy(scorePolicy) ? scorePolicy.descriptor : LEGACY_SCORE_DESCRIPTOR;
}

function compareShadowResults(
  active: BidirectionalRetrievalResult,
  shadow: BidirectionalRetrievalResult,
): SearchShadowTrace {
  const activeIds = active.evidence.map((hit) => hit.evidenceId);
  const shadowIds = shadow.evidence.map((hit) => hit.evidenceId);
  const denominator = Math.max(1, activeIds.length, shadowIds.length);
  const shadowSet = new Set(shadowIds);
  const overlap = activeIds.filter((id) => shadowSet.has(id)).length / denominator;
  const allIds = Array.from(new Set([...activeIds, ...shadowIds]));
  const activeRanks = new Map(activeIds.map((id, index) => [id, index]));
  const shadowRanks = new Map(shadowIds.map((id, index) => [id, index]));
  const disagreement =
    allIds.length === 0
      ? 0
      : allIds.reduce(
          (sum, id) =>
            sum +
            Math.abs((activeRanks.get(id) ?? denominator) - (shadowRanks.get(id) ?? denominator)) /
              denominator,
          0,
        ) / allIds.length;
  return {
    profileId: shadow.trace.scoring.profileId,
    profileVersion: shadow.trace.scoring.profileVersion,
    profileDigest: shadow.trace.scoring.profileDigest,
    featureSchemaVersion: shadow.trace.scoring.featureSchemaVersion,
    status: "completed",
    topKOverlapRatio: overlap,
    normalizedRankDisagreement: disagreement,
    visited: shadow.trace.visited,
    candidates: shadow.trace.candidates,
    elapsedMs: shadow.trace.elapsedMs,
  };
}

function recordMissingSignals(counts: Map<string, number>, signals: readonly string[]): void {
  for (const signal of signals) counts.set(signal, (counts.get(signal) ?? 0) + 1);
}

function countSeedProviders(seeds: readonly SearchSeed[]): Readonly<Record<string, number>> {
  const counts = new Map<string, number>();
  for (const seed of seeds) {
    for (const provider of seed.observations?.providers ?? ["unspecified"]) {
      counts.set(provider, (counts.get(provider) ?? 0) + 1);
    }
  }
  return Object.fromEntries(
    Array.from(counts.entries()).sort(([left], [right]) => left.localeCompare(right)),
  );
}

function bindQueryPolicy(
  policy: AnyTraversalScorePolicy,
  question: string,
): BoundTraversalScorePolicy {
  return isQueryAdaptivePolicy(policy) ? policy.bind(question) : policy;
}

function summarizeAdaptiveRouting(
  evidence: readonly RankedEvidenceHit[],
): AdaptiveRoutingTrace | undefined {
  const routeCounts = new Map<string, number>();
  const gates: number[] = [];
  let queryIntent: string | undefined;
  for (const hit of evidence) {
    const route = evidenceRoute(hit);
    routeCounts.set(route, (routeCounts.get(route) ?? 0) + 1);
    const breakdown = hit.scoreBreakdown;
    if (!breakdown) continue;
    if (queryIntent === undefined) {
      const observed = breakdown.seed.observations.queryIntent;
      if (typeof observed === "string") queryIntent = observed;
    }
    for (const edge of breakdown.edges) {
      const gate = edge.factors.adaptiveRouteGate;
      if (gate !== undefined) gates.push(gate);
    }
  }
  if (gates.length === 0 && queryIntent === undefined) return undefined;
  return {
    selectedRouteCounts: Object.fromEntries(
      Array.from(routeCounts.entries()).sort(([left], [right]) => left.localeCompare(right)),
    ),
    averageRouteGate:
      gates.length === 0 ? 1 : gates.reduce((sum, value) => sum + value, 0) / gates.length,
    ...(queryIntent === undefined ? {} : { queryIntent }),
  };
}

function evidenceRoute(hit: RankedEvidenceHit): string {
  if (hit.path.length === 0) return "direct";
  const kinds = new Set(hit.path.flatMap((edge) => [edge.from.kind, edge.to.kind]));
  if (kinds.has("fact")) return "fact";
  if (kinds.has("entity")) return "entity";
  if (kinds.has("ontology")) return "ontology";
  if (kinds.has("resource")) return "resource";
  return "other";
}

function isExplainablePolicy(
  policy: BoundTraversalScorePolicy,
): policy is ExplainableTraversalScorePolicy {
  return "explainable" in policy && policy.explainable === true;
}

function isQueryAdaptivePolicy(
  policy: AnyTraversalScorePolicy,
): policy is QueryAdaptiveTraversalScorePolicy {
  return "queryAdaptive" in policy && policy.queryAdaptive === true;
}

function isScorePolicyResolver(
  source: AnyTraversalScorePolicy | TraversalScorePolicyResolver,
): source is TraversalScorePolicyResolver {
  return "resolve" in source;
}

function legacyComputation(
  stage: ScoreComputation["stage"],
  score: number,
  factors: Record<string, number>,
  inputScore?: number,
): ScoreComputation {
  return {
    stage,
    ...(inputScore === undefined ? {} : { inputScore }),
    score,
    factors,
    observations: {},
    missingSignals: [],
  };
}

function isKgExpansion(edge: SearchEdge): boolean {
  return (
    edge.operation === "expand" &&
    ([edge.from.kind, edge.to.kind].includes("entity") ||
      [edge.from.kind, edge.to.kind].includes("fact"))
  );
}

function searchNodeKey(node: SearchNode): string {
  return `${node.kind}:${node.id}`;
}

function clampScore(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

class MaxPriorityQueue<T> {
  private readonly values: T[] = [];

  constructor(private readonly score: (value: T) => number) {}

  get size(): number {
    return this.values.length;
  }

  push(value: T): void {
    this.values.push(value);
    this.bubbleUp(this.values.length - 1);
  }

  pop(): T | undefined {
    const first = this.values[0];
    const last = this.values.pop();
    if (this.values.length > 0 && last !== undefined) {
      this.values[0] = last;
      this.bubbleDown(0);
    }
    return first;
  }

  private bubbleUp(index: number): void {
    let child = index;
    while (child > 0) {
      const parent = Math.floor((child - 1) / 2);
      const childValue = this.values[child];
      const parentValue = this.values[parent];
      if (childValue === undefined || parentValue === undefined) return;
      if (this.score(childValue) <= this.score(parentValue)) return;
      this.values[child] = parentValue;
      this.values[parent] = childValue;
      child = parent;
    }
  }

  private bubbleDown(index: number): void {
    let parent = index;
    while (true) {
      const left = parent * 2 + 1;
      const right = left + 1;
      let largest = parent;
      const parentValue = this.values[largest];
      const leftValue = this.values[left];
      const rightValue = this.values[right];
      if (
        leftValue !== undefined &&
        parentValue !== undefined &&
        this.score(leftValue) > this.score(parentValue)
      ) {
        largest = left;
      }
      const largestValue = this.values[largest];
      if (
        rightValue !== undefined &&
        largestValue !== undefined &&
        this.score(rightValue) > this.score(largestValue)
      ) {
        largest = right;
      }
      if (largest === parent) return;
      const swap = this.values[parent];
      const next = this.values[largest];
      if (swap === undefined || next === undefined) return;
      this.values[parent] = next;
      this.values[largest] = swap;
      parent = largest;
    }
  }
}
