import type { Principal } from "../knowledge/domain.js";

export type SearchNodeKind = "ontology" | "resource" | "chunk" | "entity" | "fact";
export type SearchOperation = "lift" | "expand" | "ground";

export interface SearchNode {
  readonly kind: SearchNodeKind;
  readonly id: string;
}

export interface SearchSeed {
  readonly node: SearchNode;
  readonly score: number;
}

export interface SearchEdge {
  readonly from: SearchNode;
  readonly to: SearchNode;
  readonly operation: SearchOperation;
  readonly confidence: number;
  readonly queryRelevance: number;
  readonly evidenceSupport: number;
}

export interface EvidenceHit {
  readonly evidenceId: string;
  readonly chunkId: string;
  readonly resourceId: string;
  readonly text: string;
  readonly score: number;
  readonly factKey?: string;
  readonly factStatus?: "active" | "conflict";
}

export interface RankedEvidenceHit extends EvidenceHit {
  readonly score: number;
  readonly path: readonly SearchEdge[];
}

/**
 * ACL filtering is part of this port's contract: implementations must never
 * return a seed, edge, or Evidence item that `principal` cannot access.
 */
export interface SearchGraphPort {
  seed(question: string, principal: Principal): Promise<readonly SearchSeed[]>;
  neighbors(
    node: SearchNode,
    question: string,
    principal: Principal,
  ): Promise<readonly SearchEdge[]>;
  evidence(node: SearchNode, principal: Principal): Promise<readonly EvidenceHit[]>;
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

export class BalancedTraversalScorePolicy implements TraversalScorePolicy {
  constructor(private readonly hopPenalty = 0.92) {}

  seedScore(seed: SearchSeed): number {
    return clampScore(seed.score);
  }

  edgeScore(parentScore: number, edge: SearchEdge, hop: number): number {
    const signals =
      0.5 + 0.3 * clampScore(edge.queryRelevance) + 0.2 * clampScore(edge.evidenceSupport);
    return (
      parentScore * clampScore(edge.confidence) * signals * this.hopPenalty ** Math.max(1, hop)
    );
  }

  evidenceScore(pathScore: number, evidence: EvidenceHit): number {
    return pathScore * clampScore(evidence.score);
  }
}

interface FrontierCandidate {
  readonly node: SearchNode;
  readonly score: number;
  readonly hops: number;
  readonly kgHops: number;
  readonly path: readonly SearchEdge[];
}

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
    private readonly scorePolicy: TraversalScorePolicy = new BalancedTraversalScorePolicy(),
    private readonly now: () => number = () => performance.now(),
  ) {}

  async retrieve(input: BidirectionalRetrievalInput): Promise<BidirectionalRetrievalResult> {
    const budget = { ...DEFAULT_SEARCH_BUDGET, ...input.budget };
    const startedAt = this.now();
    const frontier = new MaxPriorityQueue<FrontierCandidate>((candidate) => candidate.score);
    const bestScores = new Map<string, number>();
    const evidenceById = new Map<string, RankedEvidenceHit>();
    const evidenceScoresByNode = new Map<string, number>();
    let candidateCount = 0;
    let visited = 0;
    let stoppedBy: SearchStopReason = "frontier_exhausted";
    let candidateBudgetReached = false;

    const collectEvidence = async (
      node: SearchNode,
      pathScore: number,
      path: readonly SearchEdge[],
    ): Promise<void> => {
      const nodeKey = searchNodeKey(node);
      if (pathScore <= (evidenceScoresByNode.get(nodeKey) ?? Number.NEGATIVE_INFINITY)) return;
      evidenceScoresByNode.set(nodeKey, pathScore);
      const hits = await this.graph.evidence(node, input.principal);
      for (const hit of hits) {
        const ranked: RankedEvidenceHit = {
          ...hit,
          score: this.scorePolicy.evidenceScore(pathScore, hit),
          path,
        };
        const previous = evidenceById.get(hit.evidenceId);
        if (!previous || ranked.score > previous.score) evidenceById.set(hit.evidenceId, ranked);
      }
    };

    for (const seed of await this.graph.seed(input.question, input.principal)) {
      const score = this.scorePolicy.seedScore(seed);
      if (score < budget.minScore || candidateCount >= budget.maxCandidates) continue;
      const nodeKey = searchNodeKey(seed.node);
      if (score <= (bestScores.get(nodeKey) ?? Number.NEGATIVE_INFINITY)) continue;
      bestScores.set(nodeKey, score);
      frontier.push({ node: seed.node, score, hops: 0, kgHops: 0, path: [] });
      candidateCount++;
      // A direct seed is already a successful retrieval candidate. Collect its
      // evidence before graph expansion so a high-fanout seed cannot exhaust
      // the candidate budget and starve lower-ranked direct hits.
      await collectEvidence(seed.node, score, []);
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

      await collectEvidence(current.node, current.score, current.path);

      if (current.hops >= budget.maxHops || candidateBudgetReached) continue;
      const neighbors = await this.graph.neighbors(current.node, input.question, input.principal);
      for (const edge of neighbors) {
        if (candidateCount >= budget.maxCandidates) {
          stoppedBy = "candidate_budget";
          candidateBudgetReached = true;
          break;
        }
        const kgHops = current.kgHops + (isKgExpansion(edge) ? 1 : 0);
        if (kgHops > budget.maxKgHops) continue;
        const hops = current.hops + 1;
        const score = this.scorePolicy.edgeScore(current.score, edge, hops);
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
        });
        candidateCount++;
      }
    }

    return {
      evidence: Array.from(evidenceById.values())
        .sort((left, right) => right.score - left.score)
        .slice(0, budget.topK),
      trace: {
        visited,
        candidates: candidateCount,
        elapsedMs: this.now() - startedAt,
        stoppedBy,
      },
    };
  }
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
