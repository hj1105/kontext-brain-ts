import { describe, expect, it } from "vitest";
import {
  BidirectionalNLayerRetriever,
  type EvidenceHit,
  type Principal,
  type SearchEdge,
  type SearchGraphPort,
  type SearchNode,
  type SearchSeed,
} from "../src/index.js";

const principal: Principal = {
  organizationId: "acme",
  subjectId: "user:1",
  groupIds: ["engineering"],
};

function key(node: SearchNode): string {
  return `${node.kind}:${node.id}`;
}

class FakeSearchGraph implements SearchGraphPort {
  readonly expanded: string[] = [];
  readonly seeds: SearchSeed[] = [];
  readonly edges = new Map<string, SearchEdge[]>();
  readonly hits = new Map<string, EvidenceHit[]>();

  async seed(): Promise<readonly SearchSeed[]> {
    return this.seeds;
  }

  async neighbors(node: SearchNode): Promise<readonly SearchEdge[]> {
    this.expanded.push(key(node));
    return this.edges.get(key(node)) ?? [];
  }

  async evidence(node: SearchNode): Promise<readonly EvidenceHit[]> {
    return this.hits.get(key(node)) ?? [];
  }
}

function edge(
  from: SearchNode,
  to: SearchNode,
  operation: SearchEdge["operation"],
  confidence = 1,
): SearchEdge {
  return { from, to, operation, confidence, queryRelevance: 1, evidenceSupport: 1 };
}

describe("BidirectionalNLayerRetriever", () => {
  it("starts at a chunk, lifts as often as useful, expands, and grounds back to evidence", async () => {
    const graph = new FakeSearchGraph();
    const seedChunk: SearchNode = { kind: "chunk", id: "slack:m1" };
    const entity: SearchNode = { kind: "entity", id: "checkout" };
    const payment: SearchNode = { kind: "ontology", id: "payment" };
    const order: SearchNode = { kind: "ontology", id: "order" };
    const resource: SearchNode = { kind: "resource", id: "notion:orders" };
    const answerChunk: SearchNode = { kind: "chunk", id: "notion:block-8" };
    graph.seeds.push({ node: seedChunk, score: 0.92 });
    graph.edges.set(key(seedChunk), [edge(seedChunk, entity, "lift")]);
    graph.edges.set(key(entity), [edge(entity, payment, "lift")]);
    graph.edges.set(key(payment), [edge(payment, order, "expand")]);
    graph.edges.set(key(order), [edge(order, resource, "ground")]);
    graph.edges.set(key(resource), [edge(resource, answerChunk, "ground")]);
    graph.hits.set(key(answerChunk), [
      {
        evidenceId: "evidence:8",
        chunkId: answerChunk.id,
        resourceId: resource.id,
        text: "Order 42 was paid",
        score: 1,
      },
    ]);

    const result = await new BidirectionalNLayerRetriever(graph).retrieve({
      question: "Was order 42 paid?",
      principal,
      budget: { maxHops: 6, maxVisited: 20, maxCandidates: 20, timeBudgetMs: 1000 },
    });

    expect(result.evidence.map((item) => item.evidenceId)).toEqual(["evidence:8"]);
    expect(result.evidence[0]?.path.map((step) => step.operation)).toEqual([
      "lift",
      "lift",
      "expand",
      "ground",
      "ground",
    ]);
    expect(result.trace.stoppedBy).toBe("frontier_exhausted");
  });

  it("revisits a node only when a better-scoring path reaches it", async () => {
    const graph = new FakeSearchGraph();
    const start: SearchNode = { kind: "entity", id: "start" };
    const weak: SearchNode = { kind: "fact", id: "weak" };
    const strong: SearchNode = { kind: "fact", id: "strong" };
    const target: SearchNode = { kind: "entity", id: "target" };
    graph.seeds.push({ node: start, score: 1 });
    graph.edges.set(key(start), [
      edge(start, weak, "expand", 0.95),
      edge(start, strong, "expand", 0.9),
    ]);
    graph.edges.set(key(weak), [edge(weak, target, "expand", 0.2)]);
    graph.edges.set(key(strong), [edge(strong, target, "expand", 1)]);

    await new BidirectionalNLayerRetriever(graph).retrieve({
      question: "target",
      principal,
      budget: { maxHops: 4, maxVisited: 10, maxCandidates: 10, timeBudgetMs: 1000 },
    });

    expect(graph.expanded.filter((item) => item === key(target))).toHaveLength(1);
  });

  it("passes the principal into every graph operation so ACL filtering happens before traversal", async () => {
    const seenPrincipals: Principal[] = [];
    const graph: SearchGraphPort = {
      async seed(_question, received) {
        seenPrincipals.push(received);
        return [{ node: { kind: "ontology", id: "order" }, score: 1 }];
      },
      async neighbors(_node, _question, received) {
        seenPrincipals.push(received);
        return [];
      },
      async evidence(_node, received) {
        seenPrincipals.push(received);
        return [];
      },
    };

    await new BidirectionalNLayerRetriever(graph).retrieve({ question: "orders", principal });

    expect(seenPrincipals).toHaveLength(3);
    expect(seenPrincipals.every((received) => received === principal)).toBe(true);
  });
});
