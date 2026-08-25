import { describe, expect, it } from "vitest";
import {
  AdaptiveRouteTraversalScorePolicy,
  BidirectionalNLayerRetriever,
  type EdgeScoreInput,
  type Principal,
  type SearchGraphPort,
} from "../src/index.js";

const principal: Principal = {
  organizationId: "adaptive-test",
  subjectId: "user:1",
  groupIds: [],
};

function resourceGrounding(
  candidateCount: number,
  returnedCount: number,
  rank = 1,
): EdgeScoreInput {
  return {
    operation: "ground",
    fromKind: "resource",
    toKind: "chunk",
    observations: {
      structural: { kind: "deterministic" },
      query: { lexical: { rank, candidateCount, normalizedScore: 0.8 } },
      fanout: { returnedCount, candidateCount },
      supportApplicability: "not-applicable",
    },
  };
}

describe("AdaptiveRouteTraversalScorePolicy", () => {
  it("downweights a broad resource route from observed fanout rather than a dataset name", () => {
    const policy = new AdaptiveRouteTraversalScorePolicy().bind("What treatment is recommended?");
    const narrow = policy.edgeScore(1, resourceGrounding(4, 4), 1);
    const broad = policy.edgeScore(1, resourceGrounding(1_385, 10), 1);

    expect(narrow.factors.adaptiveRouteGate).toBe(0.8);
    expect(broad.factors.adaptiveRouteGate).toBeLessThan(0.4);
    expect(narrow.score).toBeGreaterThan(broad.score);
    expect(broad.observations).toMatchObject({
      route: "resource:ground:chunk",
      fanoutCandidateCount: 1385,
      queryIntent: "lookup",
    });
  });

  it("uses query-local intent to permit broad resource coverage for a summary", () => {
    const lookup = new AdaptiveRouteTraversalScorePolicy().bind("What treatment is recommended?");
    const summary = new AdaptiveRouteTraversalScorePolicy().bind("Summarize the main themes");
    const edge = resourceGrounding(1_385, 10, 1);

    expect(summary.edgeScore(1, edge, 1).score).toBeGreaterThan(lookup.edgeScore(1, edge, 1).score);
    expect(summary.edgeScore(1, edge, 1).observations.queryIntent).toBe("summary");
  });

  it.each([
    ["Compare checkout and payment", "comparison"],
    ["결제와 환불의 차이를 비교해 줘", "comparison"],
    ["전체 내용을 요약해 줘", "summary"],
  ] as const)("classifies %s as %s without corpus metadata", (question, expectedIntent) => {
    const scored = new AdaptiveRouteTraversalScorePolicy()
      .bind(question)
      .edgeScore(1, resourceGrounding(10, 5), 1);

    expect(scored.observations.queryIntent).toBe(expectedIntent);
  });

  it("decays lower neighbor ranks within the same dynamic route", () => {
    const policy = new AdaptiveRouteTraversalScorePolicy().bind("specific fact");

    expect(policy.edgeScore(1, resourceGrounding(100, 10, 1), 1).score).toBeGreaterThan(
      policy.edgeScore(1, resourceGrounding(100, 10, 10), 1).score,
    );
  });

  it("does not mistake the best item in a weak route for strong query evidence", () => {
    const policy = new AdaptiveRouteTraversalScorePolicy().bind("specific fact");
    const strong = resourceGrounding(1, 1, 1);
    const weak: EdgeScoreInput = {
      ...strong,
      observations: {
        ...strong.observations,
        query: { lexical: { rank: 1, candidateCount: 1, normalizedScore: 0.1 } },
      },
    };

    expect(policy.edgeScore(1, strong, 1).score).toBeGreaterThan(
      policy.edgeScore(1, weak, 1).score,
    );
    expect(policy.edgeScore(1, weak, 1).factors.routeQueryEvidence).toBe(0.1);
  });

  it("keeps every public score within the query-local zero-to-one range", () => {
    const policy = new AdaptiveRouteTraversalScorePolicy().bind("specific fact");
    const seed = policy.seedScore({
      nodeKind: "resource",
      observations: { query: { rerankerScore: 5 } },
    });
    const edge = policy.edgeScore(
      5,
      {
        ...resourceGrounding(Number.NaN, -10, Number.POSITIVE_INFINITY),
        observations: {
          ...resourceGrounding(1, 1).observations,
          query: {
            lexical: {
              rank: Number.POSITIVE_INFINITY,
              candidateCount: Number.NaN,
              normalizedScore: 5,
            },
          },
          fanout: { returnedCount: -10, candidateCount: Number.NaN },
        },
      },
      1,
    );
    const evidence = policy.evidenceScore(5, {
      factStatus: "active",
      observations: {
        origin: "curated",
        confidence: 5,
        freshnessDays: -100,
        supportApplicability: "not-applicable",
      },
    });

    for (const computation of [seed, edge, evidence]) {
      expect(computation.score).toBeGreaterThanOrEqual(0);
      expect(computation.score).toBeLessThanOrEqual(1);
    }
  });

  it("binds once per retrieval and exposes the selected adaptive route decision", async () => {
    const graph: SearchGraphPort = {
      async seed() {
        return [
          {
            node: { kind: "resource", id: "guide" },
            observations: {
              providers: ["test-lexical"],
              query: { lexical: { rank: 1, candidateCount: 1 } },
            },
          },
        ];
      },
      async neighbors(node) {
        return node.kind === "resource"
          ? [
              {
                from: node,
                to: { kind: "chunk", id: "answer" },
                operation: "ground",
                observations: {
                  structural: { kind: "deterministic" },
                  query: { lexical: { rank: 1, candidateCount: 50 } },
                  fanout: { returnedCount: 10, candidateCount: 50 },
                  supportApplicability: "not-applicable",
                },
              },
            ]
          : [];
      },
      async evidence(node) {
        return node.kind === "chunk"
          ? [
              {
                evidenceId: "e1",
                chunkId: node.id,
                resourceId: "guide",
                text: "supported answer",
                observations: {
                  origin: "derived",
                  supportApplicability: "not-applicable",
                  confidenceApplicability: "not-applicable",
                  freshnessApplicability: "not-applicable",
                },
              },
            ]
          : [];
      },
    };

    const result = await new BidirectionalNLayerRetriever(
      graph,
      new AdaptiveRouteTraversalScorePolicy(),
    ).retrieve({ question: "What is the supported answer?", principal });

    expect(result.evidence[0]?.evidenceId).toBe("e1");
    expect(result.trace.scoring).toMatchObject({
      profileId: "adaptive-route-v3",
      featureSchemaVersion: "n-layer-adaptive-routing-v1",
      routing: {
        selectedRouteCounts: { resource: 1 },
        queryIntent: "lookup",
      },
    });
  });
});
