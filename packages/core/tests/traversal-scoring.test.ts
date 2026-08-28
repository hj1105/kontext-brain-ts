import { describe, expect, it } from "vitest";
import {
  BalancedTraversalScorePolicy,
  BidirectionalNLayerRetriever,
  CalibratedTraversalScorePolicy,
  DEFAULT_CALIBRATED_SCORING_PROFILE,
  type Principal,
  type SearchGraphPort,
  type TraversalScoringProfile,
  scoringProfileDigest,
} from "../src/index.js";

const principal: Principal = {
  organizationId: "acme",
  subjectId: "user:1",
  groupIds: ["engineering"],
};

describe("traversal scoring profiles", () => {
  it("keeps legacy-v1 scoring mathematically compatible", () => {
    const policy = new BalancedTraversalScorePolicy();
    const seed = policy.seedScore({ node: { kind: "chunk", id: "c1" }, score: 0.9 });
    const edge = policy.edgeScore(
      seed,
      {
        from: { kind: "chunk", id: "c1" },
        to: { kind: "fact", id: "f1" },
        operation: "lift",
        confidence: 0.7,
        queryRelevance: 0.8,
        evidenceSupport: 0.8,
      },
      2,
    );
    const evidence = policy.evidenceScore(edge, {
      evidenceId: "e1",
      chunkId: "c1",
      resourceId: "r1",
      text: "answer",
      score: 0.6,
    });

    expect(edge).toBeCloseTo(0.9 * 0.7 * 0.9 * 0.92 ** 2);
    expect(evidence).toBeCloseTo(edge * 0.6);
  });

  it("produces a stable digest independent of object key insertion order", () => {
    const profile = DEFAULT_CALIBRATED_SCORING_PROFILE;
    const reordered = {
      version: profile.version,
      id: profile.id,
      featureSchemaVersion: profile.featureSchemaVersion,
      evidence: { ...profile.evidence },
      supportEncoding: { ...profile.supportEncoding },
      edge: { ...profile.edge },
      seed: { ...profile.seed },
      description: profile.description,
    } as TraversalScoringProfile;

    expect(scoringProfileDigest(reordered)).toBe(scoringProfileDigest(profile));
  });

  it("accepts later profile versions and rejects incomplete profile payloads", () => {
    expect(
      () =>
        new CalibratedTraversalScorePolicy({
          ...DEFAULT_CALIBRATED_SCORING_PROFILE,
          version: 2,
        }),
    ).not.toThrow();
    expect(
      () =>
        new CalibratedTraversalScorePolicy({
          ...DEFAULT_CALIBRATED_SCORING_PROFILE,
          edge: undefined,
        } as unknown as TraversalScoringProfile),
    ).toThrow("profile.edge.hopFactor");
    expect(
      () =>
        new CalibratedTraversalScorePolicy({
          ...DEFAULT_CALIBRATED_SCORING_PROFILE,
          hiddenKnob: 0.5,
        } as TraversalScoringProfile),
    ).toThrow("profile.hiddenKnob");
  });

  it.each([
    {
      name: "an out-of-range score",
      profile: {
        ...DEFAULT_CALIBRATED_SCORING_PROFILE,
        seed: { ...DEFAULT_CALIBRATED_SCORING_PROFILE.seed, exactMatchScore: 1.01 },
      },
      message: "profile.seed.exactMatchScore must be between zero and one",
    },
    {
      name: "a non-finite factor",
      profile: {
        ...DEFAULT_CALIBRATED_SCORING_PROFILE,
        edge: { ...DEFAULT_CALIBRATED_SCORING_PROFILE.edge, hopFactor: Number.NaN },
      },
      message: "profile.edge.hopFactor must be finite",
    },
    {
      name: "support weights that do not sum to one",
      profile: {
        ...DEFAULT_CALIBRATED_SCORING_PROFILE,
        supportEncoding: {
          ...DEFAULT_CALIBRATED_SCORING_PROFILE.supportEncoding,
          reliabilityWeight: 0.6,
        },
      },
      message: "Support encoding reliability, diversity, and volume weights must sum to one",
    },
  ])("rejects $name", ({ profile, message }) => {
    expect(() => new CalibratedTraversalScorePolicy(profile)).toThrow(message);
  });

  it("applies one hop factor per edge instead of repeatedly exponentiating total depth", () => {
    const policy = new CalibratedTraversalScorePolicy();
    const input = {
      operation: "ground" as const,
      fromKind: "resource" as const,
      toKind: "chunk" as const,
      observations: { structural: { kind: "deterministic" as const } },
    };

    expect(policy.edgeScore(1, input, 1).score).toBeCloseTo(policy.edgeScore(1, input, 7).score);
  });

  it("treats missing optional edge signals as neutral and reports them explicitly", () => {
    const policy = new CalibratedTraversalScorePolicy();
    const scored = policy.edgeScore(
      1,
      {
        operation: "ground",
        fromKind: "resource",
        toKind: "chunk",
        observations: { structural: { kind: "deterministic" } },
      },
      1,
    );

    expect(scored.factors.query).toBe(1);
    expect(scored.factors.support).toBe(1);
    expect(scored.missingSignals).toEqual(["edge.query", "edge.support"]);
    expect(scored.observations).not.toMatchObject({ query: 0.8, support: 0.8 });
  });

  it("treats explicitly inapplicable evidence signals as neutral without reporting them missing", () => {
    const scored = new CalibratedTraversalScorePolicy().evidenceScore(0.8, {
      factStatus: "active",
      observations: {
        origin: "curated",
        supportApplicability: "not-applicable",
        confidenceApplicability: "not-applicable",
        freshnessApplicability: "not-applicable",
      },
    });

    expect(scored.score).toBeCloseTo(0.8);
    expect(scored.factors).toMatchObject({ support: 1, confidence: 1, freshness: 1 });
    expect(scored.missingSignals).toEqual([]);
    expect(scored.observations).toMatchObject({
      supportApplicability: "not-applicable",
      confidenceApplicability: "not-applicable",
      freshnessApplicability: "not-applicable",
    });
  });

  it("is monotonic when query match and evidence support improve", () => {
    const policy = new CalibratedTraversalScorePolicy();
    const base = {
      operation: "expand" as const,
      fromKind: "entity" as const,
      toKind: "fact" as const,
      observations: {
        structural: { kind: "extracted" as const, confidence: 0.9 },
        query: {
          lexical: { rank: 8, candidateCount: 10 },
        },
        support: {
          activeEvidenceCount: 1,
          distinctResourceCount: 1,
          conflictCount: 1,
        },
      },
    };
    const stronger = {
      ...base,
      observations: {
        ...base.observations,
        query: { lexical: { rank: 1, candidateCount: 10 } },
        support: {
          activeEvidenceCount: 5,
          curatedEvidenceCount: 3,
          distinctResourceCount: 3,
          conflictCount: 0,
        },
      },
    };

    expect(policy.edgeScore(1, stronger, 1).score).toBeGreaterThan(
      policy.edgeScore(1, base, 1).score,
    );
  });

  it("ranks active, confident, fresh evidence above otherwise equivalent weaker evidence", () => {
    const policy = new CalibratedTraversalScorePolicy();
    const score = (factStatus: "active" | "conflict", confidence: number, freshnessDays: number) =>
      policy.evidenceScore(1, {
        factStatus,
        observations: {
          origin: "curated",
          confidence,
          freshnessDays,
          supportApplicability: "not-applicable",
        },
      }).score;

    expect(score("active", 0.8, 0)).toBeGreaterThan(score("conflict", 0.8, 0));
    expect(score("active", 0.9, 0)).toBeGreaterThan(score("active", 0.2, 0));
    expect(score("active", 0.8, 0)).toBeGreaterThan(score("active", 0.8, 720));
  });

  it("puts profile identity, raw factors, and missing signals in retrieval output", async () => {
    const graph: SearchGraphPort = {
      async seed() {
        return [
          {
            node: { kind: "chunk", id: "c1" },
            observations: {
              providers: ["test-lexical"],
              query: { lexical: { rank: 1, candidateCount: 2, normalizedScore: 0.9 } },
            },
          },
        ];
      },
      async neighbors() {
        return [];
      },
      async evidence() {
        return [
          {
            evidenceId: "e1",
            chunkId: "c1",
            resourceId: "r1",
            text: "answer",
            factStatus: "active",
            observations: { origin: "curated", freshnessDays: 2 },
          },
        ];
      },
    };

    const result = await new BidirectionalNLayerRetriever(
      graph,
      new CalibratedTraversalScorePolicy(),
    ).retrieve({ question: "answer", principal });

    expect(result.trace.scoring.profileId).toBe("calibrated-v2");
    expect(result.trace.scoring.missingSignals).toEqual([
      "evidence.confidence",
      "evidence.support",
    ]);
    expect(result.trace.scoring.missingSignalCounts).toEqual({
      "evidence.confidence": 1,
      "evidence.support": 1,
    });
    expect(result.trace.averageSelectedPathLength).toBe(0);
    expect(result.trace.seedProviderCounts).toEqual({ "test-lexical": 1 });
    expect(result.evidence[0]?.scoreBreakdown).toMatchObject({
      profileId: "calibrated-v2",
      seed: {
        observations: {
          lexical: 1,
          lexicalRank: 1,
          lexicalCandidateCount: 2,
          lexicalNormalizedScore: 0.9,
        },
      },
      evidence: { observations: { origin: "curated", freshnessDays: 2 } },
    });
  });

  it("runs a configured shadow frontier without changing active evidence", async () => {
    const graph: SearchGraphPort = {
      async seed() {
        return [
          {
            node: { kind: "chunk", id: "c1" },
            observations: {
              query: { lexical: { rank: 1, candidateCount: 1 } },
            },
          },
        ];
      },
      async neighbors() {
        return [];
      },
      async evidence() {
        return [
          {
            evidenceId: "e1",
            chunkId: "c1",
            resourceId: "r1",
            text: "answer",
            observations: { origin: "derived", freshnessDays: 0 },
          },
        ];
      },
    };
    const active = new CalibratedTraversalScorePolicy();
    const shadow = new CalibratedTraversalScorePolicy({
      ...DEFAULT_CALIBRATED_SCORING_PROFILE,
      id: "shadow-v2",
      version: 2,
    });
    const result = await new BidirectionalNLayerRetriever(graph, {
      async resolve() {
        return active;
      },
      async resolveShadow() {
        return shadow;
      },
    }).retrieve({ question: "answer", principal });

    expect(result.evidence.map((hit) => hit.evidenceId)).toEqual(["e1"]);
    expect(result.trace.scoring.shadow).toMatchObject({
      profileId: "shadow-v2",
      status: "completed",
      topKOverlapRatio: 1,
      normalizedRankDisagreement: 0,
    });
  });

  it("does not fail active retrieval when shadow profile resolution fails", async () => {
    const graph: SearchGraphPort = {
      async seed() {
        return [
          {
            node: { kind: "chunk", id: "c1" },
            observations: { query: { exactMatch: true } },
          },
        ];
      },
      async neighbors() {
        return [];
      },
      async evidence() {
        return [
          {
            evidenceId: "e1",
            chunkId: "c1",
            resourceId: "r1",
            text: "answer",
            factStatus: "active",
            observations: { origin: "curated", freshnessDays: 0 },
          },
        ];
      },
    };
    const result = await new BidirectionalNLayerRetriever(graph, {
      async resolve() {
        return new CalibratedTraversalScorePolicy();
      },
      async resolveShadow() {
        throw new Error("shadow store unavailable");
      },
    }).retrieve({ question: "answer", principal });

    expect(result.evidence[0]?.evidenceId).toBe("e1");
    expect(result.trace.scoring.shadow).toMatchObject({
      status: "failed",
      profileId: "unresolved-shadow",
      errorName: "Error",
    });
  });

  it("keeps active evidence when shadow traversal fails and closes the shared session once", async () => {
    let opened = 0;
    let closed = 0;
    let seedCalls = 0;
    const session = {
      async close() {
        closed++;
      },
    };
    const receivedSessions: unknown[] = [];
    const graph: SearchGraphPort = {
      async openSession() {
        opened++;
        return session;
      },
      async seed(_question, _principal, receivedSession) {
        receivedSessions.push(receivedSession);
        seedCalls++;
        if (seedCalls === 2) throw new Error("shadow traversal failed");
        return [
          {
            node: { kind: "chunk", id: "c1" },
            observations: { query: { exactMatch: true } },
          },
        ];
      },
      async neighbors() {
        return [];
      },
      async evidence() {
        return [
          {
            evidenceId: "e1",
            chunkId: "c1",
            resourceId: "r1",
            text: "active answer",
            observations: {
              origin: "curated",
              supportApplicability: "not-applicable",
              confidenceApplicability: "not-applicable",
              freshnessApplicability: "not-applicable",
            },
          },
        ];
      },
    };
    const result = await new BidirectionalNLayerRetriever(graph, {
      async resolve() {
        return new CalibratedTraversalScorePolicy();
      },
      async resolveShadow() {
        return new CalibratedTraversalScorePolicy({
          ...DEFAULT_CALIBRATED_SCORING_PROFILE,
          id: "shadow-failure",
          version: 2,
        });
      },
    }).retrieve({ question: "answer", principal });

    expect(result.evidence.map((hit) => hit.evidenceId)).toEqual(["e1"]);
    expect(result.trace.scoring.shadow).toMatchObject({
      profileId: "shadow-failure",
      status: "failed",
      errorName: "Error",
    });
    expect(opened).toBe(1);
    expect(closed).toBe(1);
    expect(receivedSessions).toEqual([session, session]);
  });

  it("reports shadow rank disagreement without replacing the active ordering", async () => {
    const graph: SearchGraphPort = {
      async seed() {
        return [
          {
            node: { kind: "chunk", id: "lexical-first" },
            observations: {
              query: {
                lexical: { rank: 1, candidateCount: 2 },
                vector: { rank: 2, candidateCount: 2 },
              },
            },
          },
          {
            node: { kind: "chunk", id: "vector-first" },
            observations: {
              query: {
                lexical: { rank: 2, candidateCount: 2 },
                vector: { rank: 1, candidateCount: 2 },
              },
            },
          },
        ];
      },
      async neighbors() {
        return [];
      },
      async evidence(node) {
        return [
          {
            evidenceId: `evidence:${node.id}`,
            chunkId: node.id,
            resourceId: `resource:${node.id}`,
            text: node.id,
            observations: {
              origin: "curated",
              supportApplicability: "not-applicable",
              confidenceApplicability: "not-applicable",
              freshnessApplicability: "not-applicable",
            },
          },
        ];
      },
    };
    const lexicalProfile = {
      ...DEFAULT_CALIBRATED_SCORING_PROFILE,
      id: "lexical-active",
      version: 2,
      seed: {
        ...DEFAULT_CALIBRATED_SCORING_PROFILE.seed,
        lexicalWeight: 10,
        vectorWeight: 1,
      },
    };
    const vectorProfile = {
      ...DEFAULT_CALIBRATED_SCORING_PROFILE,
      id: "vector-shadow",
      version: 2,
      seed: {
        ...DEFAULT_CALIBRATED_SCORING_PROFILE.seed,
        lexicalWeight: 1,
        vectorWeight: 10,
      },
    };
    const result = await new BidirectionalNLayerRetriever(graph, {
      async resolve() {
        return new CalibratedTraversalScorePolicy(lexicalProfile);
      },
      async resolveShadow() {
        return new CalibratedTraversalScorePolicy(vectorProfile);
      },
    }).retrieve({ question: "answer", principal });

    expect(result.evidence.map((hit) => hit.evidenceId)).toEqual([
      "evidence:lexical-first",
      "evidence:vector-first",
    ]);
    expect(result.trace.scoring.shadow).toMatchObject({
      profileId: "vector-shadow",
      status: "completed",
      topKOverlapRatio: 1,
      normalizedRankDisagreement: 0.5,
    });
  });
});
