import { DEFAULT_CALIBRATED_SCORING_PROFILE } from "@kontext-brain/core";
import { describe, expect, it } from "vitest";
import {
  type ScoringFeatureCandidate,
  assignScoringSplits,
  compareScoringProfiles,
  evaluateScoringProfile,
  selectProfileByValidation,
} from "./scoring-profile-evaluation.js";

const candidates: ScoringFeatureCandidate[] = [
  candidate("q1", "relevant", true, 1),
  candidate("q1", "noise", false, 5),
  candidate("q2", "relevant-2", true, 1),
  candidate("q2", "noise-2", false, 4),
];

describe("scoring profile evaluation", () => {
  it("deterministically splits queries and evaluates ranking metrics", () => {
    expect(assignScoringSplits(candidates)).toEqual(assignScoringSplits(candidates));
    const evaluation = evaluateScoringProfile(DEFAULT_CALIBRATED_SCORING_PROFILE, candidates, 1);
    expect(evaluation).toMatchObject({ queries: 2, recallAtK: 1, ndcgAtK: 1 });
  });

  it("uses paired bootstrap intervals and a recall guardrail for selection", () => {
    const baseline = evaluateScoringProfile(DEFAULT_CALIBRATED_SCORING_PROFILE, candidates, 1);
    const tuned = { ...baseline, profileId: "tuned", ndcgAtK: baseline.ndcgAtK + 0.01 };
    expect(compareScoringProfiles(baseline, tuned, 100).ndcgDifference).toMatchObject({
      mean: 0,
    });
    expect(selectProfileByValidation([tuned], baseline)?.profileId).toBe("tuned");
  });
});

function candidate(
  queryId: string,
  candidateId: string,
  relevant: boolean,
  lexicalRank: number,
): ScoringFeatureCandidate {
  return {
    queryId,
    category: "lookup",
    answerable: true,
    candidateId,
    relevant,
    seed: {
      observations: {
        query: { lexical: { rank: lexicalRank, candidateCount: 5 } },
      },
    },
    edges: [],
    evidence: {
      factStatus: "active",
      observations: {
        origin: "curated",
        freshnessDays: 0,
        support: { activeEvidenceCount: 1, distinctResourceCount: 1 },
      },
    },
  };
}
