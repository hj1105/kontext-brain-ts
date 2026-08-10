import type { RankedEvidenceHit } from "./bidirectional-retriever.js";

export interface AnswerGroundingValidator {
  validate(answer: string, evidence: readonly RankedEvidenceHit[]): Promise<void>;
}

export class NoAccessibleEvidenceError extends Error {
  constructor() {
    super("No accessible active Evidence is available for this answer");
    this.name = "NoAccessibleEvidenceError";
  }
}

export class UnsupportedAnswerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnsupportedAnswerError";
  }
}

/** Minimum fail-closed validator; richer claim coverage validators can replace it. */
export class CitationAnswerValidator implements AnswerGroundingValidator {
  async validate(answer: string, evidence: readonly RankedEvidenceHit[]): Promise<void> {
    if (evidence.length === 0) throw new NoAccessibleEvidenceError();
    const cited = evidence.filter((item) => answer.includes(item.evidenceId));
    if (cited.length === 0) {
      throw new UnsupportedAnswerError("Answer does not cite any accessible Evidence ID");
    }
    if (
      cited.some((item) => item.factStatus === "conflict") &&
      !/(conflict|uncertain|disagree|충돌|불확실|상충|서로 다)/i.test(answer)
    ) {
      throw new UnsupportedAnswerError("Answer cites conflicting Evidence without uncertainty");
    }
  }
}
