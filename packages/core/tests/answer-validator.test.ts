import { describe, expect, it } from "vitest";
import {
  CitationAnswerValidator,
  NoAccessibleEvidenceError,
  type RankedEvidenceHit,
  UnsupportedAnswerError,
} from "../src/index.js";

function evidence(status: "active" | "conflict" = "active"): RankedEvidenceHit {
  return {
    evidenceId: "evidence-1",
    resourceId: "resource-1",
    chunkId: "chunk-1",
    text: "Order 42 was paid",
    score: 1,
    factStatus: status,
    path: [],
  };
}

describe("CitationAnswerValidator", () => {
  it("rejects answers without accessible Evidence", async () => {
    await expect(new CitationAnswerValidator().validate("paid", [])).rejects.toBeInstanceOf(
      NoAccessibleEvidenceError,
    );
  });

  it("rejects unsupported answers and accepts a cited answer", async () => {
    const validator = new CitationAnswerValidator();
    await expect(validator.validate("Order 42 was paid", [evidence()])).rejects.toBeInstanceOf(
      UnsupportedAnswerError,
    );
    await expect(
      validator.validate("Order 42 was paid [Evidence evidence-1]", [evidence()]),
    ).resolves.toBeUndefined();
  });

  it("requires uncertainty language for a conflicting fact", async () => {
    const validator = new CitationAnswerValidator();
    await expect(
      validator.validate("It was paid [Evidence evidence-1]", [evidence("conflict")]),
    ).rejects.toBeInstanceOf(UnsupportedAnswerError);
    await expect(
      validator.validate("Sources conflict [Evidence evidence-1]", [evidence("conflict")]),
    ).resolves.toBeUndefined();
  });
});
