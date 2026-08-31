import { describe, expect, it } from "vitest";
import { createCodeSymbolOntologyLink, isCodeSymbolOntologyLinkValid } from "../src/index.js";

describe("Code Symbol-Ontology Link", () => {
  it("retains link origin and canonical Evidence in an immutable ID", () => {
    const link = createCodeSymbolOntologyLink({
      symbolId: "symbol:handler",
      target: {
        kind: "normative",
        normativeKind: "decision",
        recordId: "decision:handler",
        revisionId: "decision:handler@1",
      },
      origin: "curated",
      evidenceIds: ["evidence:b", "evidence:a", "evidence:b"],
      createdAt: "2026-08-28T10:00:00.000Z",
    });

    expect(link.evidenceIds).toEqual(["evidence:a", "evidence:b"]);
    expect(isCodeSymbolOntologyLinkValid(link)).toBe(true);
    expect(isCodeSymbolOntologyLinkValid({ ...link, origin: "proposed" })).toBe(false);
  });
});
