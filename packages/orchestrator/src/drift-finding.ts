import { createHash } from "node:crypto";
import type { CodeImpactIndex, CodeSymbolOntologyLink } from "@kontext-brain/code";
import type { DriftFinding, NormativeRevisionRef } from "@kontext-brain/spec";

export interface CreateDriftFindingsInput {
  readonly from: NormativeRevisionRef;
  readonly to: NormativeRevisionRef;
  readonly codeRevision: string;
  readonly links: readonly CodeSymbolOntologyLink[];
  readonly impactIndex: CodeImpactIndex;
  readonly createdAt: string;
}

export function createDriftFindings(input: CreateDriftFindingsInput): readonly DriftFinding[] {
  if (input.from.kind !== input.to.kind || input.from.recordId !== input.to.recordId) {
    throw new Error("Drift analysis requires revisions of the same normative record");
  }
  if (input.from.revisionId === input.to.revisionId) return [];

  const bindingLinks = input.links.filter(
    (link) =>
      isBindingLinkValid(link) &&
      link.origin !== "proposed" &&
      link.target.kind === "normative" &&
      link.target.normativeKind === input.from.kind &&
      link.target.recordId === input.from.recordId &&
      link.target.revisionId === input.from.revisionId,
  );
  if (bindingLinks.length === 0) return [];

  const impact = input.impactIndex.findAffectedSymbols(
    uniqueSorted(bindingLinks.map((link) => link.symbolId)),
  );
  const identity = {
    normativeKind: input.to.kind,
    recordId: input.to.recordId,
    fromRevisionId: input.from.revisionId,
    toRevisionId: input.to.revisionId,
    codeRevision: input.codeRevision,
    affectedSymbolIds: uniqueSorted(impact.affectedSymbols.map((symbol) => symbol.symbolId)),
    unresolvedSymbolIds: uniqueSorted(impact.missingSymbolIds),
    codeSymbolOntologyLinkIds: uniqueSorted(bindingLinks.map((link) => link.linkId)),
    evidenceIds: uniqueSorted(bindingLinks.flatMap((link) => link.evidenceIds)),
    status: "open" as const,
    createdAt: input.createdAt,
  };
  return [
    Object.freeze({
      findingId: `drift-finding:${createHash("sha256").update(stableJson(identity)).digest("hex")}`,
      ...identity,
    }),
  ];
}

function isBindingLinkValid(link: CodeSymbolOntologyLink): boolean {
  const { linkId, ...input } = link;
  const normalized = {
    ...input,
    evidenceIds: uniqueSorted(input.evidenceIds),
  };
  return (
    linkId ===
    `code-symbol-ontology-link:${createHash("sha256").update(stableJson(normalized)).digest("hex")}`
  );
}

export function isDriftFindingValid(finding: DriftFinding): boolean {
  const { findingId, ...identity } = finding;
  return (
    findingId ===
    `drift-finding:${createHash("sha256")
      .update(stableJson({ ...identity, status: "open" }))
      .digest("hex")}`
  );
}

function uniqueSorted(values: readonly string[]): readonly string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}

function stableJson(value: unknown): string {
  return JSON.stringify(stableValue(value));
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, nested]) => [key, stableValue(nested)]),
    );
  }
  return value;
}
