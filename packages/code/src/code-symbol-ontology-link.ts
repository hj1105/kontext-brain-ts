import { createHash } from "node:crypto";
import type { CodeSymbolOntologyLink } from "./domain.js";

export type CodeSymbolOntologyLinkInput = Omit<CodeSymbolOntologyLink, "linkId">;

export function createCodeSymbolOntologyLink(
  input: CodeSymbolOntologyLinkInput,
): CodeSymbolOntologyLink {
  const { linkId: _ignoredLinkId, ...safeInput } = input as CodeSymbolOntologyLink;
  const normalized: CodeSymbolOntologyLinkInput = {
    ...safeInput,
    evidenceIds: uniqueSorted(safeInput.evidenceIds),
  };
  return Object.freeze({
    ...normalized,
    linkId: `code-symbol-ontology-link:${createHash("sha256")
      .update(stableJson(normalized))
      .digest("hex")}`,
  });
}

export function isCodeSymbolOntologyLinkValid(link: CodeSymbolOntologyLink): boolean {
  const { linkId: _linkId, ...input } = link;
  return stableJson(link) === stableJson(createCodeSymbolOntologyLink(input));
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
