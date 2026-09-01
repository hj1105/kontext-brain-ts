import type {
  CodeSymbolIdentity,
  PlannedSymbolBinding,
  PlannedSymbolBindingIssue,
  PlannedSymbolRecord,
  PlannedSymbolResolution,
} from "./domain.js";

export function resolvePlannedSymbols(
  planned: readonly PlannedSymbolRecord[],
  symbols: readonly { readonly symbolId: string; readonly identity: CodeSymbolIdentity }[],
): PlannedSymbolResolution {
  const symbolById = new Map(symbols.map((symbol) => [symbol.symbolId, symbol] as const));
  const bindings: PlannedSymbolBinding[] = [];
  const issues: PlannedSymbolBindingIssue[] = [];
  const seen = new Set<string>();

  for (const record of [...planned].sort((left, right) =>
    left.plannedSymbolId.localeCompare(right.plannedSymbolId),
  )) {
    if (seen.has(record.plannedSymbolId)) {
      throw new Error(`Duplicate Planned Symbol: ${record.plannedSymbolId}`);
    }
    seen.add(record.plannedSymbolId);

    if (record.boundSymbolId) {
      if (symbolById.has(record.boundSymbolId)) {
        bindings.push({
          plannedSymbolId: record.plannedSymbolId,
          symbolId: record.boundSymbolId,
          boundBy: "recorded_binding",
        });
      } else {
        issues.push({
          plannedSymbolId: record.plannedSymbolId,
          code: "bound_symbol_missing",
          candidateSymbolIds: [],
        });
      }
      continue;
    }

    if (symbolById.has(record.plannedSymbolId)) {
      bindings.push({
        plannedSymbolId: record.plannedSymbolId,
        symbolId: record.plannedSymbolId,
        boundBy: "existing_symbol_id",
      });
      continue;
    }

    const candidates = symbols
      .filter((symbol) => identityMatches(record.intendedIdentity, symbol.identity))
      .sort((left, right) => left.symbolId.localeCompare(right.symbolId));
    if (candidates.length === 1 && candidates[0]) {
      bindings.push({
        plannedSymbolId: record.plannedSymbolId,
        symbolId: candidates[0].symbolId,
        boundBy: "intended_identity",
      });
    } else {
      issues.push({
        plannedSymbolId: record.plannedSymbolId,
        code: candidates.length === 0 ? "identity_not_found" : "identity_ambiguous",
        candidateSymbolIds: candidates.map((candidate) => candidate.symbolId),
      });
    }
  }

  return { bindings, issues };
}

function identityMatches(
  intended: Partial<CodeSymbolIdentity>,
  actual: CodeSymbolIdentity,
): boolean {
  const entries = Object.entries(intended) as Array<
    [keyof CodeSymbolIdentity, CodeSymbolIdentity[keyof CodeSymbolIdentity] | undefined]
  >;
  return (
    entries.length > 0 &&
    entries.every(([key, value]) =>
      value === undefined
        ? true
        : key === "relativePath"
          ? canonicalPath(String(value)) === canonicalPath(actual.relativePath)
          : value === actual[key],
    )
  );
}

function canonicalPath(value: string): string {
  return value.replaceAll("\\", "/").replace(/^\.\//, "");
}
