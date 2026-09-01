import type {
  CodeFileAnalysis,
  CodeImpactResult,
  CodeRelationship,
  CodeSymbolChange,
  CodeSymbolRecord,
  ReverseCodeDependency,
} from "./domain.js";

export class CodeImpactIndex {
  private readonly symbols = new Map<string, CodeSymbolRecord>();
  private readonly reverseDependencies = new Map<string, ReverseCodeDependency[]>();

  constructor(analyses: readonly CodeFileAnalysis[]) {
    for (const analysis of analyses) {
      for (const symbol of analysis.symbols) {
        const existing = this.symbols.get(symbol.symbolId);
        if (existing && !sameSymbol(existing, symbol)) {
          throw new Error(`Conflicting Code Symbol records for "${symbol.symbolId}"`);
        }
        this.symbols.set(symbol.symbolId, symbol);
      }
    }

    const seenRelationships = new Set<string>();
    for (const analysis of analyses) {
      for (const relationship of analysis.relationships) {
        if (seenRelationships.has(relationship.relationshipId)) continue;
        seenRelationships.add(relationship.relationshipId);
        this.addReverseDependency(relationship);
      }
    }
  }

  getSymbol(symbolId: string): CodeSymbolRecord | undefined {
    return this.symbols.get(symbolId);
  }

  findDirectDependents(symbolId: string): readonly ReverseCodeDependency[] {
    return [...(this.reverseDependencies.get(symbolId) ?? [])].sort(compareDependency);
  }

  findAffectedSymbols(symbolIds: readonly string[]): CodeImpactResult {
    const requestedSymbolIds = uniqueSorted(symbolIds);
    const missingSymbolIds = requestedSymbolIds.filter((symbolId) => !this.symbols.has(symbolId));
    const queue = requestedSymbolIds.filter((symbolId) => this.symbols.has(symbolId));
    const visited = new Set(queue);
    const traversed = new Map<string, ReverseCodeDependency>();

    for (let index = 0; index < queue.length; index++) {
      const dependencySymbolId = queue[index];
      if (!dependencySymbolId) continue;
      for (const dependency of this.findDirectDependents(dependencySymbolId)) {
        traversed.set(dependency.relationshipId, dependency);
        if (!visited.has(dependency.dependentSymbolId)) {
          visited.add(dependency.dependentSymbolId);
          queue.push(dependency.dependentSymbolId);
        }
      }
    }

    return {
      requestedSymbolIds,
      affectedSymbols: Array.from(visited)
        .map((symbolId) => this.symbols.get(symbolId))
        .filter((symbol): symbol is CodeSymbolRecord => Boolean(symbol))
        .sort(compareSymbol),
      traversedDependencies: Array.from(traversed.values()).sort(compareDependency),
      missingSymbolIds,
    };
  }

  private addReverseDependency(relationship: CodeRelationship): void {
    if (relationship.object.kind !== "symbol") return;
    const dependency: ReverseCodeDependency = {
      dependencySymbolId: relationship.object.symbolId,
      dependentSymbolId: relationship.subjectSymbolId,
      predicate: relationship.predicate,
      relationshipId: relationship.relationshipId,
    };
    const dependencies = this.reverseDependencies.get(dependency.dependencySymbolId) ?? [];
    dependencies.push(dependency);
    this.reverseDependencies.set(dependency.dependencySymbolId, dependencies);
  }
}

export function compareCodeAnalyses(
  before: CodeFileAnalysis,
  after: CodeFileAnalysis,
): readonly CodeSymbolChange[] {
  if (before.codebaseId !== after.codebaseId || before.relativePath !== after.relativePath) {
    throw new Error("Code analysis comparison requires the same Codebase and relative path");
  }
  const beforeSymbols = new Map(before.symbols.map((symbol) => [symbol.symbolId, symbol]));
  const afterSymbols = new Map(after.symbols.map((symbol) => [symbol.symbolId, symbol]));
  const symbolIds = uniqueSorted([...beforeSymbols.keys(), ...afterSymbols.keys()]);
  const changes: CodeSymbolChange[] = [];

  for (const symbolId of symbolIds) {
    const previous = beforeSymbols.get(symbolId);
    const current = afterSymbols.get(symbolId);
    if (!previous && current) {
      changes.push({ kind: "added", symbolId, after: current });
    } else if (previous && !current) {
      changes.push({ kind: "removed", symbolId, before: previous });
    } else if (previous && current && previous.contentHash !== current.contentHash) {
      changes.push({ kind: "modified", symbolId, before: previous, after: current });
    }
  }
  return changes;
}

function sameSymbol(left: CodeSymbolRecord, right: CodeSymbolRecord): boolean {
  return (
    JSON.stringify(left.identity) === JSON.stringify(right.identity) &&
    left.contentHash === right.contentHash
  );
}

function compareSymbol(left: CodeSymbolRecord, right: CodeSymbolRecord): number {
  return (
    left.identity.relativePath.localeCompare(right.identity.relativePath) ||
    left.position - right.position ||
    left.symbolId.localeCompare(right.symbolId)
  );
}

function compareDependency(left: ReverseCodeDependency, right: ReverseCodeDependency): number {
  return (
    left.dependentSymbolId.localeCompare(right.dependentSymbolId) ||
    left.predicate.localeCompare(right.predicate) ||
    left.relationshipId.localeCompare(right.relationshipId)
  );
}

function uniqueSorted(values: readonly string[]): readonly string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}
