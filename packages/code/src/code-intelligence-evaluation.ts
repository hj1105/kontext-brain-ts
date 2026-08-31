import type {
  CodeFileAnalysis,
  CodeIdentityStabilityEvaluation,
  CodeRelationshipEvaluation,
  CodeRelationshipLabel,
  CodeRelationshipPredicate,
  CodeSymbolRecord,
} from "./domain.js";

export function evaluateCodeRelationshipExtraction(
  analyses: readonly CodeFileAnalysis[],
  labels: readonly CodeRelationshipLabel[],
  predicates?: readonly CodeRelationshipPredicate[],
): CodeRelationshipEvaluation {
  const includedPredicates = predicates ? new Set(predicates) : undefined;
  const symbols = new Map(
    analyses.flatMap((analysis) =>
      analysis.symbols.map((symbol) => [symbol.symbolId, symbol] as const),
    ),
  );
  const predictedKeys = new Set<string>();
  for (const analysis of analyses) {
    for (const relationship of analysis.relationships) {
      if (includedPredicates && !includedPredicates.has(relationship.predicate)) continue;
      const subject = symbols.get(relationship.subjectSymbolId);
      if (!subject) continue;
      const object =
        relationship.object.kind === "literal"
          ? relationship.object
          : symbolLabelObject(symbols.get(relationship.object.symbolId));
      if (!object) continue;
      predictedKeys.add(
        relationshipLabelKey({
          subject: {
            relativePath: subject.identity.relativePath,
            qualifiedName: subject.identity.qualifiedName,
          },
          predicate: relationship.predicate,
          object,
        }),
      );
    }
  }

  const labelKeys = new Set(
    labels
      .filter((label) => !includedPredicates || includedPredicates.has(label.predicate))
      .map(relationshipLabelKey),
  );
  const truePositives = Array.from(predictedKeys).filter((key) => labelKeys.has(key)).length;
  const falsePositives = predictedKeys.size - truePositives;
  const falseNegatives = labelKeys.size - truePositives;
  return {
    truePositives,
    falsePositives,
    falseNegatives,
    precision: ratio(truePositives, truePositives + falsePositives),
    recall: ratio(truePositives, truePositives + falseNegatives),
  };
}

export function evaluateCodeIdentityStability(
  before: readonly CodeFileAnalysis[],
  after: readonly CodeFileAnalysis[],
): CodeIdentityStabilityEvaluation {
  const afterBySemanticKey = new Map(
    after
      .flatMap((analysis) => analysis.symbols)
      .filter((symbol) => symbol.behaviorBearing)
      .map((symbol) => [semanticSymbolKey(symbol), symbol] as const),
  );
  const comparable = before
    .flatMap((analysis) => analysis.symbols)
    .filter((symbol) => symbol.behaviorBearing)
    .flatMap((symbol) => {
      const current = afterBySemanticKey.get(semanticSymbolKey(symbol));
      return current ? [[symbol, current] as const] : [];
    });
  const stableSymbolIds = comparable.filter(
    ([left, right]) => left.symbolId === right.symbolId,
  ).length;
  const stableContentHashes = comparable.filter(
    ([left, right]) => left.contentHash === right.contentHash,
  ).length;
  return {
    comparableBehaviorSymbols: comparable.length,
    stableSymbolIds,
    stableContentHashes,
    identityStability: ratio(stableSymbolIds, comparable.length),
    contentStability: ratio(stableContentHashes, comparable.length),
  };
}

function symbolLabelObject(
  symbol: CodeSymbolRecord | undefined,
): CodeRelationshipLabel["object"] | undefined {
  return symbol
    ? {
        kind: "symbol",
        relativePath: symbol.identity.relativePath,
        qualifiedName: symbol.identity.qualifiedName,
      }
    : undefined;
}

function relationshipLabelKey(label: CodeRelationshipLabel): string {
  return JSON.stringify([
    label.subject.relativePath,
    label.subject.qualifiedName,
    label.predicate,
    label.object.kind,
    label.object.kind === "symbol" ? label.object.relativePath : "",
    label.object.kind === "symbol" ? label.object.qualifiedName : label.object.value,
  ]);
}

function semanticSymbolKey(symbol: CodeSymbolRecord): string {
  return JSON.stringify([
    symbol.identity.codebaseId,
    symbol.identity.relativePath,
    symbol.identity.kind,
    symbol.identity.qualifiedName,
  ]);
}

function ratio(numerator: number, denominator: number): number {
  return denominator === 0 ? 1 : numerator / denominator;
}
