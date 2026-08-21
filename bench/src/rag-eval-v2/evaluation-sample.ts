import { createHash } from "node:crypto";
import type { BenchmarkQuery, DatasetBundle } from "./contracts.js";

export interface EvaluationSampleCategory {
  readonly category: string;
  readonly population: number;
  readonly selected: number;
}

export interface EvaluationSampleManifest {
  readonly schemaVersion: "1.0.0";
  readonly datasetId: DatasetBundle["id"];
  readonly method: "deterministic-proportional-category-stratified";
  readonly seed: number;
  readonly requested: number;
  readonly population: number;
  readonly selected: number;
  readonly categories: readonly EvaluationSampleCategory[];
  readonly queryIds: readonly string[];
  readonly sampleDigest: string;
}

export interface EvaluationSample {
  readonly queries: readonly BenchmarkQuery[];
  readonly manifest: EvaluationSampleManifest;
}

export function createEvaluationSample(
  bundle: DatasetBundle,
  requested: number,
  seed: number,
): EvaluationSample {
  if (!Number.isInteger(requested) || requested <= 0) throw new Error("requested sample size must be positive");
  if (!Number.isInteger(seed)) throw new Error("sample seed must be an integer");
  const target = Math.min(requested, bundle.queries.length);
  const byCategory = new Map<string, BenchmarkQuery[]>();
  for (const query of bundle.queries) {
    const category = query.category.trim() || "<uncategorized>";
    const values = byCategory.get(category) ?? [];
    values.push(query);
    byCategory.set(category, values);
  }
  const categories = [...byCategory.keys()].sort();
  const allocation = allocateProportionally(
    categories.map((category) => ({ category, population: byCategory.get(category)!.length })),
    target,
    bundle.queries.length,
  );
  const selectedIds = new Set<string>();
  for (const category of categories) {
    const candidates = [...byCategory.get(category)!].sort((left, right) =>
      stableKey(seed, bundle.id, category, left.id).localeCompare(
        stableKey(seed, bundle.id, category, right.id),
      ),
    );
    for (const query of candidates.slice(0, allocation.get(category) ?? 0)) selectedIds.add(query.id);
  }
  const queries = bundle.queries.filter((query) => selectedIds.has(query.id));
  const queryIds = queries.map((query) => query.id);
  const sampleDigest = createHash("sha256")
    .update("deterministic-proportional-category-stratified")
    .update("\0")
    .update(String(seed))
    .update("\0")
    .update(bundle.id)
    .update("\0")
    .update(queries.map((query) => JSON.stringify({
      id: query.id,
      text: query.text,
      referenceAnswer: query.referenceAnswer,
      goldEvidenceIds: query.goldEvidenceIds,
      goldEvidenceText: query.goldEvidenceText,
      answerable: query.answerable,
      category: query.category,
    })).join("\0"))
    .digest("hex");
  return {
    queries,
    manifest: {
      schemaVersion: "1.0.0",
      datasetId: bundle.id,
      method: "deterministic-proportional-category-stratified",
      seed,
      requested,
      population: bundle.queries.length,
      selected: queries.length,
      categories: categories.map((category) => ({
        category,
        population: byCategory.get(category)!.length,
        selected: allocation.get(category) ?? 0,
      })),
      queryIds,
      sampleDigest,
    },
  };
}

function allocateProportionally(
  categories: readonly { readonly category: string; readonly population: number }[],
  target: number,
  population: number,
): Map<string, number> {
  const allocation = new Map(categories.map(({ category }) => [category, 0]));
  if (target === 0 || population === 0) return allocation;
  let remaining = target;
  if (target >= categories.length) {
    for (const { category, population: categoryPopulation } of categories) {
      if (categoryPopulation === 0) continue;
      allocation.set(category, 1);
      remaining -= 1;
    }
  }
  while (remaining > 0) {
    const eligible = categories
      .filter(({ category, population: categoryPopulation }) =>
        (allocation.get(category) ?? 0) < categoryPopulation,
      )
      .sort((left, right) => {
        const leftGap = target * left.population / population - (allocation.get(left.category) ?? 0);
        const rightGap = target * right.population / population - (allocation.get(right.category) ?? 0);
        return rightGap - leftGap || left.category.localeCompare(right.category);
      });
    const next = eligible[0];
    if (!next) break;
    allocation.set(next.category, (allocation.get(next.category) ?? 0) + 1);
    remaining -= 1;
  }
  return allocation;
}

function stableKey(seed: number, datasetId: string, category: string, queryId: string): string {
  return createHash("sha256")
    .update(String(seed))
    .update("\0")
    .update(datasetId)
    .update("\0")
    .update(category)
    .update("\0")
    .update(queryId)
    .digest("hex");
}
