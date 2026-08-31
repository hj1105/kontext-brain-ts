import type { ChangeBundle, LogicWorkItem } from "@kontext-brain/spec";
import type { ChangeBundleIntegrationPlan, PlanChangeBundleIntegrationInput } from "./domain.js";

export function planChangeBundleIntegration(
  input: PlanChangeBundleIntegrationInput,
): ChangeBundleIntegrationPlan {
  const workItems = uniqueBy(input.workItems, (workItem) => workItem.workItemId, "Logic Work Item");
  const bundles = uniqueBy(
    input.changeBundles,
    (bundle) => bundle.workItemId,
    "Change Bundle Work Item",
  );
  const authors = uniqueBy(input.authors, (author) => author.workItemId, "Change Bundle author");
  if (workItems.size === 0) throw new Error("Integration requires at least one Logic Work Item");
  if (bundles.size !== workItems.size) {
    throw new Error("Integration requires exactly one accepted Change Bundle per Logic Work Item");
  }

  for (const [workItemId, workItem] of workItems) {
    if (workItem.taskId !== input.taskId) {
      throw new Error(`Logic Work Item ${workItemId} belongs to another Task`);
    }
    const bundle = bundles.get(workItemId);
    if (!bundle || bundle.taskId !== input.taskId) {
      throw new Error(`Logic Work Item ${workItemId} has no Task-matching Change Bundle`);
    }
    if (!authors.has(workItemId)) {
      throw new Error(`Logic Work Item ${workItemId} has no sidecar-observed author provider`);
    }
    const allowedPaths = new Set(workItem.allowedPaths.map(canonicalPath));
    const outsidePath = bundle.changedPaths
      .map(canonicalPath)
      .find((changedPath) => !allowedPaths.has(changedPath));
    if (outsidePath) {
      throw new Error(`Change Bundle ${bundle.bundleId} changes out-of-scope path ${outsidePath}`);
    }
  }

  const symbolOwner = new Map<string, string>();
  for (const bundle of bundles.values()) {
    for (const symbolId of bundle.changedSymbolIds) {
      const previous = symbolOwner.get(symbolId);
      if (previous && previous !== bundle.bundleId) {
        throw new Error(
          `Semantic integration conflict: Code Symbol ${symbolId} changed by ${previous} and ${bundle.bundleId}`,
        );
      }
      symbolOwner.set(symbolId, bundle.bundleId);
    }
  }

  const dependencies = integrationDependencies(Array.from(workItems.values()));
  const orderedWorkItemIds = topologicalOrder(workItems, dependencies);
  return {
    taskId: input.taskId,
    orderedChangeBundles: orderedWorkItemIds.map((workItemId) => requireValue(bundles, workItemId)),
    changedPaths: uniqueSorted(
      input.changeBundles.flatMap((bundle) => bundle.changedPaths).map(canonicalPath),
    ),
    changedSymbolIds: uniqueSorted(
      input.changeBundles.flatMap((bundle) => bundle.changedSymbolIds),
    ),
    authorProviders: uniqueSorted(
      input.authors.map((author) => author.provider),
    ) as ChangeBundleIntegrationPlan["authorProviders"],
  };
}

function integrationDependencies(
  workItems: readonly LogicWorkItem[],
): ReadonlyMap<string, ReadonlySet<string>> {
  const ordered = [...workItems].sort((left, right) =>
    left.workItemId.localeCompare(right.workItemId),
  );
  const dependencies = new Map(
    ordered.map((workItem) => [workItem.workItemId, new Set(workItem.dependsOn)] as const),
  );
  const known = new Set(dependencies.keys());
  for (const workItem of ordered) {
    for (const dependency of workItem.dependsOn) {
      if (!known.has(dependency)) {
        throw new Error(`Logic Work Item ${workItem.workItemId} depends on unknown ${dependency}`);
      }
    }
  }
  for (let leftIndex = 0; leftIndex < ordered.length; leftIndex++) {
    const left = ordered[leftIndex];
    if (!left) continue;
    for (let rightIndex = leftIndex + 1; rightIndex < ordered.length; rightIndex++) {
      const right = ordered[rightIndex];
      if (!right) continue;
      const leftPaths = new Set(left.allowedPaths.map(canonicalPath));
      if (right.allowedPaths.map(canonicalPath).some((value) => leftPaths.has(value))) {
        dependencies.get(right.workItemId)?.add(left.workItemId);
      }
    }
  }
  return dependencies;
}

function topologicalOrder(
  workItems: ReadonlyMap<string, LogicWorkItem>,
  dependencies: ReadonlyMap<string, ReadonlySet<string>>,
): readonly string[] {
  const pending = new Set(workItems.keys());
  const completed = new Set<string>();
  const ordered: string[] = [];
  while (pending.size > 0) {
    const ready = Array.from(pending)
      .filter((workItemId) =>
        Array.from(dependencies.get(workItemId) ?? []).every((dependency) =>
          completed.has(dependency),
        ),
      )
      .sort();
    if (ready.length === 0) {
      throw new Error("Logic Work Item dependency graph contains a cycle");
    }
    for (const workItemId of ready) {
      pending.delete(workItemId);
      completed.add(workItemId);
      ordered.push(workItemId);
    }
  }
  return ordered;
}

function uniqueBy<T>(
  values: readonly T[],
  key: (value: T) => string,
  label: string,
): ReadonlyMap<string, T> {
  const result = new Map<string, T>();
  for (const value of values) {
    const itemKey = key(value);
    if (result.has(itemKey)) throw new Error(`Duplicate ${label}: ${itemKey}`);
    result.set(itemKey, value);
  }
  return result;
}

function requireValue<T>(values: ReadonlyMap<string, T>, key: string): T {
  const value = values.get(key);
  if (!value) throw new Error(`Missing integration value ${key}`);
  return value;
}

function canonicalPath(value: string): string {
  return value.replaceAll("\\", "/").replace(/^\.\//, "");
}

function uniqueSorted(values: readonly string[]): string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}
