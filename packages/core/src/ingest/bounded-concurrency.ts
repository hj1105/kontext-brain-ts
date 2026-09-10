/**
 * Runs `work` over `items` with at most `limit` calls in flight, keeping result
 * order. Model calls that fan out over a large corpus must be bounded: an
 * unbounded Promise.all over thousands of batches starts thousands of CLI
 * processes at once, and every one of them then misses its own time budget.
 */
export async function mapWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  work: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, async () => {
    while (next < items.length) {
      const index = next;
      next += 1;
      results[index] = await work(items[index] as T, index);
    }
  });
  await Promise.all(workers);
  return results;
}

/**
 * Up to `limit` items taken round-robin across `groupOf(item)`, so a sample
 * over many sources represents each of them instead of the largest one.
 */
export function stratifiedSample<T>(
  items: readonly T[],
  limit: number,
  groupOf: (item: T) => string,
): T[] {
  if (items.length <= limit) return [...items];
  const groups = new Map<string, T[]>();
  for (const item of items) {
    const key = groupOf(item);
    const group = groups.get(key);
    if (group) group.push(item);
    else groups.set(key, [item]);
  }
  const queues = Array.from(groups.values());
  const sample: T[] = [];
  let position = 0;
  while (sample.length < limit) {
    let took = false;
    for (const queue of queues) {
      const item = queue[position];
      if (item === undefined) continue;
      sample.push(item);
      took = true;
      if (sample.length >= limit) break;
    }
    if (!took) break;
    position += 1;
  }
  return sample;
}
