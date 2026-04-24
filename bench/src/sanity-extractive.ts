/**
 * Sanity check for v12/v16 sub-millisecond timings.
 *
 * Hypothesis 1 (user's): there's caching/JIT making subsequent runs unrealistically fast.
 * Hypothesis 2 (mine):    these variants legitimately do zero network/LLM I/O,
 *                         so they really are pure-JS in-memory ops.
 *
 * To distinguish: run each query 5x with fresh queries, and run cold instances.
 */

import { CORPUS, QUERIES } from "./corpus.js";
import { KontextV12Extractive, KontextV16Proximity } from "./kontext-runner.js";

async function timeQuery(name: string, fn: () => Promise<unknown>): Promise<number> {
  const start = performance.now();
  await fn();
  return performance.now() - start;
}

async function main(): Promise<void> {
  console.log("=== Sanity check: extractive timing ===\n");

  // Cold start: brand-new instance per measurement
  console.log("Cold-start (new instance per query):");
  for (const q of QUERIES) {
    const v12 = new KontextV12Extractive();
    await v12.index(CORPUS);  // index time NOT counted in query latency
    const t = await timeQuery("v12", () => v12.query(q.question));
    console.log(`  v12-extract ${q.id}: ${t.toFixed(3)}ms`);
  }

  console.log("\nCold-start v16:");
  for (const q of QUERIES) {
    const v16 = new KontextV16Proximity();
    await v16.index(CORPUS);
    const t = await timeQuery("v16", () => v16.query(q.question));
    console.log(`  v16-proximity ${q.id}: ${t.toFixed(3)}ms`);
  }

  // Warm: same instance, repeat each query 5x
  console.log("\nWarm (same instance, 5 repeats per query):");
  const v12Warm = new KontextV12Extractive();
  await v12Warm.index(CORPUS);
  for (const q of QUERIES.slice(0, 3)) {
    const times: number[] = [];
    for (let i = 0; i < 5; i++) {
      times.push(await timeQuery("v12", () => v12Warm.query(q.question)));
    }
    console.log(`  v12-extract ${q.id}: [${times.map((t) => t.toFixed(3)).join(", ")}]ms`);
  }

  // Brand-new query (never seen): random injection
  console.log("\nNovel query (never run before):");
  for (let i = 0; i < 5; i++) {
    const novelQ = `What about a totally unique query number ${Math.random().toString(36).slice(2)} ?`;
    const v12 = new KontextV12Extractive();
    await v12.index(CORPUS);
    const t = await timeQuery("v12", () => v12.query(novelQ));
    console.log(`  v12-extract novel: ${t.toFixed(3)}ms`);
  }

  // Compare with baseline operations: just an array find
  console.log("\nReference: pure JS in-memory ops:");
  for (let i = 0; i < 5; i++) {
    const t = await timeQuery("ref", async () => {
      const arr = CORPUS.map((d) => d.body.split(/[.!?]\s+/)).flat();
      const q = "rest api version".toLowerCase().split(" ");
      return arr.filter((s) => q.some((w) => s.toLowerCase().includes(w))).slice(0, 3);
    });
    console.log(`  pure-array-filter: ${t.toFixed(3)}ms`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
