/**
 * HotpotQA multi-hop QA bench. Runs baseline vector RAG + hybrid + ans-ensemble
 * against a HotpotQA distractor sample. Multi-hop is structurally harder than
 * SQuAD because retrieval must surface TWO gold paragraphs, not one.
 *
 * Outputs per-query results to bench/src/hotpot-results.json for manual
 * Claude-Code judgment.
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { loadHotpotSample } from "./hotpot-corpus.js";
import {
  SquadBaselineRAG,
  SquadKontextAnswerEnsemble,
  SquadKontextHybrid,
  SquadKontextBm25,
  buildHybridWithModel,
} from "./squad-runner.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

interface PerQueryResult {
  queryId: string;
  question: string;
  referenceAnswer: string;
  expectedDocIds: string[];
  systems: Record<string, {
    answer: string;
    retrievedDocIds: string[];
    contextChars: number;
    latencyMs: number;
    recall: number;
    keywordHit: number;
    bothGoldHit: number;
  }>;
}

function recall(expected: string[], retrieved: string[]): number {
  if (expected.length === 0) return 1;
  const r = new Set(retrieved);
  return expected.filter((id) => r.has(id)).length / expected.length;
}

/**
 * Multi-hop specific metric: did retrieval surface BOTH gold paragraphs?
 * (Key distinction from SQuAD where one doc is sufficient.)
 */
function bothGoldHit(expected: string[], retrieved: string[]): number {
  if (expected.length < 2) return recall(expected, retrieved);
  const r = new Set(retrieved);
  return expected.every((id) => r.has(id)) ? 1 : 0;
}

function keywordHit(answer: string, reference: string): number {
  const a = answer.toLowerCase();
  const ref = reference.toLowerCase();
  if (a.includes(ref)) return 1;
  const refTokens = ref.split(/\s+/).filter((t) => t.length > 2);
  if (refTokens.length === 0) return a.includes(ref) ? 1 : 0;
  const hits = refTokens.filter((t) => a.includes(t)).length;
  return hits / refTokens.length;
}

async function main(): Promise<void> {
  const NUM_QUERIES = Number(process.env.HOTPOT_N ?? 20);
  console.log(`=== HotpotQA multi-hop bench (N=${NUM_QUERIES}) ===`);

  const dataPath = resolve(__dirname, "../data/hotpot_dev.json");
  const { docs, queries } = loadHotpotSample(dataPath, NUM_QUERIES, 42);
  console.log(`corpus: ${docs.length} paragraphs, queries: ${queries.length}`);
  const multiHop = queries.filter((q) => q.expectedDocIds.length >= 2).length;
  console.log(`  multi-hop queries (need 2+ gold docs): ${multiHop}/${queries.length}`);

  // Core variants — keep the best 3 from SQuAD bench to avoid 10× latency blowup
  const baseline = new SquadBaselineRAG();
  const bm25 = new SquadKontextBm25();
  const hybrid = new SquadKontextHybrid();
  const hybridBig = buildHybridWithModel("qwen2.5:3b");
  const ansEnsemble = new SquadKontextAnswerEnsemble();

  console.log("\nIndexing...");
  let t = Date.now();
  await baseline.index(docs);
  console.log(`  baseline     ${Date.now() - t}ms`);
  t = Date.now();
  await bm25.index(docs);
  console.log(`  bm25-map     ${Date.now() - t}ms`);
  t = Date.now();
  await hybrid.index(docs);
  console.log(`  hybrid       ${Date.now() - t}ms`);
  t = Date.now();
  await hybridBig.index(docs);
  console.log(`  hybrid-3b    ${Date.now() - t}ms`);
  t = Date.now();
  await ansEnsemble.index(docs);
  console.log(`  ans-ensemble ${Date.now() - t}ms`);

  const systems: Array<[
    string,
    (q: string) => Promise<{ answer: string; retrievedDocIds: string[]; contextChars: number }>,
  ]> = [
    ["baseline", (q) => baseline.query(q)],
    ["bm25-map", (q) => bm25.query(q)],
    ["hybrid", (q) => hybrid.query(q)],
    ["hybrid-3b", (q) => hybridBig.query(q)],
    ["ans-ensemble", (q) => ansEnsemble.query(q)],
  ];

  const perQuery: PerQueryResult[] = [];
  for (const q of queries) {
    const row: PerQueryResult = {
      queryId: q.id,
      question: q.question,
      referenceAnswer: q.expectedKeywords[0] ?? "",
      expectedDocIds: q.expectedDocIds,
      systems: {},
    };
    console.log(`\n[${q.id}] ${q.question}`);
    console.log(`  (ref: ${q.expectedKeywords[0]} | gold docs: ${q.expectedDocIds.length})`);
    for (const [name, fn] of systems) {
      const start = performance.now();
      const res = await fn(q.question);
      const latencyMs = performance.now() - start;
      const r = recall(q.expectedDocIds, res.retrievedDocIds);
      const bg = bothGoldHit(q.expectedDocIds, res.retrievedDocIds);
      const kw = keywordHit(res.answer, q.expectedKeywords[0] ?? "");
      row.systems[name] = {
        answer: res.answer.slice(0, 400),
        retrievedDocIds: res.retrievedDocIds,
        contextChars: res.contextChars,
        latencyMs,
        recall: r,
        bothGoldHit: bg,
        keywordHit: kw,
      };
      console.log(
        `  ${name.padEnd(12)} r=${r.toFixed(2)} 2gold=${bg} kw=${kw.toFixed(2)} ctx=${res.contextChars} lat=${latencyMs.toFixed(0)}ms  ::: ${res.answer.slice(0, 70).replace(/\n/g, " ")}`,
      );
    }
    perQuery.push(row);
  }

  console.log("\n=== Aggregate ===\nsystem       recall  both-gold  keyword   ctx     latency");
  for (const [name] of systems) {
    const rs = perQuery.map((r) => r.systems[name]!);
    const avgR = rs.reduce((s, x) => s + x.recall, 0) / rs.length;
    const avgBG = rs.reduce((s, x) => s + x.bothGoldHit, 0) / rs.length;
    const avgKw = rs.reduce((s, x) => s + x.keywordHit, 0) / rs.length;
    const avgCtx = rs.reduce((s, x) => s + x.contextChars, 0) / rs.length;
    const avgLat = rs.reduce((s, x) => s + x.latencyMs, 0) / rs.length;
    console.log(
      `${name.padEnd(12)} ${avgR.toFixed(3)}   ${avgBG.toFixed(3)}      ${avgKw.toFixed(3)}   ${Math.round(avgCtx).toString().padStart(5)}ch  ${Math.round(avgLat).toString().padStart(6)}ms`,
    );
  }

  writeFileSync(
    new URL("./hotpot-results.json", import.meta.url),
    JSON.stringify(perQuery, null, 2),
  );
  console.log("\nDetailed per-query results written to bench/src/hotpot-results.json");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
