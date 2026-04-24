/**
 * Ralph loop: fast iteration vs baseline. Runs only baseline + candidate
 * variants and reports efficiency ratio.
 *
 *   efficiency = recall * keyword_hit / (context_chars * latency_ms)
 *   ratio      = kontext_efficiency / baseline_efficiency
 *
 * Target: ratio >= 10
 */

import { CORPUS, QUERIES, type BenchQuery } from "./corpus.js";
import { BaselineRAG } from "./baseline.js";
import {
  KontextV10RerankCompress,
  KontextV12Extractive,
  KontextV13SentenceLevel,
  KontextV14Parallel,
  KontextV15Adaptive,
  KontextV16Proximity,
  KontextV17HybridExtract,
  KontextV18Bm25Map,
  KontextV19Mmr,
  KontextV20EdgeAware,
  KontextV21Centroid,
  KontextV22Compose,
  KontextV23Bm25Extractive,
  KontextV24Entity,
  KontextV25EntityLLM,
} from "./kontext-runner.js";

interface Row {
  system: string;
  recall: number;
  keywordHit: number;
  contextChars: number;
  latencyMs: number;
}

function recall(expected: string[], retrieved: string[]): number {
  if (expected.length === 0) return 1;
  const ret = new Set(retrieved);
  return expected.filter((id) => ret.has(id)).length / expected.length;
}

function kwHit(answer: string, keywords: string[]): number {
  if (keywords.length === 0) return 1;
  const lc = answer.toLowerCase();
  return keywords.filter((k) => lc.includes(k.toLowerCase())).length / keywords.length;
}

async function runAll(
  name: string,
  run: (q: string) => Promise<{ answer: string; retrievedDocIds: string[]; contextChars: number }>,
): Promise<Row> {
  let recallSum = 0;
  let kwSum = 0;
  let ctxSum = 0;
  let latSum = 0;
  for (const q of QUERIES) {
    // Use performance.now() for sub-ms precision; extractive variants can
    // finish in microseconds.
    const start = performance.now();
    const res = await run(q.question);
    const latencyMs = performance.now() - start;
    recallSum += recall(q.expectedDocIds, res.retrievedDocIds);
    kwSum += kwHit(res.answer, q.expectedKeywords);
    ctxSum += res.contextChars;
    latSum += latencyMs;
    console.log(
      `  ${name} [${q.id}] r=${recall(q.expectedDocIds, res.retrievedDocIds).toFixed(2)} kw=${kwHit(res.answer, q.expectedKeywords).toFixed(2)} ctx=${res.contextChars} lat=${latencyMs.toFixed(1)}`,
    );
  }
  const n = QUERIES.length;
  return {
    system: name,
    recall: recallSum / n,
    keywordHit: kwSum / n,
    contextChars: ctxSum / n,
    latencyMs: latSum / n,
  };
}

function efficiency(r: Row): number {
  return (r.recall * r.keywordHit) / Math.max(r.contextChars * r.latencyMs, 1);
}

async function main(): Promise<void> {
  console.log("=== Ralph loop: 10x efficiency vs baseline ===\n");

  const baseline = new BaselineRAG();
  const v10 = new KontextV10RerankCompress();
  const v12 = new KontextV12Extractive();
  const v13 = new KontextV13SentenceLevel();
  const v14 = new KontextV14Parallel();
  const v15 = new KontextV15Adaptive();
  const v16 = new KontextV16Proximity();
  const v17 = new KontextV17HybridExtract();
  const v18 = new KontextV18Bm25Map();
  const v19 = new KontextV19Mmr();
  const v20 = new KontextV20EdgeAware();
  const v21 = new KontextV21Centroid();
  const v22 = new KontextV22Compose();
  const v23 = new KontextV23Bm25Extractive();
  const v24 = new KontextV24Entity();
  const v25 = new KontextV25EntityLLM();

  console.log("Indexing...");
  await baseline.index(CORPUS);
  await v10.index(CORPUS);
  await v12.index(CORPUS);
  await v13.index(CORPUS);
  await v14.index(CORPUS);
  await v15.index(CORPUS);
  await v16.index(CORPUS);
  await v17.index(CORPUS);
  await v18.index(CORPUS);
  await v19.index(CORPUS);
  await v20.index(CORPUS);
  await v21.index(CORPUS);
  await v22.index(CORPUS);
  await v23.index(CORPUS);
  await v24.index(CORPUS);
  await v25.index(CORPUS);

  const rows: Row[] = [];
  console.log("\n--- baseline ---");
  rows.push(await runAll("baseline", (q) => baseline.query(q)));
  console.log("\n--- v10 (previous best) ---");
  rows.push(await runAll("v10-rerank+c", (q) => v10.query(q)));
  console.log("\n--- v12 (extractive, no final LLM) ---");
  rows.push(await runAll("v12-extract", (q) => v12.query(q)));
  console.log("\n--- v13 (sentence-level + tiny LLM) ---");
  rows.push(await runAll("v13-sentence", (q) => v13.query(q)));
  console.log("\n--- v14 (parallel fetch) ---");
  rows.push(await runAll("v14-parallel", (q) => v14.query(q)));
  console.log("\n--- v15 (adaptive skip) ---");
  rows.push(await runAll("v15-adaptive", (q) => v15.query(q)));
  console.log("\n--- v16 (proximity extractive) ---");
  rows.push(await runAll("v16-proximity", (q) => v16.query(q)));
  console.log("\n--- v17 (hybrid extractive) ---");
  rows.push(await runAll("v17-hybrid-ex", (q) => v17.query(q)));
  console.log("\n--- v18 (BM25 node mapping) ---");
  rows.push(await runAll("v18-bm25-map", (q) => v18.query(q)));
  console.log("\n--- v19 (MMR diverse selection) ---");
  rows.push(await runAll("v19-mmr", (q) => v19.query(q)));
  console.log("\n--- v20 (edge-aware expansion) ---");
  rows.push(await runAll("v20-edge", (q) => v20.query(q)));
  console.log("\n--- v21 (centroid-refined embeddings) ---");
  rows.push(await runAll("v21-centroid", (q) => v21.query(q)));
  console.log("\n--- v22 (BM25+MMR+rerank+compress) ---");
  rows.push(await runAll("v22-compose", (q) => v22.query(q)));
  console.log("\n--- v23 (BM25 mapping + extractive) ---");
  rows.push(await runAll("v23-bm25-ex", (q) => v23.query(q)));
  console.log("\n--- v24 (entity-aware retrieval, extractive) ---");
  rows.push(await runAll("v24-entity", (q) => v24.query(q)));
  console.log("\n--- v25 (entity-aware + LLM answer) ---");
  rows.push(await runAll("v25-entity-llm", (q) => v25.query(q)));

  const baseRow = rows[0]!;
  const baseEff = efficiency(baseRow);

  console.log(
    "\n=== Summary ===\nsystem           recall  keyword    ctx    latency     eff       ratio",
  );
  for (const r of rows) {
    const eff = efficiency(r);
    const ratio = eff / baseEff;
    console.log(
      `${r.system.padEnd(17)}${r.recall.toFixed(3).padStart(6)}   ${r.keywordHit.toFixed(3).padStart(6)}  ${Math.round(r.contextChars).toString().padStart(5)}ch  ${Math.round(r.latencyMs).toString().padStart(5)}ms  ${eff.toExponential(2)}  ${ratio.toFixed(2)}x`,
    );
  }

  const best = rows
    .filter((r) => r.system !== "baseline")
    .sort((a, b) => efficiency(b) - efficiency(a))[0]!;
  const bestRatio = efficiency(best) / baseEff;
  console.log(
    `\nBest variant: ${best.system}  ratio=${bestRatio.toFixed(2)}x  ${bestRatio >= 10 ? "✓ TARGET REACHED" : "✗ need more work"}`,
  );

  const { writeFileSync } = await import("node:fs");
  writeFileSync(new URL("./ralph-results.json", import.meta.url), JSON.stringify(rows, null, 2));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
