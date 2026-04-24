/**
 * Research benchmark: SQuAD 2.0 sample.
 *
 * Runs baseline vector RAG + 4 kontext variants against a sampled SQuAD
 * corpus. Outputs detailed per-query results (question, reference answer,
 * each system's answer, retrieved doc ids, timing, context size) to
 * `squad-results.json` so Claude Code can judge answer quality line-by-line
 * without needing a paid Claude API call.
 */

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { loadSquadSample } from "./squad-corpus.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
import {
  SquadBaselineRAG,
  SquadKontextBm25,
  SquadKontextAnswerEnsemble,
  SquadKontextEnsemble,
  SquadKontextEntity,
  SquadKontextExtractive,
  SquadKontextExtractThenAnswer,
  SquadKontextHybrid,
  SquadKontextKeyword,
  SquadKontextSynthesis,
  SquadKontextVectorDocs,
  buildHybridWithModel,
} from "./squad-runner.js";

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
  }>;
}

function recall(expected: string[], retrieved: string[]): number {
  if (expected.length === 0) return 1;
  const r = new Set(retrieved);
  return expected.filter((id) => r.has(id)).length / expected.length;
}

function keywordHit(answer: string, reference: string): number {
  // Simple: does the answer contain the reference as a substring?
  // Claude Code will do the real quality scoring later.
  const a = answer.toLowerCase();
  const ref = reference.toLowerCase();
  if (a.includes(ref)) return 1;
  // Partial credit: fraction of reference tokens found
  const refTokens = ref.split(/\s+/).filter((t) => t.length > 2);
  if (refTokens.length === 0) return a.includes(ref) ? 1 : 0;
  const hits = refTokens.filter((t) => a.includes(t)).length;
  return hits / refTokens.length;
}

async function main(): Promise<void> {
  const NUM_QUERIES = Number(process.env.SQUAD_N ?? 30);

  console.log(`=== SQuAD 2.0 bench (N=${NUM_QUERIES}) ===`);
  const dataPath = resolve(__dirname, "../data/squad_dev.json");
  const { docs, queries } = loadSquadSample(dataPath, NUM_QUERIES, 42);
  console.log(`corpus: ${docs.length} paragraphs, queries: ${queries.length}`);

  const baseline = new SquadBaselineRAG();
  const kwMap = new SquadKontextKeyword();
  const bm25 = new SquadKontextBm25();
  const extract = new SquadKontextExtractive();
  const entity = new SquadKontextEntity();
  const hybrid = new SquadKontextHybrid();
  const vectorDocs = new SquadKontextVectorDocs();
  const ensemble = new SquadKontextEnsemble();
  const extractAnswer = new SquadKontextExtractThenAnswer();
  const hybridBig = buildHybridWithModel("qwen2.5:3b");
  const answerEnsemble = new SquadKontextAnswerEnsemble();
  const synthesis = new SquadKontextSynthesis();

  console.log("\nIndexing...");
  const t0 = Date.now();
  await baseline.index(docs);
  console.log(`  baseline     ${Date.now() - t0}ms`);
  let t = Date.now();
  await kwMap.index(docs);
  console.log(`  keyword-map  ${Date.now() - t}ms`);
  t = Date.now();
  await bm25.index(docs);
  console.log(`  bm25-map     ${Date.now() - t}ms`);
  t = Date.now();
  await extract.index(docs);
  console.log(`  extractive   ${Date.now() - t}ms`);
  t = Date.now();
  await entity.index(docs);
  console.log(`  entity       ${Date.now() - t}ms`);
  t = Date.now();
  await vectorDocs.index(docs);
  console.log(`  vector-docs  ${Date.now() - t}ms`);
  t = Date.now();
  await hybrid.index(docs);
  console.log(`  hybrid       ${Date.now() - t}ms`);
  t = Date.now();
  await ensemble.index(docs);
  console.log(`  ensemble     ${Date.now() - t}ms`);
  t = Date.now();
  await extractAnswer.index(docs);
  console.log(`  extract-ans  ${Date.now() - t}ms`);
  t = Date.now();
  await hybridBig.index(docs);
  console.log(`  hybrid-3b    ${Date.now() - t}ms`);
  t = Date.now();
  await answerEnsemble.index(docs);
  console.log(`  ans-ensemble ${Date.now() - t}ms`);
  t = Date.now();
  await synthesis.index(docs);
  console.log(`  synthesis    ${Date.now() - t}ms`);

  const systems: Array<[string, (q: string) => Promise<{ answer: string; retrievedDocIds: string[]; contextChars: number }>]> = [
    ["baseline", (q) => baseline.query(q)],
    ["kw-map", (q) => kwMap.query(q)],
    ["bm25-map", (q) => bm25.query(q)],
    ["extractive", (q) => extract.query(q)],
    ["entity", (q) => entity.query(q)],
    ["vector-docs", (q) => vectorDocs.query(q)],
    ["hybrid", (q) => hybrid.query(q)],
    ["ensemble", (q) => ensemble.query(q)],
    ["extract-ans", (q) => extractAnswer.query(q)],
    ["hybrid-3b", (q) => hybridBig.query(q)],
    ["ans-ensemble", (q) => answerEnsemble.query(q)],
    ["synthesis", (q) => synthesis.query(q)],
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
    console.log(`  (ref: ${q.expectedKeywords[0]})`);
    for (const [name, fn] of systems) {
      const start = performance.now();
      const res = await fn(q.question);
      const latencyMs = performance.now() - start;
      const r = recall(q.expectedDocIds, res.retrievedDocIds);
      const kw = keywordHit(res.answer, q.expectedKeywords[0] ?? "");
      row.systems[name] = {
        answer: res.answer.slice(0, 400),
        retrievedDocIds: res.retrievedDocIds,
        contextChars: res.contextChars,
        latencyMs,
        recall: r,
        keywordHit: kw,
      };
      console.log(
        `  ${name.padEnd(11)} r=${r.toFixed(2)} kw=${kw.toFixed(2)} ctx=${res.contextChars} lat=${latencyMs.toFixed(0)}ms  ::: ${res.answer.slice(0, 80).replace(/\n/g, " ")}`,
      );
    }
    perQuery.push(row);
  }

  // Aggregate
  console.log("\n=== Aggregate ===\nsystem       recall  keyword   ctx     latency");
  for (const [name] of systems) {
    const rs = perQuery.map((r) => r.systems[name]!);
    const avgR = rs.reduce((s, x) => s + x.recall, 0) / rs.length;
    const avgKw = rs.reduce((s, x) => s + x.keywordHit, 0) / rs.length;
    const avgCtx = rs.reduce((s, x) => s + x.contextChars, 0) / rs.length;
    const avgLat = rs.reduce((s, x) => s + x.latencyMs, 0) / rs.length;
    console.log(
      `${name.padEnd(12)} ${avgR.toFixed(3)}   ${avgKw.toFixed(3)}   ${Math.round(avgCtx).toString().padStart(5)}ch  ${Math.round(avgLat).toString().padStart(6)}ms`,
    );
  }

  writeFileSync(
    new URL("./squad-results.json", import.meta.url),
    JSON.stringify(perQuery, null, 2),
  );
  console.log("\nDetailed per-query results written to bench/src/squad-results.json");
  console.log("-> Claude Code can judge answer quality by reading that file.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
