/**
 * Run baseline + 4 kontext variants on the same corpus + queries.
 */

import { CORPUS, QUERIES, type BenchQuery } from "./corpus.js";
import { BaselineRAG } from "./baseline.js";
import {
  FlatKeywordControl,
  KontextV1,
  KontextV1Fixed,
  KontextV2Keyword,
  KontextV3Vector,
  KontextV4LLM,
  KontextV5Compressed,
  KontextV6Hybrid,
  KontextV7HyDE,
  KontextV8QueryExpand,
  KontextV9Rerank,
  KontextV10RerankCompress,
  KontextV11HybridHeavy,
} from "./kontext-runner.js";

interface Row {
  queryId: string;
  system: string;
  retrievedDocIds: string[];
  expectedDocIds: string[];
  recall: number;
  keywordHit: number;
  contextChars: number;
  latencyMs: number;
  answer: string;
}

function recall(expected: string[], retrieved: string[]): number {
  if (expected.length === 0) return 1;
  const ret = new Set(retrieved);
  const hit = expected.filter((id) => ret.has(id)).length;
  return hit / expected.length;
}

function keywordHit(answer: string, keywords: string[]): number {
  if (keywords.length === 0) return 1;
  const lc = answer.toLowerCase();
  const hit = keywords.filter((k) => lc.includes(k.toLowerCase())).length;
  return hit / keywords.length;
}

async function runOne(
  system: string,
  run: (q: string) => Promise<{ answer: string; retrievedDocIds: string[]; contextChars: number }>,
  q: BenchQuery,
): Promise<Row> {
  const start = Date.now();
  const res = await run(q.question);
  const latencyMs = Date.now() - start;
  return {
    queryId: q.id,
    system,
    retrievedDocIds: res.retrievedDocIds,
    expectedDocIds: q.expectedDocIds,
    recall: recall(q.expectedDocIds, res.retrievedDocIds),
    keywordHit: keywordHit(res.answer, q.expectedKeywords),
    contextChars: res.contextChars,
    latencyMs,
    answer: res.answer,
  };
}

function avg(xs: number[]): number {
  return xs.length === 0 ? 0 : xs.reduce((s, x) => s + x, 0) / xs.length;
}

async function main(): Promise<void> {
  console.log("=== kontext-brain benchmark (v2 — 5 systems) ===");
  console.log(`corpus: ${CORPUS.length} docs, queries: ${QUERIES.length}`);
  console.log("LLM: qwen2.5:1.5b via Ollama (CPU), embedding: nomic-embed-text\n");

  const baseline = new BaselineRAG();
  const flat = new FlatKeywordControl();
  const v1 = new KontextV1();
  const v1f = new KontextV1Fixed();
  const v2 = new KontextV2Keyword();
  const v3 = new KontextV3Vector();
  const v4 = new KontextV4LLM();
  const v5 = new KontextV5Compressed();
  const v6 = new KontextV6Hybrid();
  const v7 = new KontextV7HyDE();
  const v8 = new KontextV8QueryExpand();
  const v9 = new KontextV9Rerank();
  const v10 = new KontextV10RerankCompress();
  const v11 = new KontextV11HybridHeavy();

  console.log("Indexing all systems...");
  const t0 = Date.now();
  await baseline.index(CORPUS);
  console.log(`  baseline       ${Date.now() - t0}ms`);
  let t = Date.now();
  await flat.index(CORPUS);
  console.log(`  flat-kw        ${Date.now() - t}ms`);
  for (const [name, sys] of [
    ["v1-default", v1],
    ["v1-fixed", v1f],
    ["v2-keyword", v2],
    ["v3-vector", v3],
    ["v4-llm", v4],
    ["v5-compress", v5],
    ["v6-hybrid", v6],
    ["v7-hyde", v7],
    ["v8-expand", v8],
    ["v9-rerank", v9],
    ["v10-rerank+c", v10],
    ["v11-hybrid+c", v11],
  ] as const) {
    t = Date.now();
    await sys.index(CORPUS);
    console.log(`  ${name.padEnd(14)} ${Date.now() - t}ms`);
  }

  const results: Row[] = [];
  for (const q of QUERIES) {
    console.log(`\n[${q.id}] ${q.question}`);
    const variants: Array<[string, (s: string) => Promise<RunResult>]> = [
      ["baseline", (s) => baseline.query(s)],
      ["flat-kw", (s) => flat.query(s)],
      ["v1-default", (s) => v1.query(s)],
      ["v1-fixed", (s) => v1f.query(s)],
      ["v2-keyword", (s) => v2.query(s)],
      ["v3-vector", (s) => v3.query(s)],
      ["v4-llm", (s) => v4.query(s)],
      ["v5-compress", (s) => v5.query(s)],
      ["v6-hybrid", (s) => v6.query(s)],
      ["v7-hyde", (s) => v7.query(s)],
      ["v8-expand", (s) => v8.query(s)],
      ["v9-rerank", (s) => v9.query(s)],
      ["v10-rerank+c", (s) => v10.query(s)],
      ["v11-hybrid+c", (s) => v11.query(s)],
    ];
    for (const [name, fn] of variants) {
      const r = await runOne(name, fn, q);
      console.log(
        `  ${name.padEnd(11)} recall=${r.recall.toFixed(2)} kw=${r.keywordHit.toFixed(2)} ctx=${r.contextChars}ch ${r.latencyMs}ms`,
      );
      results.push(r);
    }
  }

  console.log("\n=== Summary ===");
  console.log(
    "system           avgRecall  avgKeyword  avgContext  avgLatency",
  );
  const systems = [
    "baseline",
    "flat-kw",
    "v1-default",
    "v1-fixed",
    "v2-keyword",
    "v3-vector",
    "v4-llm",
    "v5-compress",
    "v6-hybrid",
    "v7-hyde",
    "v8-expand",
    "v9-rerank",
    "v10-rerank+c",
    "v11-hybrid+c",
  ];
  for (const sys of systems) {
    const rows = results.filter((r) => r.system === sys);
    console.log(
      `${sys.padEnd(17)}${avg(rows.map((r) => r.recall)).toFixed(3).padStart(8)}   ` +
        `${avg(rows.map((r) => r.keywordHit)).toFixed(3).padStart(8)}   ` +
        `${Math.round(avg(rows.map((r) => r.contextChars))).toString().padStart(6)}ch   ` +
        `${Math.round(avg(rows.map((r) => r.latencyMs))).toString().padStart(6)}ms`,
    );
  }

  const { writeFileSync } = await import("node:fs");
  writeFileSync(
    new URL("./results.json", import.meta.url),
    JSON.stringify(results, null, 2),
  );
  console.log("\nFull results written to bench/src/results.json");
}

interface RunResult {
  answer: string;
  retrievedDocIds: string[];
  contextChars: number;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
