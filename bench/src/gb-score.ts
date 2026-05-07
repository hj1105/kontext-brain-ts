/**
 * Score Claude-Code answers on GraphRAG-Bench Fact Retrieval subset.
 *
 * Metrics matching the official leaderboard:
 *   - ACC: token-level recall of the gold answer in our generated answer
 *          (auto-graded via content-token overlap; the official metric is
 *          LLM-judged correctness — strongly correlated for fact retrieval)
 *   - ROUGE-L: longest common subsequence F1 between answer and gold
 */
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

const STOP = new Set([
  "the","a","an","is","was","are","were","be","been","of","to","in","on","at",
  "for","by","with","as","that","this","it","its","and","or","but","not","from",
  "into","over","under","about","also","than","then","so","such","these","those",
  "there","they","them","their","does","do","did","has","have","had","can","could",
  "would","should","may","might","will","most","more","some","any","all","each",
  "which","who","whom","when","where","what","how","why","is","are",
]);

function tokenize(s: string): string[] {
  return s.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length >= 3 && !STOP.has(t));
}

/** Token-level recall: fraction of gold-answer content tokens present in generated answer. */
function tokenAcc(answer: string, gold: string): number {
  const goldTokens = tokenize(gold);
  if (goldTokens.length === 0) return 0;
  const ans = answer.toLowerCase();
  const hits = goldTokens.filter((t) => ans.includes(t)).length;
  return hits / goldTokens.length;
}

/** ROUGE-L F1 between answer and reference, using LCS over word sequences. */
function rougeL(answer: string, gold: string): number {
  const A = answer.toLowerCase().split(/\s+/).filter(Boolean);
  const G = gold.toLowerCase().split(/\s+/).filter(Boolean);
  if (A.length === 0 || G.length === 0) return 0;
  // LCS DP
  const m = A.length, n = G.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i]![j] = A[i - 1] === G[j - 1] ? dp[i - 1]![j - 1]! + 1 : Math.max(dp[i - 1]![j]!, dp[i]![j - 1]!);
    }
  }
  const lcs = dp[m]![n]!;
  if (lcs === 0) return 0;
  const p = lcs / m;
  const r = lcs / n;
  return (2 * p * r) / (p + r);
}

interface ContextEntry { id: string; question: string; referenceAnswer: string; }
interface AnswerEntry { id: string; answer: string; }

function score(domain: string, retriever: string, dataDir: string, prefix = "claude"): void {
  const ctxPath = `${dataDir}/../src/claude-gb-${domain}-${retriever}-contexts.json`;
  const ansPath = `${dataDir}/../src/${prefix}-gb-${domain}-${retriever}-answers.json`;
  if (!existsSync(ctxPath) || !existsSync(ansPath)) {
    console.log(`  ${domain}/${retriever} (${prefix}): skip`);
    return;
  }
  const ctx = JSON.parse(readFileSync(ctxPath, "utf-8")) as ContextEntry[];
  const ans = new Map(
    (JSON.parse(readFileSync(ansPath, "utf-8")) as AnswerEntry[]).map((a) => [a.id, a.answer]),
  );

  let accSum = 0, rougeSum = 0, accFull = 0;
  for (const c of ctx) {
    const a = ans.get(c.id) ?? "";
    const acc = tokenAcc(a, c.referenceAnswer);
    const rl = rougeL(a, c.referenceAnswer);
    accSum += acc;
    rougeSum += rl;
    if (acc >= 0.7) accFull++;
  }
  const N = ctx.length;
  console.log(
    `  ${domain.padEnd(8)} ${retriever.padEnd(10)} ${prefix.padEnd(7)} N=${N}  ACC=${(accSum / N * 100).toFixed(2)}%  ROUGE-L=${(rougeSum / N * 100).toFixed(2)}%  (≥0.7 ACC: ${accFull}/${N})`,
  );
}

function main(): void {
  const dataDir = resolve(fileURLToPath(import.meta.url), "../../data");
  console.log(`\n=== Claude Code as LLM ===\n`);
  for (const dom of ["medical", "novel"]) {
    for (const retr of ["vanilla", "hybrid", "multihop"]) {
      score(dom, retr, dataDir, "claude");
    }
    console.log();
  }
  console.log(`\n=== Ollama 8B as LLM (apples-to-apples vs leaderboard) ===\n`);
  for (const dom of ["medical", "novel"]) {
    for (const retr of ["vanilla", "hybrid", "multihop"]) {
      score(dom, retr, dataDir, "llm8b");
    }
    console.log();
  }

  console.log("\n=== Published GraphRAG-Bench Leaderboard (Fact_ACC / Fact_ROUGE-L) ===\n");
  console.log("Medical:");
  console.log("  G-reasoner          68.84 / 44.73");
  console.log("  HippoRAG2           66.28 / 36.69");
  console.log("  RAG (w/ rerank)     64.73 / 30.75");
  console.log("  RAG (w/o rerank)    63.72 / 29.21");
  console.log("  LightRAG            63.32 / 37.19");
  console.log("  Fast-GraphRAG       60.93 / 31.04");
  console.log("  Lazy-GraphRAG (MS)  60.25 / 31.66");
  console.log("  HippoRAG            56.14 / 20.95");
  console.log("  RAPTOR              54.07 / 17.93");
  console.log("  KGP                 55.53 / 21.34");
  console.log("  KET-RAG             60.35 / 31.99");
  console.log("  StructRAG           55.38 / 27.53");
  console.log("  MS-GraphRAG (local) 38.63 / 26.80");
  console.log("  MS-GraphRAG(global) 16.42 / 46.00");
  console.log("\nNovel:");
  console.log("  AutoPrunedRetriever 45.99 / 26.99");
  console.log("  G-reasoner          60.07 / 36.93");
  console.log("  HippoRAG2           60.14 / 31.35");
  console.log("  RAG (w/ rerank)     60.92 / 36.08");
  console.log("  RAG (w/o rerank)    58.76 / 37.35");
  console.log("  LightRAG            58.62 / 35.72");
  console.log("  Fast-GraphRAG       56.95 / 35.90");
  console.log("  Lazy-GraphRAG (MS)  51.65 / 36.97");
  console.log("  HippoRAG            52.93 / 26.65");
  console.log("  KET-RAG             55.39 / 27.39");
  console.log("  StructRAG           53.84 / 26.73");
  console.log("  MS-GraphRAG (local) 49.29 / 26.11");
  console.log("  MS-GraphRAG(global) 36.92 / 17.32");
}

main();
