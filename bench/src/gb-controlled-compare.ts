/**
 * Clean controlled comparison: kontext-brain retrievers vs vanilla RAG on
 * GraphRAG-Bench Fact Retrieval. Removes confounders by:
 *
 *   1. Same corpus, same chunker, same embedder, same questions for all 3.
 *   2. Multiple retrieval-only metrics that DON'T involve the LLM:
 *      - evidence-token coverage (token recall of gold evidence)
 *      - evidence-substring presence (is the gold sentence verbatim in
 *        any retrieved chunk?)
 *      - top-1 evidence containment (does the #1 retrieved chunk hold the
 *        evidence sentence?)
 *   3. For end-to-end (multi-hop only since we have answers), apply a
 *      STRICT protocol that rejects "general knowledge fallback" answers.
 *
 * Output: console-formatted table for direct paste into the report.
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
  "which","who","whom","when","where","what","how","why",
]);

function tokenize(s: string): string[] {
  return s.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length >= 4 && !STOP.has(t));
}

function tokenCoverage(text: string, ref: string): number {
  const tokens = tokenize(ref);
  if (tokens.length === 0) return 0;
  const lc = text.toLowerCase();
  return tokens.filter((t) => lc.includes(t)).length / tokens.length;
}

/** Does the first 60 chars of evidence appear (case-insensitive) in text? */
function evidenceSubstring(text: string, evidence: string): boolean {
  if (!evidence) return false;
  const ev = evidence.trim().slice(0, 60).toLowerCase();
  return text.toLowerCase().includes(ev);
}

interface ContextEntry {
  id: string;
  question: string;
  referenceAnswer: string;
  evidence: string;
  retrievedDocIds: string[];
  evidenceCoverage: number;
  context: string;
}

interface AnswerEntry { id: string; answer: string; }

interface RetrievalMetrics {
  N: number;
  avgTokenCov: number;
  fullCovCount: number; // ≥0.7 token coverage
  evidenceSubstrCount: number;
  top1ChunkCovCount: number; // ≥0.7 cov in just the #1 retrieved chunk
}

function computeRetrievalMetrics(entries: ContextEntry[]): RetrievalMetrics {
  const N = entries.length;
  let tcSum = 0, full = 0, subStr = 0, top1Hit = 0;
  for (const e of entries) {
    tcSum += e.evidenceCoverage;
    if (e.evidenceCoverage >= 0.7) full++;
    if (evidenceSubstring(e.context, e.evidence)) subStr++;
    // top-1 chunk coverage — take first ### block
    const firstChunk = e.context.split(/\n### /)[0];
    if (tokenCoverage(firstChunk, e.evidence) >= 0.7) top1Hit++;
  }
  return {
    N,
    avgTokenCov: tcSum / N,
    fullCovCount: full,
    evidenceSubstrCount: subStr,
    top1ChunkCovCount: top1Hit,
  };
}

/** Strict end-to-end ACC: token recall of gold answer in our answer,
 * BUT mark answers containing fallback keywords as automatically wrong. */
function strictEndToEnd(
  ctxPath: string,
  ansPath: string,
): { acc: number; rouge: number; correctCount: number; N: number; fallbackRejected: number } {
  if (!existsSync(ctxPath) || !existsSync(ansPath)) {
    return { acc: 0, rouge: 0, correctCount: 0, N: 0, fallbackRejected: 0 };
  }
  const ctx = JSON.parse(readFileSync(ctxPath, "utf-8")) as ContextEntry[];
  const ans = new Map(
    (JSON.parse(readFileSync(ansPath, "utf-8")) as AnswerEntry[]).map((a) => [a.id, a.answer]),
  );

  const fallbackPatterns = [
    /not in retrieved context/i,
    /from general knowledge/i,
    /retrieval failed/i,
    /not in context/i,
    /retrieval did not surface/i,
  ];

  let accSum = 0, rougeSum = 0, correct = 0, fallback = 0;
  for (const c of ctx) {
    const a = ans.get(c.id) ?? "";
    const isFallback = fallbackPatterns.some((p) => p.test(a));
    if (isFallback) {
      fallback++;
      continue; // strict: 0 credit
    }
    const cov = tokenCoverage(a, c.referenceAnswer);
    accSum += cov;
    if (cov >= 0.5) correct++;

    // ROUGE-L
    const A = a.toLowerCase().split(/\s+/).filter(Boolean);
    const G = c.referenceAnswer.toLowerCase().split(/\s+/).filter(Boolean);
    if (A.length && G.length) {
      const dp: number[][] = Array.from({ length: A.length + 1 }, () => new Array(G.length + 1).fill(0));
      for (let i = 1; i <= A.length; i++)
        for (let j = 1; j <= G.length; j++)
          dp[i]![j] = A[i - 1] === G[j - 1] ? dp[i - 1]![j - 1]! + 1 : Math.max(dp[i - 1]![j]!, dp[i]![j - 1]!);
      const lcs = dp[A.length]![G.length]!;
      if (lcs > 0) {
        const p = lcs / A.length, r = lcs / G.length;
        rougeSum += (2 * p * r) / (p + r);
      }
    }
  }
  return {
    acc: accSum / ctx.length,
    rouge: rougeSum / ctx.length,
    correctCount: correct,
    N: ctx.length,
    fallbackRejected: fallback,
  };
}

function main(): void {
  const dataDir = resolve(fileURLToPath(import.meta.url), "../../data");
  const srcDir = resolve(dataDir, "../src");

  console.log("\n========================================================");
  console.log("CONTROLLED COMPARISON — GraphRAG-Bench Fact Retrieval");
  console.log("All retrievers: same corpus, same chunker, same embedder,");
  console.log("same N=30 questions per domain.");
  console.log("========================================================\n");

  for (const dom of ["medical", "novel"]) {
    console.log(`### ${dom.toUpperCase()} (N=30) ###`);
    console.log("retriever  | tokenCov | ≥0.7-cov | ev-substr | top1-hit");
    console.log("-----------|----------|----------|-----------|----------");
    for (const r of ["vanilla", "hybrid", "multihop"]) {
      const ctxPath = `${srcDir}/claude-gb-${dom}-${r}-contexts.json`;
      if (!existsSync(ctxPath)) continue;
      const ctx = JSON.parse(readFileSync(ctxPath, "utf-8")) as ContextEntry[];
      const m = computeRetrievalMetrics(ctx);
      console.log(
        `${r.padEnd(10)} | ${m.avgTokenCov.toFixed(3)}   |   ${m.fullCovCount}/${m.N}    |   ${m.evidenceSubstrCount}/${m.N}     |   ${m.top1ChunkCovCount}/${m.N}`,
      );
    }
    console.log();
  }

  console.log("========================================================");
  console.log("STRICT END-TO-END (multi-hop only, fallback answers rejected)");
  console.log("========================================================\n");
  for (const dom of ["medical", "novel"]) {
    const r = strictEndToEnd(
      `${srcDir}/claude-gb-${dom}-multihop-contexts.json`,
      `${srcDir}/claude-gb-${dom}-multihop-answers.json`,
    );
    console.log(`${dom.padEnd(10)} N=${r.N}`);
    console.log(`  fallback-rejected: ${r.fallbackRejected}/${r.N}`);
    console.log(`  strict ACC (token recall): ${(r.acc * 100).toFixed(2)}%`);
    console.log(`  strict ROUGE-L:            ${(r.rouge * 100).toFixed(2)}%`);
    console.log(`  ≥0.5 ACC count:            ${r.correctCount}/${r.N}`);
    console.log();
  }
}

main();
