/**
 * Build a real knowledge graph from the GraphRAG-Bench corpus (HippoRAG2-style)
 * and use it as a 4th retriever ('kg') alongside vanilla / hybrid / multi-hop.
 *
 * Pipeline:
 *   1. Chunk corpus (1024-char, matching leaderboard)
 *   2. Per-chunk LLM extraction of (entities, triples) → KG
 *   3. KG retriever uses personalized PageRank from query entities
 *   4. Dump retrieved contexts for downstream gb-llm-answer + gb-llm-judge
 *
 * Cache: KG saved to bench/data/gb-{domain}-kg.json — re-runs skip ingest.
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { loadMedicalSample, loadNovelSample, type GBSample } from "./gb-corpus.js";
import { BM25BodyExtractor } from "@kontext-brain/core";
import { buildKG, type KGStore } from "./kg-builder.js";
import { KGRetriever } from "./kg-retriever.js";

interface ContextEntry {
  id: string;
  question: string;
  referenceAnswer: string;
  evidence: string;
  retrievedDocIds: string[];
  matchedEntities: string[];
  evidenceCoverage: number;
  context: string;
}

const STOP = new Set([
  "the","a","an","is","was","are","were","be","been","of","to","in","on","at",
  "for","by","with","as","that","this","it","its","and","or","but","not","from",
  "into","over","under","about","also","than","then","so","such","these","those",
  "there","they","them","their","does","do","did","has","have","had","can","could",
  "would","should","may","might","will","most","more","some","any","all","each",
  "which","who","whom","when","where","what","how","why",
]);

function evidenceTokenCoverage(text: string, evidence: string): number {
  if (!evidence) return 0;
  const lc = text.toLowerCase();
  const tokens = evidence
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length >= 4 && !STOP.has(t));
  if (tokens.length === 0) return 0;
  return tokens.filter((t) => lc.includes(t)).length / tokens.length;
}

async function dumpKG(
  sample: GBSample,
  kg: KGStore,
  topK: number,
  outPath: string,
): Promise<void> {
  console.log(`\n[${sample.domain}/kg] queries=${sample.queries.length} entities=${kg.entities.size} edges=${kg.edges.length} chunks-in-kg=${kg.chunkToEntities.size}`);
  const docById = new Map(sample.docs.map((d) => [d.id, d]));
  const retriever = new KGRetriever(kg);

  const entries: ContextEntry[] = [];
  for (const q of sample.queries) {
    const ranked = retriever.retrieve(q.question, { topK });
    const retrievedIds: string[] = [];
    const parts: string[] = [];
    const allMatched = new Set<string>();
    for (const r of ranked) {
      const d = docById.get(r.chunkId);
      if (!d) continue;
      retrievedIds.push(r.chunkId);
      r.matchedEntities.forEach((e) => allMatched.add(e));
      parts.push(`### ${d.title}\n${BM25BodyExtractor.extract(d.body, q.question, 5)}`);
    }
    const evidence = sample.evidenceById.get(q.id) ?? "";
    const ctxJoined = parts.join("\n\n");
    entries.push({
      id: q.id,
      question: q.question,
      referenceAnswer: q.expectedKeywords[0] ?? "",
      evidence,
      retrievedDocIds: retrievedIds,
      matchedEntities: Array.from(allMatched),
      evidenceCoverage: evidenceTokenCoverage(ctxJoined, evidence),
      context: ctxJoined,
    });
  }
  const avg = entries.reduce((s, e) => s + e.evidenceCoverage, 0) / entries.length;
  const full = entries.filter((e) => e.evidenceCoverage >= 0.7).length;
  console.log(`  kg: avg evidence-token coverage ${avg.toFixed(3)} | ≥0.7 in ${full}/${entries.length}`);
  writeFileSync(outPath, JSON.stringify(entries, null, 2));
  console.log(`  → ${outPath}`);
}

async function main(): Promise<void> {
  const N = Number(process.env.GB_N ?? 30);
  const dataDir = resolve(fileURLToPath(import.meta.url), "../../data");
  const srcDir = resolve(dataDir, "../src");
  const model = process.env.KG_MODEL ?? "llama3.1:8b-instruct-q4_K_M";

  const medical = loadMedicalSample(
    `${dataDir}/gb-medical.json`,
    `${dataDir}/gb-medical-questions.json`,
    N,
    "Fact Retrieval",
    1024,
  );
  console.log(`\n=== Medical (${medical.queries.length} queries, ${medical.docs.length} chunks) ===`);
  console.log(`  building KG via ${model}...`);
  const medKG = await buildKG(
    model,
    medical.docs.map((d) => ({ id: d.id, body: d.body })),
    `${dataDir}/gb-medical-kg.json`,
    (i, t) => { if (i % 50 === 0) console.log(`    extracted ${i}/${t}`); },
  );
  await dumpKG(medical, medKG, 5, `${srcDir}/claude-gb-medical-kg-contexts.json`);

  const novel = loadNovelSample(
    `${dataDir}/gb-novel.json`,
    `${dataDir}/gb-novel-questions.json`,
    N,
    "Fact Retrieval",
    1024,
  );
  console.log(`\n=== Novel (${novel.queries.length} queries, ${novel.docs.length} chunks) ===`);
  console.log(`  building KG via ${model}...`);
  const novKG = await buildKG(
    model,
    novel.docs.map((d) => ({ id: d.id, body: d.body })),
    `${dataDir}/gb-novel-kg.json`,
    (i, t) => { if (i % 50 === 0) console.log(`    extracted ${i}/${t}`); },
  );
  await dumpKG(novel, novKG, 5, `${srcDir}/claude-gb-novel-kg-contexts.json`);
}

main().catch((e) => { console.error(e); process.exit(1); });
