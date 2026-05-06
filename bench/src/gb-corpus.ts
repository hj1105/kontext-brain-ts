/**
 * GraphRAG-Bench corpus + question loader.
 *
 * Source: https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench
 * Mirror used: GitHub raw at GraphRAG-Bench/GraphRAG-Benchmark.
 *
 * Two domains: medical (one big corpus) + novel (20 separate novels).
 * Questions tagged by question_type: Fact Retrieval / Complex Reasoning /
 * Contextual Summarize / Creative Generation. We focus on Fact Retrieval
 * for tractable scoring (ACC + ROUGE-L).
 */
import { readFileSync } from "node:fs";
import type { BenchDoc, BenchQuery } from "./corpus.js";

interface GBQuestion {
  id: string;
  source: string;
  question: string;
  answer: string;
  question_type: string;
  evidence: string;
  evidence_relations?: string;
  evidence_triple?: string;
}

interface GBMedicalCorpus {
  corpus_name: string;
  context: string;
}

interface GBNovelCorpus {
  corpus_name: string;
  context: string;
}

export interface GBSample {
  domain: "medical" | "novel";
  docs: BenchDoc[];
  queries: BenchQuery[];
  evidenceById: Map<string, string>;
}

/** Chunk a long text into ~chunkChars-sized passages with overlap. Splits
 * on paragraph boundaries when available, falls back to sentence splits,
 * then character splits for very long single paragraphs. */
function chunkText(text: string, chunkChars: number, idPrefix: string): BenchDoc[] {
  // First split into rough segments — paragraphs OR sentences if no para breaks
  const hasParaBreaks = /\n\n|\n\s*\n/.test(text);
  const segments = hasParaBreaks
    ? text.split(/\n+/).map((p) => p.trim()).filter((p) => p.length > 0)
    : text.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter((s) => s.length > 0);

  // Force-split any segment longer than chunkChars * 1.5 by character window
  const splitLong: string[] = [];
  for (const s of segments) {
    if (s.length <= chunkChars * 1.5) {
      splitLong.push(s);
    } else {
      for (let off = 0; off < s.length; off += chunkChars) {
        splitLong.push(s.slice(off, off + chunkChars));
      }
    }
  }

  // Pack into chunks with light overlap (last 200 chars of prior buf carry over)
  const chunks: BenchDoc[] = [];
  let buf = "";
  let i = 0;
  for (const seg of splitLong) {
    if (buf.length + seg.length + 1 > chunkChars && buf.length > 0) {
      chunks.push({ id: `${idPrefix}-${i}`, title: `${idPrefix} chunk ${i}`, body: buf });
      i++;
      buf = buf.slice(-200); // overlap
    }
    buf = buf ? `${buf} ${seg}` : seg;
  }
  if (buf.length > 0) {
    chunks.push({ id: `${idPrefix}-${i}`, title: `${idPrefix} chunk ${i}`, body: buf });
  }
  return chunks;
}

function mulberry32(seed: number): () => number {
  let t = seed;
  return () => {
    t = (t + 0x6d2b79f5) | 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function loadMedicalSample(
  corpusPath: string,
  questionsPath: string,
  numQuestions: number,
  questionType = "Fact Retrieval",
  chunkChars = 1000,
  seed = 42,
): GBSample {
  const corpus = JSON.parse(readFileSync(corpusPath, "utf-8")) as GBMedicalCorpus;
  const questions = JSON.parse(readFileSync(questionsPath, "utf-8")) as GBQuestion[];

  const docs = chunkText(corpus.context, chunkChars, "med");
  const filtered = questions.filter((q) => q.question_type === questionType && q.evidence);
  const rng = mulberry32(seed);
  const shuffled = [...filtered].sort(() => rng() - 0.5);
  const sampled = shuffled.slice(0, numQuestions);

  const queries: BenchQuery[] = sampled.map((q) => ({
    id: q.id,
    question: q.question,
    expectedDocIds: [],
    expectedKeywords: [q.answer],
  }));
  const evidenceById = new Map(sampled.map((q) => [q.id, q.evidence]));

  return { domain: "medical", docs, queries, evidenceById };
}

export function loadNovelSample(
  corpusPath: string,
  questionsPath: string,
  numQuestions: number,
  questionType = "Fact Retrieval",
  chunkChars = 1000,
  seed = 42,
): GBSample {
  const corpora = JSON.parse(readFileSync(corpusPath, "utf-8")) as GBNovelCorpus[];
  const questions = JSON.parse(readFileSync(questionsPath, "utf-8")) as GBQuestion[];

  // Filter to fact retrieval first; sample N
  const filtered = questions.filter((q) => q.question_type === questionType && q.evidence);
  const rng = mulberry32(seed);
  const shuffled = [...filtered].sort(() => rng() - 0.5);
  const sampled = shuffled.slice(0, numQuestions);

  // Build corpus from the novels referenced by sampled questions (keeps
  // corpus relevant; full 20 novels = 5MB which is too much to embed)
  const sourceNovels = new Set(sampled.map((q) => q.source));
  const docs: BenchDoc[] = [];
  for (const corp of corpora) {
    if (!sourceNovels.has(corp.corpus_name)) continue;
    docs.push(...chunkText(corp.context, chunkChars, corp.corpus_name));
  }

  const queries: BenchQuery[] = sampled.map((q) => ({
    id: q.id,
    question: q.question,
    expectedDocIds: [],
    expectedKeywords: [q.answer],
  }));
  const evidenceById = new Map(sampled.map((q) => [q.id, q.evidence]));

  return { domain: "novel", docs, queries, evidenceById };
}
