/**
 * HotpotQA → bench corpus adapter.
 *
 * HotpotQA is a *multi-hop* QA dataset: each question requires reasoning over
 * TWO Wikipedia paragraphs to answer. The "distractor" setting gives the
 * system 10 candidate paragraphs per question (2 gold + 8 distractors).
 *
 * This is a harder test than SQuAD for a RAG system because:
 *   1. Retrieval must surface BOTH gold paragraphs, not just one
 *   2. The answer often does not appear verbatim in either paragraph
 *      (requires composition: "bridge" or "comparison" reasoning)
 *
 * Adapter design:
 *   - Each paragraph (flattened across questions) is a BenchDoc
 *   - Each question's `supporting_facts` gives the titles of the 2 gold
 *     paragraphs — those become `expectedDocIds`
 *   - `expectedKeywords = [answer]`
 *   - We sample a small question set deterministically
 */
import { readFileSync } from "node:fs";
import type { BenchDoc, BenchQuery } from "./corpus.js";

interface HotpotSupportingFact {
  0: string; // title
  1: number; // sentence idx
}

interface HotpotQuestion {
  _id: string;
  question: string;
  answer: string;
  type: "comparison" | "bridge";
  level: "easy" | "medium" | "hard";
  supporting_facts: [string, number][];
  context: [string, string[]][]; // [title, [sentence, ...]]
}

export interface HotpotBench {
  docs: BenchDoc[];
  queries: BenchQuery[];
}

export function loadHotpotSample(
  path: string,
  numQuestions = 30,
  seed = 42,
): HotpotBench {
  const raw = readFileSync(path, "utf-8");
  const data = JSON.parse(raw) as HotpotQuestion[];

  // Deterministic shuffle + pick
  const rng = mulberry32(seed);
  const shuffled = [...data].sort(() => rng() - 0.5);
  const selected = shuffled.slice(0, numQuestions);

  // Flatten unique (title) → doc. HotpotQA uses title as the paragraph id.
  const docMap = new Map<string, BenchDoc>();
  for (const q of selected) {
    for (const [title, sentences] of q.context) {
      const docId = slug(title);
      if (!docMap.has(docId)) {
        docMap.set(docId, {
          id: docId,
          title,
          body: sentences.join(" ").trim(),
        });
      }
    }
  }

  const queries: BenchQuery[] = selected.map((q, i) => {
    const supportTitles = Array.from(
      new Set(q.supporting_facts.map((sf) => slug(sf[0]))),
    );
    return {
      id: `hp-${i}`,
      question: q.question,
      expectedDocIds: supportTitles,
      expectedKeywords: [q.answer],
    };
  });

  return { docs: Array.from(docMap.values()), queries };
}

function slug(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 60);
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
