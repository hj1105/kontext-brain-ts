/**
 * SQuAD 2.0 → bench corpus adapter.
 *
 * SQuAD is a reading-comprehension dataset: each question is paired with a
 * Wikipedia paragraph that contains (or does not contain) the answer.
 *
 * To turn it into a RAG benchmark:
 *   - each paragraph becomes a document in the corpus
 *   - each question becomes a bench query
 *   - expectedDocIds = [the paragraph id this question was authored against]
 *   - expectedKeywords = [the answer text itself, first answer]
 *
 * Because every question has exactly one "right" paragraph, this is a
 * pure-retrieval test. Questions marked `is_impossible: true` are skipped
 * (they have no valid answer and would distort keyword-hit scoring).
 */

import { readFileSync } from "node:fs";
import type { BenchDoc, BenchQuery } from "./corpus.js";

interface SquadAnswer { text: string; answer_start: number }
interface SquadQa { question: string; id: string; answers: SquadAnswer[]; is_impossible: boolean }
interface SquadParagraph { qas: SquadQa[]; context: string }
interface SquadArticle { title: string; paragraphs: SquadParagraph[] }
interface SquadData { version: string; data: SquadArticle[] }

export interface SquadBench {
  docs: BenchDoc[];
  queries: BenchQuery[];
}

/**
 * Load a sample from SQuAD 2.0 dev set.
 *
 * @param path filesystem path to the downloaded SQuAD JSON file
 * @param numQuestions how many questions to include (also caps docs indirectly)
 * @param seed deterministic sampling seed
 */
export function loadSquadSample(
  path: string,
  numQuestions = 50,
  seed = 42,
): SquadBench {
  const raw = readFileSync(path, "utf-8");
  const data = JSON.parse(raw) as SquadData;

  // Flatten all (paragraph, qa) pairs that have valid answers
  type Pair = { docId: string; title: string; context: string; qa: SquadQa };
  const pairs: Pair[] = [];

  for (const article of data.data) {
    for (let pIdx = 0; pIdx < article.paragraphs.length; pIdx++) {
      const paragraph = article.paragraphs[pIdx]!;
      const docId = `${slug(article.title)}-${pIdx}`;
      for (const qa of paragraph.qas) {
        if (qa.is_impossible) continue;
        if (qa.answers.length === 0) continue;
        pairs.push({ docId, title: article.title, context: paragraph.context, qa });
      }
    }
  }

  // Deterministic sample
  const rng = mulberry32(seed);
  const shuffled = [...pairs].sort(() => rng() - 0.5);
  const selected = shuffled.slice(0, numQuestions);

  // Build corpus from all selected paragraphs (+ some distractors from other article paragraphs)
  const docMap = new Map<string, BenchDoc>();
  for (const pair of selected) {
    if (!docMap.has(pair.docId)) {
      docMap.set(pair.docId, {
        id: pair.docId,
        title: `${pair.title} (paragraph ${pair.docId.split("-").pop()})`,
        body: pair.context,
      });
    }
  }

  // Add distractor paragraphs from the same articles we sampled from, so
  // retrieval is non-trivial (otherwise corpus size equals query count).
  const selectedTitles = new Set(selected.map((p) => p.title));
  for (const article of data.data) {
    if (!selectedTitles.has(article.title)) continue;
    for (let pIdx = 0; pIdx < Math.min(article.paragraphs.length, 3); pIdx++) {
      const paragraph = article.paragraphs[pIdx]!;
      const docId = `${slug(article.title)}-${pIdx}`;
      if (!docMap.has(docId)) {
        docMap.set(docId, {
          id: docId,
          title: `${article.title} (paragraph ${pIdx})`,
          body: paragraph.context,
        });
      }
    }
  }

  const docs = Array.from(docMap.values());
  const queries: BenchQuery[] = selected.map((pair, i) => ({
    id: `sq-${i}`,
    question: pair.qa.question,
    expectedDocIds: [pair.docId],
    expectedKeywords: [pair.qa.answers[0]!.text],
  }));

  return { docs, queries };
}

function slug(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 40);
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
