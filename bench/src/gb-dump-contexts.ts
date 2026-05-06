/**
 * Dump retrieved contexts for GraphRAG-Bench (Fact Retrieval subset) using
 * three retrievers: vanilla vector RAG, kontext-brain hybrid, kontext-brain
 * multi-hop. Output JSONs are then answered by Claude Code and scored
 * against the gold answer + evidence.
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { loadMedicalSample, loadNovelSample, type GBSample } from "./gb-corpus.js";
import { BM25BodyExtractor } from "@kontext-brain/core";
import { LangChainVectorStore } from "@kontext-brain/llm";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MultiHopRetriever, extractEntities } from "./multihop-retriever.js";

interface ContextEntry {
  id: string;
  question: string;
  referenceAnswer: string;
  evidence: string;
  retrievedDocIds: string[];
  evidenceCoverage: number; // fraction of evidence tokens in retrieved context
  context: string;
}

const STOP = new Set([
  "the","a","an","is","was","are","were","be","been","being","of","to","in","on",
  "at","for","by","with","as","that","this","it","its","and","or","but","not",
  "from","into","over","under","about","after","before","when","where","what",
  "who","which","how","why","also","than","then","so","such","these","those",
  "there","they","them","their","does","do","did","has","have","had","can","could",
  "would","should","may","might","will","most","more","some","any","all","each",
]);

/**
 * Coverage = fraction of evidence content-tokens (4+ chars, non-stopword) that
 * appear in the joined retrieved context. Fairer than substring because the
 * corpus typically paraphrases the gold evidence sentence.
 */
function evidenceTokenCoverage(retrievedTexts: string[], evidence: string): number {
  if (!evidence) return 0;
  const ctx = retrievedTexts.join(" ").toLowerCase();
  const tokens = evidence
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length >= 4 && !STOP.has(t));
  if (tokens.length === 0) return 0;
  const hits = tokens.filter((t) => ctx.includes(t)).length;
  return hits / tokens.length;
}

async function buildVectorStore(sample: GBSample): Promise<LangChainVectorStore> {
  const baseUrl = "http://localhost:11434";
  const embeddings = new OllamaEmbeddings({ baseUrl, model: "nomic-embed-text" });
  const vectorStore = new LangChainVectorStore(embeddings);
  console.log(`  embedding ${sample.docs.length} chunks (shared across retrievers)...`);
  let n = 0;
  for (const doc of sample.docs) {
    const vec = await vectorStore.embed(`${doc.title}\n${doc.body.slice(0, 1500)}`);
    await vectorStore.upsert(`doc:${doc.id}`, vec, { docId: doc.id });
    n++;
    if (n % 50 === 0) console.log(`    indexed ${n}/${sample.docs.length}`);
  }
  return vectorStore;
}

async function indexAndQuery(
  sample: GBSample,
  retrieverName: "vanilla" | "hybrid" | "multihop",
  topK: number,
  vectorStore: LangChainVectorStore,
): Promise<ContextEntry[]> {
  console.log(`\n[${sample.domain}/${retrieverName}] corpus=${sample.docs.length} queries=${sample.queries.length} topK=${topK}`);

  const docById = new Map(sample.docs.map((d) => [d.id, d]));
  const retriever =
    retrieverName === "multihop"
      ? new MultiHopRetriever(
          sample.docs.map((d) => ({ id: d.id, title: d.title, body: d.body })),
          vectorStore,
        )
      : null;

  const entries: ContextEntry[] = [];
  for (const q of sample.queries) {
    let retrievedIds: string[] = [];
    let retrievedTexts: string[] = [];

    if (retrieverName === "vanilla") {
      const hits = await vectorStore.similaritySearchWithPrefix(q.question, "doc:", topK);
      for (const id of hits) {
        const d = docById.get(id);
        if (d) {
          retrievedIds.push(id);
          retrievedTexts.push(d.body);
        }
      }
    } else if (retrieverName === "multihop") {
      const ranked = await retriever!.retrieve(q.question, topK);
      for (const r of ranked) {
        retrievedIds.push(r.docId);
        retrievedTexts.push(r.body);
      }
    } else if (retrieverName === "hybrid") {
      // Hybrid: entity-BM25 (top-2 per entity) + vector (top-3) merged
      const entities = extractEntities(q.question);
      const scores = new Map<string, number>();
      const vecHits = await vectorStore.similaritySearchWithPrefix(q.question, "doc:", 5);
      for (let i = 0; i < vecHits.length; i++) {
        scores.set(vecHits[i]!, (scores.get(vecHits[i]!) ?? 0) + (1 - i / 5) * 0.6);
      }
      // For each entity, BM25-like scan: any doc whose body mentions the entity gets +1
      for (const ent of entities) {
        const lc = ent.toLowerCase();
        let cnt = 0;
        for (const d of sample.docs) {
          if (cnt >= 3) break;
          if (d.body.toLowerCase().includes(lc) || d.title.toLowerCase().includes(lc)) {
            scores.set(d.id, (scores.get(d.id) ?? 0) + 0.5);
            cnt++;
          }
        }
      }
      const ranked = Array.from(scores.entries()).sort((a, b) => b[1] - a[1]).slice(0, topK);
      for (const [id] of ranked) {
        const d = docById.get(id);
        if (d) {
          retrievedIds.push(id);
          retrievedTexts.push(d.body);
        }
      }
    }

    const parts = retrievedIds.map((id) => {
      const d = docById.get(id);
      if (!d) return "";
      return `### ${d.title}\n${BM25BodyExtractor.extract(d.body, q.question, 5)}`;
    });
    const evidence = sample.evidenceById.get(q.id) ?? "";
    entries.push({
      id: q.id,
      question: q.question,
      referenceAnswer: q.expectedKeywords[0] ?? "",
      evidence,
      retrievedDocIds: retrievedIds,
      evidenceCoverage: evidenceTokenCoverage(retrievedTexts, evidence),
      context: parts.join("\n\n"),
    });
  }

  return entries;
}

async function main(): Promise<void> {
  const N = Number(process.env.GB_N ?? 30);
  const dataDir = resolve(fileURLToPath(import.meta.url), "../../data");

  const medical = loadMedicalSample(
    `${dataDir}/gb-medical.json`,
    `${dataDir}/gb-medical-questions.json`,
    N,
    "Fact Retrieval",
    4000, // larger chunks → fewer embeddings
  );
  console.log(`\n=== Medical (${medical.queries.length} fact-retrieval queries, ${medical.docs.length} chunks) ===`);
  const medVS = await buildVectorStore(medical);

  for (const r of ["vanilla", "hybrid", "multihop"] as const) {
    const entries = await indexAndQuery(medical, r, 5, medVS);
    const avgCov = entries.reduce((s, e) => s + e.evidenceCoverage, 0) / entries.length;
    const fullCov = entries.filter((e) => e.evidenceCoverage >= 0.7).length;
    console.log(`  ${r}: avg evidence-token coverage ${avgCov.toFixed(3)} | ≥0.7 coverage in ${fullCov}/${entries.length} queries`);
    writeFileSync(
      `${dataDir}/../src/claude-gb-medical-${r}-contexts.json`,
      JSON.stringify(entries, null, 2),
    );
  }

  const novel = loadNovelSample(
    `${dataDir}/gb-novel.json`,
    `${dataDir}/gb-novel-questions.json`,
    N,
    "Fact Retrieval",
    4000,
  );
  console.log(`\n=== Novel (${novel.queries.length} fact-retrieval queries, ${novel.docs.length} chunks across ${new Set(novel.docs.map((d) => d.id.replace(/-\d+$/, ""))).size} novels) ===`);
  const novVS = await buildVectorStore(novel);

  for (const r of ["vanilla", "hybrid", "multihop"] as const) {
    const entries = await indexAndQuery(novel, r, 5, novVS);
    const avgCov = entries.reduce((s, e) => s + e.evidenceCoverage, 0) / entries.length;
    const fullCov = entries.filter((e) => e.evidenceCoverage >= 0.7).length;
    console.log(`  ${r}: avg evidence-token coverage ${avgCov.toFixed(3)} | ≥0.7 coverage in ${fullCov}/${entries.length} queries`);
    writeFileSync(
      `${dataDir}/../src/claude-gb-novel-${r}-contexts.json`,
      JSON.stringify(entries, null, 2),
    );
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
