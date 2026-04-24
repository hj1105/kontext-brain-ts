/**
 * Dump retrieved contexts using the multi-hop retriever v2 (entity BM25
 * + full-question vector). Targets HotpotQA multi-hop questions.
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { loadSquadSample } from "./squad-corpus.js";
import { loadHotpotSample } from "./hotpot-corpus.js";
import type { BenchDoc, BenchQuery } from "./corpus.js";
import { BM25BodyExtractor } from "@kontext-brain/core";
import { LangChainVectorStore } from "@kontext-brain/llm";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MultiHopRetriever, extractEntities } from "./multihop-retriever.js";

interface ContextEntry {
  id: string;
  question: string;
  referenceAnswer: string;
  expectedDocIds: string[];
  retrievedDocIds: string[];
  extractedEntities: string[];
  context: string;
}

async function dumpMultiHop(
  label: string,
  docs: BenchDoc[],
  queries: BenchQuery[],
  outPath: string,
  topK: number,
): Promise<void> {
  console.log(`\n[${label}] corpus=${docs.length} queries=${queries.length} topK=${topK}`);
  const baseUrl = "http://localhost:11434";
  const embeddings = new OllamaEmbeddings({ baseUrl, model: "nomic-embed-text" });
  const vectorStore = new LangChainVectorStore(embeddings);

  for (const doc of docs) {
    const vec = await vectorStore.embed(`${doc.title}\n${doc.body.slice(0, 1500)}`);
    await vectorStore.upsert(`doc:${doc.id}`, vec, { docId: doc.id });
  }

  const retriever = new MultiHopRetriever(
    docs.map((d) => ({ id: d.id, title: d.title, body: d.body })),
    vectorStore,
  );

  const entries: ContextEntry[] = [];
  for (const q of queries) {
    const ents = extractEntities(q.question);
    const ranked = await retriever.retrieve(q.question, topK);
    const retrievedIds: string[] = [];
    const parts: string[] = [];
    for (const r of ranked) {
      retrievedIds.push(r.docId);
      parts.push(`### ${r.title}\n${BM25BodyExtractor.extract(r.body, q.question, 4)}`);
    }
    entries.push({
      id: q.id,
      question: q.question,
      referenceAnswer: q.expectedKeywords[0] ?? "",
      expectedDocIds: q.expectedDocIds,
      retrievedDocIds: retrievedIds,
      extractedEntities: ents,
      context: parts.join("\n\n"),
    });
    if (entries.length % 10 === 0) console.log(`  dumped ${entries.length}`);
  }

  writeFileSync(outPath, JSON.stringify(entries, null, 2));
  console.log(`  → ${outPath} (${entries.length} entries)`);
}

async function main(): Promise<void> {
  const squadData = loadSquadSample(
    resolve(fileURLToPath(import.meta.url), "../../data/squad_dev.json"),
    30,
    42,
  );
  await dumpMultiHop(
    "SQuAD (multi-hop v2, k=3)",
    squadData.docs,
    squadData.queries,
    resolve(fileURLToPath(import.meta.url), "../claude-squad-multihop-contexts.json"),
    3,
  );

  const hotpotData = loadHotpotSample(
    resolve(fileURLToPath(import.meta.url), "../../data/hotpot_dev.json"),
    20,
    42,
  );
  await dumpMultiHop(
    "HotpotQA (multi-hop v2, k=6)",
    hotpotData.docs,
    hotpotData.queries,
    resolve(fileURLToPath(import.meta.url), "../claude-hotpot-multihop-contexts.json"),
    6,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
