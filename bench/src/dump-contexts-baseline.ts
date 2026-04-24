/**
 * Dump retrieved contexts using the BASELINE vector RAG retriever (not
 * kontext-brain's hybrid). Enables apples-to-apples comparison where only
 * the retriever differs and the answerer (Claude Code) is constant.
 *
 * Outputs:
 *   claude-squad-baseline-contexts.json
 *   claude-hotpot-baseline-contexts.json
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { loadSquadSample } from "./squad-corpus.js";
import { loadHotpotSample } from "./hotpot-corpus.js";
import type { BenchDoc, BenchQuery } from "./corpus.js";
import { BM25BodyExtractor } from "@kontext-brain/core";
import { LangChainVectorStore } from "@kontext-brain/llm";
import { OllamaEmbeddings } from "@langchain/ollama";

interface ContextEntry {
  id: string;
  question: string;
  referenceAnswer: string;
  expectedDocIds: string[];
  retrievedDocIds: string[];
  context: string;
}

/** Pure vector-RAG: embed each doc at index time, cosine-similarity retrieve. */
async function dumpBaseline(
  label: string,
  docs: BenchDoc[],
  queries: BenchQuery[],
  outPath: string,
): Promise<void> {
  console.log(`\n[${label}] corpus=${docs.length} queries=${queries.length}`);
  const baseUrl = "http://localhost:11434";
  const embeddings = new OllamaEmbeddings({ baseUrl, model: "nomic-embed-text" });
  const vectorStore = new LangChainVectorStore(embeddings);
  const docById = new Map(docs.map((d) => [d.id, d]));

  // Index: embed each doc (title + body) and upsert by id
  for (const doc of docs) {
    const vec = await vectorStore.embed(`${doc.title}\n${doc.body.slice(0, 1500)}`);
    await vectorStore.upsert(`doc:${doc.id}`, vec, { docId: doc.id });
  }

  const entries: ContextEntry[] = [];
  for (const q of queries) {
    const hits = await vectorStore.similaritySearchWithPrefix(q.question, "doc:", 3);
    const retrievedIds: string[] = [];
    const parts: string[] = [];
    for (const id of hits) {
      const doc = docById.get(id);
      if (!doc) continue;
      retrievedIds.push(id);
      parts.push(
        `### ${doc.title}\n${BM25BodyExtractor.extract(doc.body, q.question, 4)}`,
      );
    }
    entries.push({
      id: q.id,
      question: q.question,
      referenceAnswer: q.expectedKeywords[0] ?? "",
      expectedDocIds: q.expectedDocIds,
      retrievedDocIds: retrievedIds,
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
  await dumpBaseline(
    "SQuAD (baseline RAG)",
    squadData.docs,
    squadData.queries,
    resolve(fileURLToPath(import.meta.url), "../claude-squad-baseline-contexts.json"),
  );

  const hotpotData = loadHotpotSample(
    resolve(fileURLToPath(import.meta.url), "../../data/hotpot_dev.json"),
    20,
    42,
  );
  await dumpBaseline(
    "HotpotQA (baseline RAG)",
    hotpotData.docs,
    hotpotData.queries,
    resolve(fileURLToPath(import.meta.url), "../claude-hotpot-baseline-contexts.json"),
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
