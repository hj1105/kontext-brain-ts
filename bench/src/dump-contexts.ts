/**
 * Dump retrieved contexts (hybrid retriever) for SQuAD + HotpotQA samples
 * to JSON so Claude Code can act as the LLM answerer instead of Ollama.
 *
 * Output files:
 *   bench/src/claude-squad-contexts.json   — 30 SQuAD queries
 *   bench/src/claude-hotpot-contexts.json  — 20 HotpotQA queries
 *
 * Each entry: { id, question, referenceAnswer, expectedDocIds,
 *               retrievedDocIds, context }
 *
 * Claude Code then reads these, writes answers to
 *   bench/src/claude-squad-answers.json / claude-hotpot-answers.json
 *   as [{ id, answer }, ...]
 *
 * A third script (score-claude.ts) computes accuracy from the answers.
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { loadSquadSample } from "./squad-corpus.js";
import { loadHotpotSample } from "./hotpot-corpus.js";
import { SquadKontextHybrid } from "./squad-runner.js";
import type { BenchDoc, BenchQuery } from "./corpus.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

interface ContextEntry {
  id: string;
  question: string;
  referenceAnswer: string;
  expectedDocIds: string[];
  retrievedDocIds: string[];
  context: string;
}

async function dumpFor(
  label: string,
  docs: BenchDoc[],
  queries: BenchQuery[],
  outPath: string,
): Promise<void> {
  console.log(`\n[${label}] corpus=${docs.length} queries=${queries.length}`);
  const hybrid = new SquadKontextHybrid();
  await hybrid.index(docs);
  // Override the query method to skip the LLM call — we only want context.
  // Access internals via cast.
  const state = (hybrid as unknown as {
    entityIndex: import("@kontext-brain/core").InMemoryEntityIndex;
    metaByDoc: Map<string, import("@kontext-brain/core").MetaDocument>;
    state: { vectorStore: import("@kontext-brain/core").VectorStore };
    corpusById: Map<string, BenchDoc>;
  });
  const { HybridRetriever, BM25BodyExtractor } = await import("@kontext-brain/core");
  const retriever = new HybridRetriever(
    state.entityIndex,
    state.state.vectorStore,
    async () => state.metaByDoc,
    0.4,
    0.6,
    1,
    "doc:",
    20,
  );

  const entries: ContextEntry[] = [];
  for (const q of queries) {
    const ranked = await retriever.retrieve(q.question, 3);
    const retrievedIds: string[] = [];
    const parts: string[] = [];
    for (const { doc } of ranked) {
      const body = state.corpusById.get(doc.id);
      if (!body) continue;
      retrievedIds.push(doc.id);
      parts.push(
        `### ${body.title}\n${BM25BodyExtractor.extract(body.body, q.question, 4)}`,
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
    resolve(__dirname, "../data/squad_dev.json"),
    30,
    42,
  );
  await dumpFor(
    "SQuAD",
    squadData.docs,
    squadData.queries,
    resolve(__dirname, "claude-squad-contexts.json"),
  );

  const hotpotData = loadHotpotSample(
    resolve(__dirname, "../data/hotpot_dev.json"),
    20,
    42,
  );
  await dumpFor(
    "HotpotQA",
    hotpotData.docs,
    hotpotData.queries,
    resolve(__dirname, "claude-hotpot-contexts.json"),
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
