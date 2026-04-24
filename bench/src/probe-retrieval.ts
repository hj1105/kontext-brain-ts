/**
 * Quick probe: index HotpotQA corpus, then try retrieval for various
 * entity queries to see what nomic-embed-text actually returns.
 */
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { loadHotpotSample } from "./hotpot-corpus.js";
import { LangChainVectorStore } from "@kontext-brain/llm";
import { OllamaEmbeddings } from "@langchain/ollama";

async function main(): Promise<void> {
  const { docs } = loadHotpotSample(
    resolve(fileURLToPath(import.meta.url), "../../data/hotpot_dev.json"),
    20,
    42,
  );
  console.log(`corpus=${docs.length}`);
  const baseUrl = "http://localhost:11434";
  const embeddings = new OllamaEmbeddings({ baseUrl, model: "nomic-embed-text" });
  const vectorStore = new LangChainVectorStore(embeddings);
  for (const d of docs) {
    const vec = await vectorStore.embed(`search_document: ${d.title}\n${d.body.slice(0, 1500)}`);
    await vectorStore.upsert(`doc:${d.id}`, vec, { docId: d.id });
  }
  const queryEmbed = async (q: string) => vectorStore.embed(`search_query: ${q}`);
  // Use lower-level embed for queries with prefix
  const similar = async (q: string, k: number) => {
    const qvec = await queryEmbed(q);
    const scored: Array<{ key: string; score: number }> = [];
    const { cosineSimilarity } = await import("@kontext-brain/core");
    const inner = (vectorStore as unknown as { index: Map<string, { embedding: Float32Array }> }).index;
    for (const [key, { embedding }] of inner) {
      if (!key.startsWith("doc:")) continue;
      scored.push({ key, score: cosineSimilarity(qvec, embedding) });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k).map((s) => s.key.slice(4));
  };

  // Verify traveling-wilburys and tom-petty are in corpus
  const hasTW = docs.find((d) => d.id === "traveling-wilburys");
  const hasTP = docs.find((d) => d.id === "tom-petty");
  console.log("traveling-wilburys present:", !!hasTW);
  console.log("tom-petty present:", !!hasTP);

  const probes = [
    "Who or what is Traveling Wilburys?",
    "Who or what is Tom Petty?",
    "Tell me about Traveling Wilburys.",
    "Tell me about Tom Petty.",
  ];
  for (const p of probes) {
    const hits = await similar(p, 5);
    console.log(`\nquery="${p}" (with search_query prefix)`);
    for (const h of hits) console.log(`  ${h}`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
