/**
 * Dump all chunked corpus to text files for Claude to read in batches and
 * extract entities + triples. Writes one chunk per JSONL line.
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { loadMedicalSample, loadNovelSample } from "./gb-corpus.js";

async function main(): Promise<void> {
  const dataDir = resolve(fileURLToPath(import.meta.url), "../../data");
  const N = Number(process.env.GB_N ?? 30);

  const medical = loadMedicalSample(
    `${dataDir}/gb-medical.json`,
    `${dataDir}/gb-medical-questions.json`,
    N, "Fact Retrieval", 1024,
  );
  const medOut = medical.docs.map((d) => JSON.stringify({ id: d.id, body: d.body })).join("\n");
  writeFileSync(`${dataDir}/gb-medical-chunks.jsonl`, medOut);
  console.log(`medical: ${medical.docs.length} chunks → gb-medical-chunks.jsonl`);

  const novel = loadNovelSample(
    `${dataDir}/gb-novel.json`,
    `${dataDir}/gb-novel-questions.json`,
    N, "Fact Retrieval", 1024,
  );
  const novOut = novel.docs.map((d) => JSON.stringify({ id: d.id, body: d.body })).join("\n");
  writeFileSync(`${dataDir}/gb-novel-chunks.jsonl`, novOut);
  console.log(`novel: ${novel.docs.length} chunks → gb-novel-chunks.jsonl`);
}

main().catch((e) => { console.error(e); process.exit(1); });
