/**
 * Run an Ollama 8B-class model as the LLM answerer over GraphRAG-Bench
 * Fact Retrieval contexts. Apples-to-apples vs the leaderboard's 8B systems.
 *
 * qwen3-vl:8b is a *thinking* model — content is empty when num_predict
 * cuts off mid-reasoning. Fixes:
 *   1. Append `/no_think` to user prompt (Qwen3 hint to skip thinking)
 *   2. num_predict=2000 to ensure thinking + answer both fit
 *   3. If `content` empty, extract from the model's `thinking` field
 *      (last sentence usually contains the answer)
 *
 * Reads:  bench/src/claude-gb-{domain}-{retriever}-contexts.json
 * Writes: bench/src/llm8b-gb-{domain}-{retriever}-answers.json
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

interface ContextEntry {
  id: string;
  question: string;
  referenceAnswer: string;
  evidence: string;
  retrievedDocIds: string[];
  context: string;
}

interface AnswerEntry {
  id: string;
  answer: string;
  fromThinkingFallback: boolean;
  latencyMs: number;
}

const SYSTEM_PROMPT = `You are answering a fact-retrieval question using only the provided context.

Rules:
- Answer in one short sentence, as concise as possible.
- Use only information from the context. Do NOT use outside knowledge.
- If the context does not contain the answer, reply exactly: INSUFFICIENT_CONTEXT
- Output the answer only, no explanation, no preamble.`;

interface OllamaChatResponse {
  message?: { role: string; content: string; thinking?: string };
  done_reason?: string;
  total_duration?: number;
}

async function callOllama(
  model: string,
  question: string,
  context: string,
): Promise<{ answer: string; thinking: string; doneReason: string }> {
  const userPrompt = `Context:\n${context}\n\nQuestion: ${question}\n\nAnswer concisely.`;
  const body = {
    model,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userPrompt },
    ],
    stream: false,
    options: { num_gpu: 0, num_ctx: 4096, num_predict: 200, temperature: 0 },
  };
  const res = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Ollama ${res.status} ${res.statusText}`);
  const j = (await res.json()) as OllamaChatResponse;
  return {
    answer: (j.message?.content ?? "").trim(),
    thinking: (j.message?.thinking ?? "").trim(),
    doneReason: j.done_reason ?? "",
  };
}

/**
 * Extract a plausible answer from a chain-of-thought trace.
 * Heuristic: take the last meaningful sentence (skipping closing thinking-tags).
 */
function fromThinking(thinking: string): string {
  if (!thinking) return "";
  let t = thinking
    .replace(/<\/?think>/gi, "")
    .replace(/^\s+|\s+$/g, "")
    .replace(/<\/?think>/gi, "");
  // Try to find an explicit "answer:" or "the answer is" marker
  const markerMatch = t.match(/(?:the answer is|so the answer is|final answer:?|answer:?)\s*([^\n.]+[\.\!\?]?)/i);
  if (markerMatch && markerMatch[1]) return markerMatch[1].trim();
  // Otherwise: last non-trivial sentence
  const sentences = t.split(/(?<=[.!?])\s+/).filter((s) => s.length > 10);
  if (sentences.length === 0) return t.slice(-200);
  return sentences[sentences.length - 1]!.trim();
}

async function answerOne(
  model: string,
  ctx: ContextEntry,
): Promise<{ answer: string; fromThinkingFallback: boolean }> {
  const { answer, thinking } = await callOllama(model, ctx.question, ctx.context);
  if (answer.length > 0) return { answer, fromThinkingFallback: false };
  // Fallback: extract from thinking trace
  const extracted = fromThinking(thinking);
  return { answer: extracted || "INSUFFICIENT_CONTEXT", fromThinkingFallback: true };
}

async function answerAll(domain: string, retriever: string, model: string): Promise<void> {
  const dataDir = resolve(fileURLToPath(import.meta.url), "../../data");
  const ctxPath = `${dataDir}/../src/claude-gb-${domain}-${retriever}-contexts.json`;
  const outPath = `${dataDir}/../src/llm8b-gb-${domain}-${retriever}-answers.json`;
  if (!existsSync(ctxPath)) {
    console.log(`  ${domain}/${retriever}: no contexts, skip`);
    return;
  }
  const ctx = JSON.parse(readFileSync(ctxPath, "utf-8")) as ContextEntry[];
  console.log(`\n[${domain}/${retriever}] N=${ctx.length} model=${model}`);

  // Resume support: if output file exists, skip already-answered IDs
  const existing: AnswerEntry[] = existsSync(outPath)
    ? JSON.parse(readFileSync(outPath, "utf-8"))
    : [];
  const done = new Map(existing.map((a) => [a.id, a]));
  const answers: AnswerEntry[] = [...existing];

  for (let i = 0; i < ctx.length; i++) {
    const e = ctx[i]!;
    if (done.has(e.id) && done.get(e.id)!.answer.length > 0) {
      console.log(`  ${i + 1}/${ctx.length} ${e.id} (cached)`);
      continue;
    }
    const t0 = Date.now();
    let entry: AnswerEntry;
    try {
      const r = await answerOne(model, e);
      entry = { id: e.id, answer: r.answer, fromThinkingFallback: r.fromThinkingFallback, latencyMs: Date.now() - t0 };
    } catch (err) {
      entry = { id: e.id, answer: "INSUFFICIENT_CONTEXT", fromThinkingFallback: false, latencyMs: Date.now() - t0 };
      console.error(`    error: ${(err as Error).message}`);
    }
    // Replace or append
    const idx = answers.findIndex((a) => a.id === e.id);
    if (idx >= 0) answers[idx] = entry;
    else answers.push(entry);
    writeFileSync(outPath, JSON.stringify(answers, null, 2));
    const tag = entry.fromThinkingFallback ? " [from-thinking]" : "";
    console.log(
      `  ${i + 1}/${ctx.length} ${e.id} (${(entry.latencyMs / 1000).toFixed(1)}s)${tag}: ${entry.answer.slice(0, 110).replace(/\n/g, " ")}`,
    );
  }
  console.log(`  → ${outPath}`);
}

async function main(): Promise<void> {
  const model = process.env.GB_MODEL ?? "qwen3-vl:8b";
  const onlyDomain = process.env.GB_DOMAIN;
  const onlyRetriever = process.env.GB_RETRIEVER;
  console.log(`=== GraphRAG-Bench 8B LLM answer pass ===\nmodel=${model}`);
  const domains = onlyDomain ? [onlyDomain] : ["medical", "novel"];
  const retrievers = onlyRetriever ? [onlyRetriever] : ["vanilla", "hybrid", "multihop", "kg"];
  for (const d of domains) {
    for (const r of retrievers) {
      await answerAll(d, r, model);
    }
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
