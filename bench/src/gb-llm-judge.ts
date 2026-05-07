/**
 * LLM-judge ACC for GraphRAG-Bench answers, matching the leaderboard's
 * evaluation methodology (the leaderboard uses an LLM judge for Fact_ACC,
 * not token recall — token recall punishes correct-but-brief answers).
 *
 * Judge model: llama3.1:8b (same as answerer for consistency).
 * Prompt: given question + gold + candidate, output JSON {"correct": 0|1}.
 *
 * Reads:  bench/src/llm8b-gb-{domain}-{retriever}-answers.json
 *         bench/src/claude-gb-{domain}-{retriever}-contexts.json (gold)
 * Writes: bench/src/llm8b-gb-{domain}-{retriever}-judged.json
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

interface ContextEntry {
  id: string;
  question: string;
  referenceAnswer: string;
}
interface AnswerEntry { id: string; answer: string; }
interface JudgedEntry { id: string; question: string; gold: string; ans: string; correct: 0 | 1; raw: string; }

const JUDGE_PROMPT = `You are evaluating whether a candidate answer is correct given a question and a gold reference answer.

Rules:
- Output strict JSON only: {"correct": 0} or {"correct": 1}
- "correct: 1" means the candidate captures the same factual answer as the gold (paraphrasing is fine).
- "correct: 0" means the candidate is wrong, missing, or says INSUFFICIENT_CONTEXT.
- Do not add explanation. Do not add any text outside the JSON.`;

async function judgeOne(
  model: string,
  question: string,
  gold: string,
  candidate: string,
): Promise<{ correct: 0 | 1; raw: string }> {
  const userPrompt = `Question: ${question}\nGold answer: ${gold}\nCandidate answer: ${candidate}\n\nIs the candidate correct?`;
  const body = {
    model,
    messages: [
      { role: "system", content: JUDGE_PROMPT },
      { role: "user", content: userPrompt },
    ],
    stream: false,
    options: { num_gpu: 0, num_ctx: 1024, num_predict: 50, temperature: 0 },
    format: "json",
  };
  const res = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status}`);
  const j = await res.json() as { message?: { content: string } };
  const raw = j.message?.content ?? "";
  try {
    const parsed = JSON.parse(raw);
    return { correct: parsed.correct === 1 ? 1 : 0, raw };
  } catch {
    // Fallback: look for "correct: 1" or "1" near "correct"
    const m = raw.match(/correct[":\s]*(\d)/i);
    return { correct: m && m[1] === "1" ? 1 : 0, raw };
  }
}

async function judgeAll(domain: string, retriever: string, model: string): Promise<void> {
  const dataDir = resolve(fileURLToPath(import.meta.url), "../../data");
  const ctxPath = `${dataDir}/../src/claude-gb-${domain}-${retriever}-contexts.json`;
  const ansPath = `${dataDir}/../src/llm8b-gb-${domain}-${retriever}-answers.json`;
  const outPath = `${dataDir}/../src/llm8b-gb-${domain}-${retriever}-judged.json`;
  if (!existsSync(ctxPath) || !existsSync(ansPath)) {
    console.log(`  ${domain}/${retriever}: missing input, skip`);
    return;
  }
  const ctx = JSON.parse(readFileSync(ctxPath, "utf-8")) as ContextEntry[];
  const ansList = JSON.parse(readFileSync(ansPath, "utf-8")) as AnswerEntry[];
  const ansMap = new Map(ansList.map((a) => [a.id, a.answer]));
  console.log(`\n[judge ${domain}/${retriever}] N=${ctx.length} judge=${model}`);

  const out: JudgedEntry[] = [];
  let correct = 0;
  for (let i = 0; i < ctx.length; i++) {
    const c = ctx[i]!;
    const a = ansMap.get(c.id) ?? "";
    let r: { correct: 0 | 1; raw: string };
    try {
      r = await judgeOne(model, c.question, c.referenceAnswer, a);
    } catch {
      r = { correct: 0, raw: "ERROR" };
    }
    correct += r.correct;
    out.push({ id: c.id, question: c.question, gold: c.referenceAnswer, ans: a, correct: r.correct, raw: r.raw });
    if ((i + 1) % 10 === 0) console.log(`  judged ${i + 1}/${ctx.length} (running ACC=${(correct / (i + 1) * 100).toFixed(1)}%)`);
  }
  writeFileSync(outPath, JSON.stringify(out, null, 2));
  console.log(`  → ${outPath}  ACC=${correct}/${ctx.length} = ${(correct / ctx.length * 100).toFixed(2)}%`);
}

async function main(): Promise<void> {
  const model = process.env.GB_JUDGE_MODEL ?? "llama3.1:8b-instruct-q4_K_M";
  console.log(`=== LLM-judge ACC pass ===\nmodel=${model}`);
  for (const d of ["medical", "novel"]) {
    for (const r of ["vanilla", "hybrid", "multihop"]) {
      await judgeAll(d, r, model);
    }
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
