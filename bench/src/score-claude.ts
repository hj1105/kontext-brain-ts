/**
 * Score Claude-Code-as-LLM answers vs references + compare to Ollama.
 *
 * Reads:
 *   claude-squad-contexts.json, claude-squad-answers.json, squad-results.json
 *   claude-hotpot-contexts.json, claude-hotpot-answers.json, hotpot-results.json
 *
 * Computes: kw-substring accuracy per system (including new "claude-code" column).
 * For Claude-Code's accuracy, a manual judgment header is printed so the user
 * can review each pairing.
 */
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

interface ContextEntry {
  id: string;
  question: string;
  referenceAnswer: string;
}
interface AnswerEntry {
  id: string;
  answer: string;
}
interface SystemResult {
  answer: string;
  retrievedDocIds: string[];
  contextChars: number;
  latencyMs: number;
  recall: number;
  keywordHit: number;
  bothGoldHit?: number;
}
interface BenchRow {
  queryId: string;
  question: string;
  referenceAnswer: string;
  expectedDocIds: string[];
  systems: Record<string, SystemResult>;
}

function keywordHit(answer: string, reference: string): number {
  const a = answer.toLowerCase();
  const ref = reference.toLowerCase();
  if (a.includes(ref)) return 1;
  const refTokens = ref.split(/\s+/).filter((t) => t.length > 2);
  if (refTokens.length === 0) return a.includes(ref) ? 1 : 0;
  const hits = refTokens.filter((t) => a.includes(t)).length;
  return hits / refTokens.length;
}

function scorePair(label: string, contextsPath: string, answersPath: string, resultsPath: string): void {
  console.log(`\n=== ${label} ===`);
  if (!existsSync(contextsPath) || !existsSync(answersPath) || !existsSync(resultsPath)) {
    console.log(`  missing file(s); skip`);
    return;
  }
  const contexts = JSON.parse(readFileSync(contextsPath, "utf-8")) as ContextEntry[];
  const answers = JSON.parse(readFileSync(answersPath, "utf-8")) as AnswerEntry[];
  const ansById = new Map(answers.map((a) => [a.id, a.answer]));
  const results = JSON.parse(readFileSync(resultsPath, "utf-8")) as BenchRow[];

  const systemNames = Object.keys(results[0]?.systems ?? {});
  const N = contexts.length;

  const kwSum: Record<string, number> = Object.fromEntries(
    [...systemNames, "claude-code"].map((n) => [n, 0]),
  );

  console.log(`\nper-query kw-hit (0..1):`);
  console.log(`id       ref                                 | ${systemNames.join(" | ")} | claude-code`);
  for (const ctx of contexts) {
    const row = results.find((r) => r.queryId === ctx.id)!;
    const claudeAns = ansById.get(ctx.id) ?? "";
    const claudeKw = keywordHit(claudeAns, ctx.referenceAnswer);
    kwSum["claude-code"]! += claudeKw;
    const scores: string[] = [];
    for (const name of systemNames) {
      const kw = row.systems[name]?.keywordHit ?? 0;
      kwSum[name]! += kw;
      scores.push(kw.toFixed(2));
    }
    console.log(
      `${ctx.id.padEnd(8)} ${ctx.referenceAnswer.slice(0, 35).padEnd(36)} | ${scores.join("   ")} | ${claudeKw.toFixed(2)}`,
    );
  }

  console.log(`\naverage kw-hit across ${N} queries:`);
  const allCols = [...systemNames, "claude-code"];
  for (const name of allCols) {
    const avg = kwSum[name]! / N;
    console.log(`  ${name.padEnd(14)} ${avg.toFixed(3)}`);
  }
}

function main(): void {
  scorePair(
    "SQuAD 2.0 (30 queries)",
    resolve(__dirname, "claude-squad-contexts.json"),
    resolve(__dirname, "claude-squad-answers.json"),
    resolve(__dirname, "squad-results.json"),
  );
  scorePair(
    "HotpotQA (20 queries, multi-hop)",
    resolve(__dirname, "claude-hotpot-contexts.json"),
    resolve(__dirname, "claude-hotpot-answers.json"),
    resolve(__dirname, "hotpot-results.json"),
  );
}

main();
