import { execFile } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { afterEach, describe, expect, it } from "vitest";
import { buildContextBundle } from "./corpus.js";
import { fixtureCorpus } from "./test-fixtures.js";

const execFileAsync = promisify(execFile);
const toolPath = fileURLToPath(new URL("context_tool.py", import.meta.url));
const cleanup = new Set<string>();

interface ToolResult {
  readonly ok: boolean;
  readonly arm: string;
  readonly results?: readonly unknown[];
  readonly editingAllowed?: boolean;
  readonly receipt?: {
    readonly mandatoryRecords: readonly { readonly recordId: string }[];
    readonly sources: readonly { readonly documentId: string }[];
  };
}

afterEach(async () => {
  await Promise.all([...cleanup].map((entry) => rm(entry, { recursive: true, force: true })));
  cleanup.clear();
});

describe("DeepSWE context command", () => {
  it("returns no context in baseline while preserving the same command contract", async () => {
    const prepared = await fixture("baseline");
    const result = await runTool(prepared, ["search", "--query", "stable parser"]);
    expect(result).toMatchObject({ ok: true, arm: "baseline", results: [] });
  });

  it("returns selector-matched records together with their exact evidence closure", async () => {
    const prepared = await fixture("kontext");
    const result = await runTool(prepared, [
      "begin-logic",
      "--path",
      "src/parser.py",
      "--symbol",
      "Parser.equal_values",
      "--responsibility",
      "unrelated wording",
    ]);
    const receipt = required(result.receipt);
    expect(result.editingAllowed).toBe(true);
    expect(receipt.mandatoryRecords).toHaveLength(1);
    expect(receipt.mandatoryRecords[0]?.recordId).toBe("invariant:stable-order");
    expect(receipt.sources.map((source) => source.documentId)).toEqual(["doc:design"]);
  });

  it("does not pad lexical retrieval with zero-relevance documents", async () => {
    const prepared = await fixture("rag");
    const result = await runTool(prepared, ["search", "--query", "completely-unrelated-token"]);
    expect(result.results).toEqual([]);
  });
});

async function fixture(arm: "baseline" | "rag" | "kontext"): Promise<{
  bundle: string;
  log: string;
}> {
  const root = await mkdtemp(path.join(tmpdir(), "kontext-deepswe-tool-"));
  cleanup.add(root);
  const bundle = path.join(root, "bundle.json");
  const log = path.join(root, "calls.jsonl");
  await writeFile(bundle, JSON.stringify(buildContextBundle(arm, fixtureCorpus())), "utf8");
  return { bundle, log };
}

async function runTool(
  prepared: { bundle: string; log: string },
  args: readonly string[],
): Promise<ToolResult> {
  const result = await execFileAsync("python3", [toolPath, ...args], {
    env: {
      ...process.env,
      KONTEXT_EVAL_BUNDLE: prepared.bundle,
      KONTEXT_EVAL_LOG: prepared.log,
    },
  });
  return JSON.parse(result.stdout) as ToolResult;
}

function required<T>(value: T | undefined): T {
  if (value === undefined) throw new Error("Missing test fixture value");
  return value;
}
