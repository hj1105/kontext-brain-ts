import { execFile } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { afterEach, describe, expect, it } from "vitest";
import type { DeepSweContextCorpus } from "./contracts.js";
import { buildContextBundle, sha256 } from "./corpus.js";
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
    readonly mandatoryRecords: readonly {
      readonly kind: string;
      readonly recordId: string;
      readonly statement: string;
      readonly verifiers: readonly { readonly kind: string; readonly ref: string }[];
    }[];
    readonly sources: readonly { readonly evidenceId: string; readonly text: string }[];
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
    expect(receipt.mandatoryRecords[0]).toMatchObject({
      kind: "invariant",
      recordId: "invariant:stable-order",
      statement: "Parser.equal_values must preserve stable ordering.",
      verifiers: [{ kind: "test", ref: "tests/test_parser.py" }],
    });
    expect(receipt.sources.map((source) => source.evidenceId)).toEqual(["evidence:design"]);
  });

  it("does not pad lexical retrieval with zero-relevance Evidence", async () => {
    const prepared = await fixture("rag");
    const result = await runTool(prepared, ["search", "--query", "completely-unrelated-token"]);
    expect(result.results).toEqual([]);
  });

  it("retains Organization rules and every matching requirement regardless of the limit", async () => {
    const corpus = fixtureCorpus();
    const record = required(corpus.normativeRecords[0]);
    const invariant = record.revision;
    if (invariant.kind !== "invariant") throw new Error("Expected an Invariant fixture");
    const prepared = await fixture("kontext", {
      ...corpus,
      normativeRecords: [
        ...Array.from({ length: 9 }, (_, index) => ({
          ...record,
          revision: { ...record.revision, recordId: `invariant:required-${index}` },
        })),
        {
          revision: {
            ...invariant,
            recordId: "invariant:organization",
            scope: { kind: "organization", organizationId: corpus.organizationId },
            statement: "Retain audit history.",
          },
        },
      ],
    });
    const result = await runTool(prepared, [...beginArguments, "--limit", "1"]);
    const receipt = required(result.receipt);
    expect(result.editingAllowed).toBe(true);
    expect(receipt.mandatoryRecords).toHaveLength(10);
    expect(receipt.mandatoryRecords.map((entry) => entry.recordId)).toContain(
      "invariant:organization",
    );
    expect(receipt.sources.map((source) => source.evidenceId)).toEqual(["evidence:design"]);
  });

  it("requires both selector fields to match instead of promoting lexical neighbors", async () => {
    const corpus = fixtureCorpus();
    const record = required(corpus.normativeRecords[0]);
    const prepared = await fixture("kontext", {
      ...corpus,
      normativeRecords: [
        {
          ...record,
          symbolSelectors: [{ relativePath: "src/parser.py", qualifiedName: "Parser.other" }],
        },
        {
          ...record,
          revision: { ...record.revision, recordId: "invariant:other-file" },
          symbolSelectors: [{ relativePath: "src/other.py", qualifiedName: "Parser.equal_values" }],
        },
      ],
    });
    const result = await runTool(prepared, beginArguments);
    expect(required(result.receipt).mandatoryRecords).toEqual([]);
  });

  it("returns full mandatory Evidence including requirements after the search snippet", async () => {
    const corpus = fixtureCorpus();
    const evidence = required(corpus.evidence[0]);
    const text = `${"Background. ".repeat(500)}The final constraint must also be preserved.`;
    const prepared = await fixture("kontext", {
      ...corpus,
      evidence: [{ ...evidence, text, contentSha256: sha256(text) }],
    });
    const result = await runTool(prepared, beginArguments);
    expect(required(result.receipt).sources[0]?.text).toBe(text);
  });

  it.each([
    "missing evidence",
    "corrupt evidence",
    "evidence egress denied",
    "record egress denied",
    "wrong organization",
  ] as const)("blocks editing when mandatory context has %s", async (failure) => {
    const corpus = fixtureCorpus();
    const evidence = required(corpus.evidence[0]);
    const record = required(corpus.normativeRecords[0]);
    const prepared = await fixture("kontext", {
      ...corpus,
      evidence:
        failure === "missing evidence"
          ? []
          : [
              {
                ...evidence,
                text: failure === "corrupt evidence" ? "Changed content" : evidence.text,
                allowedRuntimeProviders:
                  failure === "evidence egress denied" ? [] : evidence.allowedRuntimeProviders,
              },
            ],
      normativeRecords: [
        {
          ...record,
          revision: {
            ...record.revision,
            organizationId:
              failure === "wrong organization" ? "organization:other" : corpus.organizationId,
            egress: {
              ...record.revision.egress,
              allowedRuntimeProviders:
                failure === "record egress denied"
                  ? []
                  : record.revision.egress.allowedRuntimeProviders,
            },
          },
        },
      ],
    });
    await expect(runTool(prepared, beginArguments)).rejects.toMatchObject({
      code: 2,
      stdout: expect.stringContaining('"editingAllowed": false'),
    });
  });

  it("allows the explicitly empty infrastructure corpus", async () => {
    const prepared = await fixture("kontext", {
      ...fixtureCorpus(),
      evidence: [],
      normativeRecords: [],
    });
    const result = await runTool(prepared, beginArguments);
    expect(result).toMatchObject({
      ok: true,
      editingAllowed: true,
      receipt: { mandatoryRecords: [], sources: [] },
    });
  });
});

const beginArguments = [
  "begin-logic",
  "--path",
  "src/parser.py",
  "--symbol",
  "Parser.equal_values",
] as const;

async function fixture(
  arm: "baseline" | "rag" | "kontext",
  corpus: DeepSweContextCorpus = fixtureCorpus(),
): Promise<{
  bundle: string;
  log: string;
}> {
  const root = await mkdtemp(path.join(tmpdir(), "kontext-deepswe-tool-"));
  cleanup.add(root);
  const bundle = path.join(root, "bundle.json");
  const log = path.join(root, "calls.jsonl");
  await writeFile(bundle, JSON.stringify(buildContextBundle(arm, corpus)), "utf8");
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
