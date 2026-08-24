import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { CodexJsonClient, type CommandRunner } from "./codex-json.js";
import type { AnswerResult, DatasetBundle, RetrievalResult } from "./contracts.js";
import { defaultDatasetPaths } from "./datasets.js";
import { DEFAULT_RAG_EVAL_MANIFEST, manifestDigest } from "./manifest.js";
import {
  answerInputDigest,
  answerQueries,
  freezeEvaluationSample,
  freezeRunManifest,
  judgeAnswers,
  judgeInputDigest,
  loadCompletedRetrieval,
} from "./pipeline.js";

const directories: string[] = [];

afterEach(() => {
  for (const directory of directories.splice(0))
    rmSync(directory, { recursive: true, force: true });
});

describe("run manifest freeze", () => {
  it("rejects checkpoint reuse after a benchmark configuration change", () => {
    const directory = mkdtempSync(join(tmpdir(), "rag-eval-manifest-"));
    directories.push(directory);
    freezeRunManifest(DEFAULT_RAG_EVAL_MANIFEST, directory);
    const changed = {
      ...DEFAULT_RAG_EVAL_MANIFEST,
      benchmarkPolicy: {
        ...DEFAULT_RAG_EVAL_MANIFEST.benchmarkPolicy,
        humanAuditPerDataset: 99,
      },
    };
    expect(() => freezeRunManifest(changed, directory)).toThrow(/manifest mismatch/);
  });

  it("rejects reuse when the frozen evaluation sample changes", () => {
    const directory = mkdtempSync(join(tmpdir(), "rag-eval-sample-"));
    directories.push(directory);
    const sample = {
      schemaVersion: "1.0.0" as const,
      datasetId: "graphrag-bench-medical" as const,
      method: "deterministic-proportional-category-stratified" as const,
      seed: 20260814,
      requested: 200,
      population: 10,
      selected: 1,
      categories: [{ category: "Fact", population: 10, selected: 1 }],
      queryIds: ["q1"],
      sampleDigest: "first",
    };
    freezeEvaluationSample(sample, directory);
    expect(() => freezeEvaluationSample({ ...sample, sampleDigest: "second" }, directory)).toThrow(
      /sample mismatch/i,
    );
  });
});

describe("retrieval completion cache", () => {
  it.each(["error", "blocked"] as const)(
    "does not reuse a full-size retrieval file containing %s rows",
    (status) => {
      const directory = mkdtempSync(join(tmpdir(), "rag-eval-error-retrieval-cache-"));
      directories.push(directory);
      const bundle = checkpointBundle("Question?");
      const retrievalPath = join(directory, "retrieval.jsonl");
      const failed: RetrievalResult = {
        datasetId: bundle.id,
        frameworkId: "kontext-brain",
        queryId: "q1",
        status,
        evidence: [],
        latencyMs: 0,
        inputTokens: null,
        error: "temporary network failure",
        frameworkVersion: "unresolved",
        configDigest: manifestDigest(DEFAULT_RAG_EVAL_MANIFEST),
      };
      writeFileSync(retrievalPath, `${JSON.stringify(failed)}\n`, "utf8");

      const cached = loadCompletedRetrieval(
        "kontext-brain",
        bundle,
        {
          workDirectory: directory,
          datasetPaths: defaultDatasetPaths(process.cwd()),
          stage: "retrieval",
          topK: 10,
          candidateK: 50,
        },
        DEFAULT_RAG_EVAL_MANIFEST,
        retrievalPath,
      );

      expect(cached).toBeNull();
    },
  );
});

describe("v13 answer policy", () => {
  it("opts v13 retrieval into evidence-needs answers while leaving v12 unchanged", async () => {
    const directory = mkdtempSync(join(tmpdir(), "rag-eval-v13-answer-"));
    directories.push(directory);
    const prompts: string[] = [];
    const runner: CommandRunner = async (_command, args, stdin) => {
      prompts.push(stdin);
      const supportedNeeds = stdin.includes("supported evidence need");
      writeFileSync(
        outputMessagePath(args),
        JSON.stringify({
          results: [
            supportedNeeds
              ? {
                  query_id: "q1",
                  claims: [{ claim: "Supported answer.", citation: "e1" }],
                  abstained: false,
                  abstention_reason: null,
                }
              : {
                  query_id: "q1",
                  answer: "Supported answer.",
                  citations: ["e1"],
                  abstained: false,
                  abstention_reason: null,
                },
          ],
        }),
      );
      return { exitCode: 0, stdout: "", stderr: "", durationMs: 1 };
    };
    const client = new CodexJsonClient(runner);
    const bundle: DatasetBundle = {
      id: "graphrag-bench-medical",
      track: "static-kb",
      documents: [],
      queries: [
        {
          id: "q1",
          text: "Question?",
          referenceAnswer: "private reference",
          goldEvidenceIds: ["private-gold"],
          goldEvidenceText: ["private evidence"],
          answerable: true,
          category: "test",
          metadata: {},
        },
      ],
      provenance: { source: "test", version: "1", license: "test" },
    };
    const retrieval = (version: string, mode: string): RetrievalResult => ({
      datasetId: bundle.id,
      frameworkId: "kontext-brain",
      queryId: "q1",
      status: "ok",
      evidence: [
        {
          id: "e1",
          sourceId: "source",
          text: "Literal retrieved evidence.",
          score: 1,
          rank: 1,
          metadata: { retrievalMode: mode },
        },
      ],
      latencyMs: 1,
      inputTokens: 1,
      error: null,
      frameworkVersion: version,
      configDigest: "test",
      answerPolicy:
        mode === "v13-anchored-evidence-answer-stack" ? "supported-evidence-needs" : undefined,
    });

    await answerQueries(
      DEFAULT_RAG_EVAL_MANIFEST,
      bundle,
      "kontext-brain",
      [
        retrieval(
          "workspace-0.1.0+v13-anchored-evidence-answer-stack",
          "v13-anchored-evidence-answer-stack",
        ),
      ],
      bundle.queries,
      join(directory, "v13"),
      client,
    );
    await answerQueries(
      DEFAULT_RAG_EVAL_MANIFEST,
      bundle,
      "kontext-brain",
      [
        retrieval(
          "workspace-0.1.0+v12-multi-query-plan-aware-coverage-stack",
          "v12-multi-query-plan-aware-coverage-stack",
        ),
      ],
      bundle.queries,
      join(directory, "v12"),
      client,
    );

    expect(prompts[0]).toContain("Internally identify the distinct evidence needs");
    expect(prompts[1]).not.toContain("Internally identify the distinct evidence needs");
    expect(prompts.join("\n")).not.toContain("private reference");
    expect(prompts.join("\n")).not.toContain("private-gold");
  });

  it("invalidates answer checkpoints when evidence, policy, model, or query changes", async () => {
    const directory = mkdtempSync(join(tmpdir(), "rag-eval-answer-digest-"));
    directories.push(directory);
    let calls = 0;
    const runner: CommandRunner = async (_command, args, stdin) => {
      calls += 1;
      const outputPath = outputMessagePath(args);
      const result = stdin.includes("supported evidence need")
        ? {
            query_id: "q1",
            claims: [{ claim: `Structured call ${calls}`, citation: "e1" }],
            abstained: false,
            abstention_reason: null,
          }
        : {
            query_id: "q1",
            answer: `Default call ${calls}`,
            citations: ["e1"],
            abstained: false,
            abstention_reason: null,
          };
      writeFileSync(outputPath, JSON.stringify({ results: [result] }));
      return { exitCode: 0, stdout: "", stderr: "", durationMs: 1 };
    };
    const client = new CodexJsonClient(runner);
    const bundle = checkpointBundle("Question one?");
    const retrieval = (
      text: string,
      answerPolicy?: "supported-evidence-needs",
    ): RetrievalResult => ({
      datasetId: bundle.id,
      frameworkId: "kontext-brain",
      queryId: "q1",
      status: "ok",
      evidence: [
        {
          id: "e1",
          sourceId: "source",
          text,
          score: 1,
          rank: 1,
          metadata: {},
        },
      ],
      latencyMs: 1,
      inputTokens: 1,
      error: null,
      frameworkVersion: answerPolicy ? "v13" : "v12",
      configDigest: answerPolicy ? "v13" : "v12",
      answerPolicy,
    });
    const inputDigests: string[] = [];
    const run = async (
      currentBundle: DatasetBundle,
      currentRetrieval: RetrievalResult,
      manifest = DEFAULT_RAG_EVAL_MANIFEST,
    ) => {
      const results = await answerQueries(
        manifest,
        currentBundle,
        "kontext-brain",
        [currentRetrieval],
        currentBundle.queries,
        directory,
        client,
      );
      inputDigests.push(results[0]?.inputDigest ?? "");
    };

    await run(bundle, retrieval("Evidence one."));
    await run(bundle, retrieval("Evidence two."));
    await run(bundle, retrieval("Evidence two.", "supported-evidence-needs"));
    await run(bundle, retrieval("Evidence two.", "supported-evidence-needs"), {
      ...DEFAULT_RAG_EVAL_MANIFEST,
      models: {
        ...DEFAULT_RAG_EVAL_MANIFEST.models,
        answer: { ...DEFAULT_RAG_EVAL_MANIFEST.models.answer, model: "gpt-different" },
      },
    });
    const changedBundle = checkpointBundle("Question two?");
    await run(changedBundle, {
      ...retrieval("Evidence two.", "supported-evidence-needs"),
      datasetId: changedBundle.id,
    });

    expect(calls).toBe(5);
    expect(new Set(inputDigests).size).toBe(5);
    expect(inputDigests).not.toContain("");
  });

  it("invalidates judge checkpoints when the candidate answer changes", async () => {
    const directory = mkdtempSync(join(tmpdir(), "rag-eval-judge-digest-"));
    directories.push(directory);
    let calls = 0;
    const runner: CommandRunner = async (_command, args) => {
      calls += 1;
      writeFileSync(
        outputMessagePath(args),
        JSON.stringify({
          results: [
            {
              query_id: "q1",
              answer_correctness: 1,
              completeness: 1,
              strict_faithfulness: 1,
              citation_precision: 1,
              citation_recall: 1,
              acceptable_abstention: false,
              clarity: 1,
              conciseness: 1,
              fluency: 1,
              claims: [
                {
                  claim: `Judged call ${calls}`,
                  supported: true,
                  correct: true,
                  citations: ["e1"],
                  reason: "supported",
                },
              ],
            },
          ],
        }),
      );
      return { exitCode: 0, stdout: "", stderr: "", durationMs: 1 };
    };
    const client = new CodexJsonClient(runner);
    const bundle = checkpointBundle("Question?");
    const retrieval: RetrievalResult = {
      datasetId: bundle.id,
      frameworkId: "kontext-brain",
      queryId: "q1",
      status: "ok",
      evidence: [
        {
          id: "e1",
          sourceId: "source",
          text: "Evidence.",
          score: 1,
          rank: 1,
          metadata: {},
        },
      ],
      latencyMs: 1,
      inputTokens: 1,
      error: null,
      frameworkVersion: "v13",
      configDigest: "v13",
      answerPolicy: "supported-evidence-needs",
    };
    const answerDigest = answerInputDigest(
      DEFAULT_RAG_EVAL_MANIFEST,
      requiredRecord(bundle.queries, 0),
      retrieval,
    );
    const answer = (text: string): AnswerResult => ({
      datasetId: bundle.id,
      frameworkId: "kontext-brain",
      queryId: "q1",
      status: "ok",
      output: { answer: text, citations: ["e1"], abstained: false, abstentionReason: null },
      latencyMs: 1,
      inputTokens: 1,
      outputTokens: 1,
      error: null,
      inputDigest: answerDigest,
    });

    const first = await judgeAnswers(
      DEFAULT_RAG_EVAL_MANIFEST,
      bundle,
      "kontext-brain",
      [retrieval],
      [answer("First answer")],
      bundle.queries,
      directory,
      client,
    );
    const second = await judgeAnswers(
      DEFAULT_RAG_EVAL_MANIFEST,
      bundle,
      "kontext-brain",
      [retrieval],
      [answer("Changed answer")],
      bundle.queries,
      directory,
      client,
    );

    expect(calls).toBe(2);
    expect(first[0]?.inputDigest).toBeTruthy();
    expect(second[0]?.inputDigest).toBeTruthy();
    expect(second[0]?.inputDigest).not.toBe(first[0]?.inputDigest);
  });

  it("blocks judging stale answers without invoking the model", async () => {
    const directory = mkdtempSync(join(tmpdir(), "rag-eval-stale-answer-"));
    directories.push(directory);
    let calls = 0;
    const runner: CommandRunner = async () => {
      calls += 1;
      throw new Error("judge must not be invoked for stale answers");
    };
    const bundle = checkpointBundle("Question?");
    const retrieval: RetrievalResult = {
      datasetId: bundle.id,
      frameworkId: "kontext-brain",
      queryId: "q1",
      status: "ok",
      evidence: [
        {
          id: "e1",
          sourceId: "source",
          text: "Evidence.",
          score: 1,
          rank: 1,
          metadata: {},
        },
      ],
      latencyMs: 1,
      inputTokens: 1,
      error: null,
      frameworkVersion: "v13",
      configDigest: "v13",
      answerPolicy: "supported-evidence-needs",
    };
    const staleAnswer: AnswerResult = {
      datasetId: bundle.id,
      frameworkId: "kontext-brain",
      queryId: "q1",
      status: "ok",
      output: {
        answer: "Stale answer [e1]",
        citations: ["e1"],
        abstained: false,
        abstentionReason: null,
      },
      latencyMs: 1,
      inputTokens: 1,
      outputTokens: 1,
      error: null,
      inputDigest: "stale-answer-input",
    };

    const results = await judgeAnswers(
      DEFAULT_RAG_EVAL_MANIFEST,
      bundle,
      "kontext-brain",
      [retrieval],
      [staleAnswer],
      bundle.queries,
      directory,
      new CodexJsonClient(runner),
    );

    expect(calls).toBe(0);
    expect(results).toHaveLength(1);
    expect(results[0]?.status).toBe("blocked");
    expect(results[0]?.error).toMatch(/answer input digest mismatch/i);
  });

  it("binds retrieval dataset and framework identity into stage digests", () => {
    const bundle = checkpointBundle("Question?");
    const query = requiredRecord(bundle.queries, 0);
    const retrieval: RetrievalResult = {
      datasetId: bundle.id,
      frameworkId: "kontext-brain",
      queryId: query.id,
      status: "ok",
      evidence: [],
      latencyMs: 1,
      inputTokens: 1,
      error: null,
      frameworkVersion: "v13",
      configDigest: "v13",
    };
    const foreignRetrieval: RetrievalResult = {
      ...retrieval,
      datasetId: "graphrag-bench-novel",
      frameworkId: "vector-rag-reranker",
    };
    const answer: AnswerResult = {
      datasetId: bundle.id,
      frameworkId: "kontext-brain",
      queryId: query.id,
      status: "ok",
      output: { answer: "", citations: [], abstained: true, abstentionReason: "unsupported" },
      latencyMs: 1,
      inputTokens: 1,
      outputTokens: 1,
      error: null,
      inputDigest: answerInputDigest(DEFAULT_RAG_EVAL_MANIFEST, query, retrieval),
    };

    expect(answerInputDigest(DEFAULT_RAG_EVAL_MANIFEST, query, foreignRetrieval)).not.toBe(
      answer.inputDigest,
    );
    expect(judgeInputDigest(DEFAULT_RAG_EVAL_MANIFEST, query, foreignRetrieval, answer)).not.toBe(
      judgeInputDigest(DEFAULT_RAG_EVAL_MANIFEST, query, retrieval, answer),
    );
    expect(
      judgeInputDigest(DEFAULT_RAG_EVAL_MANIFEST, query, retrieval, {
        ...answer,
        datasetId: "graphrag-bench-novel",
        frameworkId: "vector-rag-reranker",
      }),
    ).not.toBe(judgeInputDigest(DEFAULT_RAG_EVAL_MANIFEST, query, retrieval, answer));
  });

  it("blocks self-consistent foreign retrieval and answer identities before judging", async () => {
    const directory = mkdtempSync(join(tmpdir(), "rag-eval-foreign-identity-"));
    directories.push(directory);
    let calls = 0;
    const runner: CommandRunner = async () => {
      calls += 1;
      throw new Error("judge must not be invoked for foreign identities");
    };
    const bundle = checkpointBundle("Question?");
    const query = requiredRecord(bundle.queries, 0);
    const retrieval: RetrievalResult = {
      datasetId: "graphrag-bench-novel",
      frameworkId: "vector-rag-reranker",
      queryId: query.id,
      status: "ok",
      evidence: [],
      latencyMs: 1,
      inputTokens: 1,
      error: null,
      frameworkVersion: "foreign",
      configDigest: "foreign",
    };
    const answer: AnswerResult = {
      datasetId: retrieval.datasetId,
      frameworkId: retrieval.frameworkId,
      queryId: query.id,
      status: "ok",
      output: { answer: "", citations: [], abstained: true, abstentionReason: "unsupported" },
      latencyMs: 1,
      inputTokens: 1,
      outputTokens: 1,
      error: null,
      inputDigest: answerInputDigest(DEFAULT_RAG_EVAL_MANIFEST, query, retrieval),
    };

    const results = await judgeAnswers(
      DEFAULT_RAG_EVAL_MANIFEST,
      bundle,
      "kontext-brain",
      [retrieval],
      [answer],
      bundle.queries,
      directory,
      new CodexJsonClient(runner),
    );

    expect(calls).toBe(0);
    expect(results[0]?.status).toBe("blocked");
    expect(results[0]?.error).toMatch(/identity mismatch/i);
  });

  it("blocks foreign retrieval identity before answering", async () => {
    const directory = mkdtempSync(join(tmpdir(), "rag-eval-foreign-retrieval-"));
    directories.push(directory);
    let calls = 0;
    const runner: CommandRunner = async () => {
      calls += 1;
      throw new Error("answer must not be invoked for foreign retrieval identity");
    };
    const bundle = checkpointBundle("Question?");
    const query = requiredRecord(bundle.queries, 0);
    const retrieval: RetrievalResult = {
      datasetId: "graphrag-bench-novel",
      frameworkId: "vector-rag-reranker",
      queryId: query.id,
      status: "ok",
      evidence: [],
      latencyMs: 1,
      inputTokens: 1,
      error: null,
      frameworkVersion: "foreign",
      configDigest: "foreign",
    };

    const results = await answerQueries(
      DEFAULT_RAG_EVAL_MANIFEST,
      bundle,
      "kontext-brain",
      [retrieval],
      bundle.queries,
      directory,
      new CodexJsonClient(runner),
    );

    expect(calls).toBe(0);
    expect(results[0]?.status).toBe("blocked");
    expect(results[0]?.error).toMatch(/retrieval identity mismatch/i);
  });
});

function outputMessagePath(args: readonly string[]): string {
  const outputIndex = args.indexOf("--output-last-message");
  const path = args[outputIndex + 1];
  if (!path) throw new Error("Codex command omitted --output-last-message path");
  return path;
}

function checkpointBundle(question: string): DatasetBundle {
  return {
    id: "graphrag-bench-medical",
    track: "static-kb",
    documents: [],
    queries: [
      {
        id: "q1",
        text: question,
        referenceAnswer: "reference",
        goldEvidenceIds: ["e1"],
        goldEvidenceText: ["Evidence."],
        answerable: true,
        category: "test",
        metadata: {},
      },
    ],
    provenance: { source: "test", version: "1", license: "test" },
  };
}

function requiredRecord<T>(records: readonly T[], index: number): T {
  const record = records[index];
  if (!record) throw new Error(`Missing test record ${index}`);
  return record;
}
