import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type {
  AnswerContract,
  AnswerPolicy,
  BenchmarkQuery,
  JudgeContract,
  RetrievedEvidence,
} from "./contracts.js";

export interface CodexModelConfig {
  readonly model: string;
  readonly reasoningEffort: "low" | "medium" | "high" | "xhigh";
  readonly timeoutMs?: number;
}

export interface CommandResult {
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
  readonly durationMs: number;
}

export type CommandRunner = (
  command: string,
  args: readonly string[],
  stdin: string,
  timeoutMs: number,
  environment?: NodeJS.ProcessEnv,
) => Promise<CommandResult>;

const PROVIDER_API_KEY_ENVIRONMENT_VARIABLES = [
  "OPENAI_API_KEY",
  "CODEX_API_KEY",
  "ANTHROPIC_API_KEY",
  "GEMINI_API_KEY",
  "GOOGLE_API_KEY",
  "AZURE_OPENAI_API_KEY",
  "NVIDIA_API_KEY",
] as const;

export function codexCliEnvironment(
  baseEnvironment: NodeJS.ProcessEnv = process.env,
): NodeJS.ProcessEnv {
  const environment = { ...baseEnvironment };
  for (const name of PROVIDER_API_KEY_ENVIRONMENT_VARIABLES) delete environment[name];
  return environment;
}

export interface CodexJsonResult<T> {
  readonly value: T;
  readonly latencyMs: number;
  readonly inputTokens: number | null;
  readonly outputTokens: number | null;
}

export interface AnswerBatchInput {
  readonly query: BenchmarkQuery;
  readonly evidence: readonly RetrievedEvidence[];
  readonly answerPolicy?: AnswerPolicy;
}

export interface JudgeBatchInput extends AnswerBatchInput {
  readonly answer: AnswerContract;
}

export interface BatchItem<T> {
  readonly queryId: string;
  readonly value: T;
}

const ANSWER_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["answer", "citations", "abstained", "abstention_reason"],
  properties: {
    answer: { type: "string" },
    citations: { type: "array", items: { type: "string" } },
    abstained: { type: "boolean" },
    abstention_reason: { anyOf: [{ type: "string" }, { type: "null" }] },
  },
} as const;

const JUDGE_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: [
    "answer_correctness",
    "completeness",
    "strict_faithfulness",
    "citation_precision",
    "citation_recall",
    "acceptable_abstention",
    "clarity",
    "conciseness",
    "fluency",
    "claims",
  ],
  properties: {
    answer_correctness: { type: "number", minimum: 0, maximum: 1 },
    completeness: { type: "number", minimum: 0, maximum: 1 },
    strict_faithfulness: { type: "number", minimum: 0, maximum: 1 },
    citation_precision: { type: "number", minimum: 0, maximum: 1 },
    citation_recall: { type: "number", minimum: 0, maximum: 1 },
    acceptable_abstention: { type: "boolean" },
    clarity: { type: "number", minimum: 0, maximum: 1 },
    conciseness: { type: "number", minimum: 0, maximum: 1 },
    fluency: { type: "number", minimum: 0, maximum: 1 },
    claims: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["claim", "supported", "correct", "citations", "reason"],
        properties: {
          claim: { type: "string" },
          supported: { type: "boolean" },
          correct: { type: "boolean" },
          citations: { type: "array", items: { type: "string" } },
          reason: { type: "string" },
        },
      },
    },
  },
} as const;

const TEXT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["text"],
  properties: { text: { type: "string" } },
} as const;

export class CodexJsonClient {
  constructor(private readonly commandRunner: CommandRunner = runCommand) {}

  async answer(
    model: CodexModelConfig,
    query: BenchmarkQuery,
    evidence: readonly RetrievedEvidence[],
    answerPolicy?: AnswerPolicy,
  ): Promise<CodexJsonResult<AnswerContract>> {
    const supportedNeeds = answerPolicy === "supported-evidence-needs";
    const result = await this.execute(
      model,
      answerPrompt(query, evidence, answerPolicy),
      supportedNeeds ? supportedEvidenceNeedsAnswerSchema(evidence) : ANSWER_SCHEMA,
    );
    return {
      value: supportedNeeds
        ? parseSupportedEvidenceNeedsAnswer(result.value, evidence)
        : parseAnswerContract(result.value),
      latencyMs: result.latencyMs,
      inputTokens: result.inputTokens,
      outputTokens: result.outputTokens,
    };
  }

  async judge(
    model: CodexModelConfig,
    query: BenchmarkQuery,
    evidence: readonly RetrievedEvidence[],
    answer: AnswerContract,
  ): Promise<CodexJsonResult<JudgeContract>> {
    const result = await this.execute(model, judgePrompt(query, evidence, answer), JUDGE_SCHEMA);
    return {
      value: parseJudgeContract(result.value),
      latencyMs: result.latencyMs,
      inputTokens: result.inputTokens,
      outputTokens: result.outputTokens,
    };
  }

  async answerBatch(
    model: CodexModelConfig,
    inputs: readonly AnswerBatchInput[],
  ): Promise<CodexJsonResult<readonly BatchItem<AnswerContract>[]>> {
    const queryIds = validateBatchInputs(inputs);
    const inputByQueryId = new Map(inputs.map((input) => [input.query.id, input]));
    const hasSupportedNeedsPolicy = inputs.some(
      (input) => input.answerPolicy === "supported-evidence-needs",
    );
    const result = await this.execute(
      model,
      answerBatchPrompt(inputs),
      hasSupportedNeedsPolicy ? policyAwareAnswerBatchSchema(inputs) : answerBatchSchema(queryIds),
    );
    return {
      value: parseBatchResults(result.value, queryIds, (item, queryId) => {
        const input = inputByQueryId.get(queryId);
        if (!input) throw new Error(`Missing answer input for ${queryId}`);
        return input.answerPolicy === "supported-evidence-needs"
          ? parseSupportedEvidenceNeedsAnswer(item, input.evidence)
          : parseAnswerContract(item);
      }),
      latencyMs: result.latencyMs,
      inputTokens: result.inputTokens,
      outputTokens: result.outputTokens,
    };
  }

  async judgeBatch(
    model: CodexModelConfig,
    inputs: readonly JudgeBatchInput[],
  ): Promise<CodexJsonResult<readonly BatchItem<JudgeContract>[]>> {
    const queryIds = validateBatchInputs(inputs);
    const result = await this.execute(model, judgeBatchPrompt(inputs), judgeBatchSchema(queryIds));
    return {
      value: parseBatchResults(result.value, queryIds, parseJudgeContract),
      latencyMs: result.latencyMs,
      inputTokens: result.inputTokens,
      outputTokens: result.outputTokens,
    };
  }

  async completeText(
    model: CodexModelConfig,
    systemPrompt: string,
    context: string,
    query: string,
  ): Promise<CodexJsonResult<string>> {
    const prompt = [
      "Act as a deterministic text completion backend for the embedded request below.",
      "Do not use tools, files, web search, or prior conversation.",
      "Put the exact completion that the embedded request asks for in the JSON field `text`.",
      "If it requests JSON, `text` must itself contain valid JSON with no markdown fence.",
      "",
      "<system>",
      systemPrompt,
      "</system>",
      "<context>",
      context,
      "</context>",
      "<query>",
      query,
      "</query>",
    ].join("\n");
    const result = await this.execute(model, prompt, TEXT_SCHEMA);
    const object = asObject(result.value, "text completion");
    return {
      value: stringValue(object.text, "text"),
      latencyMs: result.latencyMs,
      inputTokens: result.inputTokens,
      outputTokens: result.outputTokens,
    };
  }

  private async execute(
    model: CodexModelConfig,
    prompt: string,
    schema: unknown,
  ): Promise<CodexJsonResult<unknown>> {
    const temporaryDirectory = mkdtempSync(join(tmpdir(), "kontext-rag-eval-"));
    const schemaPath = join(temporaryDirectory, "schema.json");
    const outputPath = join(temporaryDirectory, "output.json");
    writeFileSync(schemaPath, JSON.stringify(schema), "utf8");
    const args = [
      "exec",
      "--ephemeral",
      "--ignore-user-config",
      "--ignore-rules",
      "--disable",
      "plugins",
      "--disable",
      "remote_plugin",
      "--disable",
      "plugin_sharing",
      "--disable",
      "apps",
      "--disable",
      "browser_use",
      "--disable",
      "browser_use_external",
      "--disable",
      "browser_use_full_cdp_access",
      "--disable",
      "in_app_browser",
      "--disable",
      "skill_mcp_dependency_install",
      "--skip-git-repo-check",
      "--sandbox",
      "read-only",
      "--json",
      "--model",
      model.model,
      "--config",
      `model_reasoning_effort=${JSON.stringify(model.reasoningEffort)}`,
      "--output-schema",
      schemaPath,
      "--output-last-message",
      outputPath,
      "-",
    ];
    try {
      const commandResult = await this.commandRunner(
        "codex",
        args,
        prompt,
        model.timeoutMs ?? 180_000,
        codexCliEnvironment(),
      );
      if (commandResult.exitCode !== 0) {
        const diagnostic = [commandResult.stderr.trim(), commandResult.stdout.trim()]
          .filter(Boolean)
          .join("\n");
        throw new Error(`codex exec failed with exit ${commandResult.exitCode}: ${diagnostic}`);
      }
      const rawOutput = readFileSync(outputPath, "utf8");
      const usage = extractLastUsage(commandResult.stdout);
      return {
        value: JSON.parse(rawOutput) as unknown,
        latencyMs: commandResult.durationMs,
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
      };
    } finally {
      rmSync(temporaryDirectory, { recursive: true, force: true });
    }
  }
}

function answerBatchSchema(queryIds: readonly string[]): unknown {
  return {
    type: "object",
    additionalProperties: false,
    required: ["results"],
    properties: {
      results: {
        type: "array",
        minItems: queryIds.length,
        maxItems: queryIds.length,
        items: {
          type: "object",
          additionalProperties: false,
          required: ["query_id", ...ANSWER_SCHEMA.required],
          properties: {
            query_id: { type: "string", enum: queryIds },
            ...ANSWER_SCHEMA.properties,
          },
        },
      },
    },
  };
}

function supportedEvidenceNeedsAnswerSchema(
  evidence: readonly RetrievedEvidence[],
): Record<string, unknown> {
  const evidenceIds = [...new Set(evidence.map((item) => item.id))];
  return {
    type: "object",
    additionalProperties: false,
    required: ["claims", "abstained", "abstention_reason"],
    properties: {
      claims: {
        type: "array",
        maxItems: 8,
        items: {
          type: "object",
          additionalProperties: false,
          required: ["claim", "citation"],
          properties: {
            claim: { type: "string", minLength: 1 },
            citation:
              evidenceIds.length > 0 ? { type: "string", enum: evidenceIds } : { type: "string" },
          },
        },
      },
      abstained: { type: "boolean" },
      abstention_reason: { anyOf: [{ type: "string" }, { type: "null" }] },
    },
  };
}

function policyAwareAnswerBatchSchema(inputs: readonly AnswerBatchInput[]): unknown {
  const itemSchemas = inputs.map((input) => {
    const answerSchema =
      input.answerPolicy === "supported-evidence-needs"
        ? supportedEvidenceNeedsAnswerSchema(input.evidence)
        : ANSWER_SCHEMA;
    const required = answerSchema.required;
    const properties = answerSchema.properties;
    if (!Array.isArray(required) || !properties || typeof properties !== "object") {
      throw new Error("Answer schema must declare required fields and properties");
    }
    return {
      type: "object",
      additionalProperties: false,
      required: ["query_id", ...required],
      properties: {
        query_id: { type: "string", enum: [input.query.id] },
        ...properties,
      },
    };
  });
  return {
    type: "object",
    additionalProperties: false,
    required: ["results"],
    properties: {
      results: {
        type: "array",
        minItems: inputs.length,
        maxItems: inputs.length,
        items: itemSchemas.length === 1 ? itemSchemas[0] : { anyOf: itemSchemas },
      },
    },
  };
}

function judgeBatchSchema(queryIds: readonly string[]): unknown {
  return {
    type: "object",
    additionalProperties: false,
    required: ["results"],
    properties: {
      results: {
        type: "array",
        minItems: queryIds.length,
        maxItems: queryIds.length,
        items: {
          type: "object",
          additionalProperties: false,
          required: ["query_id", ...JUDGE_SCHEMA.required],
          properties: {
            query_id: { type: "string", enum: queryIds },
            ...JUDGE_SCHEMA.properties,
          },
        },
      },
    },
  };
}

function validateBatchInputs(inputs: readonly AnswerBatchInput[]): string[] {
  if (inputs.length === 0) throw new Error("Codex batch cannot be empty");
  const queryIds = inputs.map((input) => input.query.id);
  if (new Set(queryIds).size !== queryIds.length)
    throw new Error("Codex batch query IDs must be unique");
  return queryIds;
}

function parseBatchResults<T>(
  value: unknown,
  queryIds: readonly string[],
  parse: (item: unknown, queryId: string) => T,
): BatchItem<T>[] {
  const object = asObject(value, "batch");
  if (!Array.isArray(object.results)) throw new Error("batch results must be an array");
  const byQuery = new Map<string, T>();
  for (const [index, raw] of object.results.entries()) {
    const item = asObject(raw, `results[${index}]`);
    const queryId = stringValue(item.query_id, `results[${index}].query_id`);
    if (!queryIds.includes(queryId)) throw new Error(`Unexpected batch query ID ${queryId}`);
    if (byQuery.has(queryId)) throw new Error(`Duplicate batch query ID ${queryId}`);
    byQuery.set(queryId, parse(item, queryId));
  }
  const missing = queryIds.filter((queryId) => !byQuery.has(queryId));
  if (missing.length > 0) throw new Error(`Missing batch query IDs: ${missing.join(", ")}`);
  return queryIds.map((queryId) => ({ queryId, value: byQuery.get(queryId)! }));
}

function answerPrompt(
  query: BenchmarkQuery,
  evidence: readonly RetrievedEvidence[],
  answerPolicy?: AnswerPolicy,
): string {
  return [
    "Answer the benchmark question using only the supplied evidence.",
    "Every factual claim must cite one or more exact evidence IDs from the supplied list.",
    "If the evidence is insufficient, abstain. Do not use tools, files, web search, or prior knowledge.",
    ...answerPolicyInstructions(answerPolicy),
    "Return only the JSON object required by the output schema.",
    "",
    `Question: ${query.text}`,
    "",
    "Evidence:",
    ...evidence.map((item) => `[${item.id}] ${item.text}`),
  ].join("\n");
}

function answerBatchPrompt(inputs: readonly AnswerBatchInput[]): string {
  return [
    "Answer every benchmark case independently using only that case's supplied evidence.",
    "Never transfer facts, citations, or conclusions between cases.",
    "Every factual claim must cite exact evidence IDs from its own case.",
    "If a case's evidence is insufficient, abstain. Do not use tools, files, web search, or prior knowledge.",
    "Return one result for every query_id and only the JSON object required by the output schema.",
    "",
    ...inputs.flatMap(({ query, evidence, answerPolicy }) => [
      `<case query_id=${JSON.stringify(query.id)}>`,
      `Question: ${query.text}`,
      ...answerPolicyInstructions(answerPolicy),
      "Evidence:",
      ...evidence.map((item) => `[${item.id}] ${item.text}`),
      "</case>",
      "",
    ]),
  ].join("\n");
}

function answerPolicyInstructions(answerPolicy: AnswerPolicy | undefined): string[] {
  if (answerPolicy !== "supported-evidence-needs") return [];
  return [
    "Internally identify the distinct evidence needs in the question using only the question and retrieved evidence.",
    "At most one atomic claim for each supported evidence need may be included; add exactly one best supporting evidence ID for that claim in citations.",
    "Return each supported atomic claim as one claims item with exactly claim and citation fields; do not add citation markers to claim text.",
    "Use no more than eight factual claims, and remove redundant claims.",
    "If only some evidence needs are supported, answer only the supported parts and omit unsupported needs; abstain only when none are supported.",
  ];
}

function judgePrompt(
  query: BenchmarkQuery,
  evidence: readonly RetrievedEvidence[],
  answer: AnswerContract,
): string {
  return [
    "Evaluate the candidate answer strictly and independently.",
    "Split only the literal semantic content of the candidate answer into atomic claims; never add detail from the reference answer.",
    "A claim is supported only when the cited evidence entails the complete claim",
    "including names, dates, quantities, and negation. Do not use tools, files, web search, or prior knowledge.",
    "Set completeness to the fraction of necessary reference-answer claims covered by the candidate; this is Claim Recall.",
    "Set strict_faithfulness from complete evidence entailment of every candidate claim, and mark each claim.supported independently for Claim Support Precision.",
    "Citation precision measures whether cited evidence supports its attached claim; citation recall measures whether every claim needing support has a valid citation.",
    "For unanswerable questions, acceptable_abstention is true only when the candidate appropriately abstains.",
    "Score clarity for understandable organization, conciseness for avoiding unnecessary or redundant wording without penalizing required coverage, and fluency for grammatical naturalness.",
    "Return only the JSON object required by the output schema.",
    "",
    `Question: ${query.text}`,
    `Answerable: ${query.answerable}`,
    `Reference answer: ${query.referenceAnswer ?? "<none>"}`,
    `Gold evidence text: ${JSON.stringify(query.goldEvidenceText)}`,
    `Candidate answer: ${JSON.stringify(answer)}`,
    "",
    "Retrieved evidence:",
    ...evidence.map((item) => `[${item.id}] ${item.text}`),
  ].join("\n");
}

function judgeBatchPrompt(inputs: readonly JudgeBatchInput[]): string {
  return [
    "Evaluate every candidate case strictly and independently.",
    "Never transfer facts, evidence, or judgements between cases.",
    "Split only the literal semantic content of each candidate answer into atomic claims.",
    "A claim is supported only when that case's cited evidence entails the complete claim, including names, dates, quantities, and negation.",
    "Set completeness to the fraction of necessary reference-answer claims covered by the candidate; this is Claim Recall.",
    "Set strict_faithfulness from complete evidence entailment of every candidate claim, and mark each claim.supported independently for Claim Support Precision.",
    "Citation precision measures whether cited evidence supports its attached claim; citation recall measures whether every claim needing support has a valid citation.",
    "For unanswerable questions, acceptable_abstention is true only when the candidate appropriately abstains.",
    "Score clarity for understandable organization, conciseness for avoiding unnecessary or redundant wording without penalizing required coverage, and fluency for grammatical naturalness.",
    "Do not use tools, files, web search, or prior knowledge.",
    "Return one result for every query_id and only the JSON object required by the output schema.",
    "",
    ...inputs.flatMap(({ query, evidence, answer }) => [
      `<case query_id=${JSON.stringify(query.id)}>`,
      `Question: ${query.text}`,
      `Answerable: ${query.answerable}`,
      `Reference answer: ${query.referenceAnswer ?? "<none>"}`,
      `Gold evidence text: ${JSON.stringify(query.goldEvidenceText)}`,
      `Candidate answer: ${JSON.stringify(answer)}`,
      "Retrieved evidence:",
      ...evidence.map((item) => `[${item.id}] ${item.text}`),
      "</case>",
      "",
    ]),
  ].join("\n");
}

function parseAnswerContract(value: unknown): AnswerContract {
  const object = asObject(value, "answer");
  const citations = stringArray(object.citations, "citations");
  const abstained = booleanValue(object.abstained, "abstained");
  const answer = stringValue(object.answer, "answer");
  const reason = nullableString(object.abstention_reason, "abstention_reason");
  if (abstained && reason === null) throw new Error("Abstained answers require abstention_reason");
  if (!abstained && answer.trim().length === 0)
    throw new Error("Non-abstained answers require answer text");
  return { answer, citations, abstained, abstentionReason: reason };
}

function parseSupportedEvidenceNeedsAnswer(
  value: unknown,
  evidence: readonly RetrievedEvidence[],
): AnswerContract {
  const object = asObject(value, "supported evidence-needs answer");
  assertOnlyProperties(
    object,
    ["query_id", "claims", "abstained", "abstention_reason"],
    "supported evidence-needs answer",
  );
  const claimsValue = object.claims;
  if (!Array.isArray(claimsValue)) throw new Error("claims must be an array");
  if (claimsValue.length > 8) throw new Error("claims must contain at most eight items");
  const abstained = booleanValue(object.abstained, "abstained");
  const reason = nullableString(object.abstention_reason, "abstention_reason");
  if (claimsValue.length === 0) {
    if (!abstained) throw new Error("Zero supported claims require abstention");
    if (reason === null || reason.trim().length === 0) {
      throw new Error("Abstention requires a non-empty abstention_reason");
    }
    return { answer: "", citations: [], abstained: true, abstentionReason: reason.trim() };
  }
  if (abstained) throw new Error("Supported claims cannot be paired with abstention");
  if (reason !== null) throw new Error("Non-abstained answers require a null abstention_reason");

  const allowedEvidenceIds = new Set(evidence.map((item) => item.id));
  const seenClaims = new Set<string>();
  const claims = claimsValue.map((rawClaim, index) => {
    const item = asObject(rawClaim, `claims[${index}]`);
    assertOnlyProperties(item, ["claim", "citation"], `claims[${index}]`);
    const claim = stringValue(item.claim, `claims[${index}].claim`).trim();
    if (claim.length === 0) throw new Error(`claims[${index}].claim must not be empty`);
    if (/\r|\n/.test(claim)) {
      throw new Error(`claims[${index}].claim must be one atomic line`);
    }
    const claimKey = normalizedClaimKey(claim);
    if (seenClaims.has(claimKey)) throw new Error(`Duplicate or redundant claim at index ${index}`);
    seenClaims.add(claimKey);

    const citation = stringValue(item.citation, `claims[${index}].citation`);
    if (!allowedEvidenceIds.has(citation)) {
      throw new Error(`claims[${index}].citation is not supplied evidence: ${citation}`);
    }
    return { claim, citation };
  });
  return {
    answer: claims.map((item) => `${item.claim} [${item.citation}]`).join("\n"),
    citations: [...new Set(claims.map((item) => item.citation))],
    abstained: false,
    abstentionReason: null,
  };
}

function normalizedClaimKey(value: string): string {
  return value
    .toLocaleLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[.!?]+$/, "")
    .trim();
}

function parseJudgeContract(value: unknown): JudgeContract {
  const object = asObject(value, "judge");
  const claimsValue = object.claims;
  if (!Array.isArray(claimsValue)) throw new Error("claims must be an array");
  return {
    answerCorrectness: scoreValue(object.answer_correctness, "answer_correctness"),
    completeness: scoreValue(object.completeness, "completeness"),
    strictFaithfulness: scoreValue(object.strict_faithfulness, "strict_faithfulness"),
    citationPrecision: scoreValue(object.citation_precision, "citation_precision"),
    citationRecall: scoreValue(object.citation_recall, "citation_recall"),
    acceptableAbstention: booleanValue(object.acceptable_abstention, "acceptable_abstention"),
    clarity: scoreValue(object.clarity, "clarity"),
    conciseness: scoreValue(object.conciseness, "conciseness"),
    fluency: scoreValue(object.fluency, "fluency"),
    claims: claimsValue.map((claim, index) => {
      const item = asObject(claim, `claims[${index}]`);
      return {
        claim: stringValue(item.claim, `claims[${index}].claim`),
        supported: booleanValue(item.supported, `claims[${index}].supported`),
        correct: booleanValue(item.correct, `claims[${index}].correct`),
        citations: stringArray(item.citations, `claims[${index}].citations`),
        reason: stringValue(item.reason, `claims[${index}].reason`),
      };
    }),
  };
}

function asObject(value: unknown, name: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error(`${name} must be an object`);
  return value as Record<string, unknown>;
}

function assertOnlyProperties(
  value: Readonly<Record<string, unknown>>,
  allowed: readonly string[],
  name: string,
): void {
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unknown.length > 0)
    throw new Error(`${name} has unexpected properties: ${unknown.join(", ")}`);
}

function stringValue(value: unknown, name: string): string {
  if (typeof value !== "string") throw new Error(`${name} must be a string`);
  return value;
}

function nullableString(value: unknown, name: string): string | null {
  if (value === null) return null;
  return stringValue(value, name);
}

function booleanValue(value: unknown, name: string): boolean {
  if (typeof value !== "boolean") throw new Error(`${name} must be a boolean`);
  return value;
}

function stringArray(value: unknown, name: string): string[] {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    throw new Error(`${name} must be a string array`);
  }
  return [...new Set(value as string[])];
}

function scoreValue(value: unknown, name: string): number {
  if (typeof value !== "number" || value < 0 || value > 1 || !Number.isFinite(value)) {
    throw new Error(`${name} must be a number in [0, 1]`);
  }
  return value;
}

export async function runCommand(
  command: string,
  args: readonly string[],
  stdin: string,
  timeoutMs: number,
  environment?: NodeJS.ProcessEnv,
): Promise<CommandResult> {
  const startedAt = performance.now();
  return await new Promise((resolve, reject) => {
    const child = spawn(command, [...args], {
      stdio: ["pipe", "pipe", "pipe"],
      ...(environment ? { env: environment } : {}),
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let settled = false;
    let forceKillTimeout: NodeJS.Timeout | undefined;
    const timeoutError = (): Error => {
      const diagnostic = [stderr.trim(), stdout.trim()].filter(Boolean).join("\n");
      const suffix = diagnostic ? `\nLast command output:\n${diagnostic.slice(-2_000)}` : "";
      return new Error(`${command} timed out after ${timeoutMs}ms${suffix}`);
    };
    const settle = (operation: () => void): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      if (forceKillTimeout) clearTimeout(forceKillTimeout);
      operation();
    };
    const timeout = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
      forceKillTimeout = setTimeout(() => {
        child.kill("SIGKILL");
        settle(() => reject(timeoutError()));
      }, 2_000);
    }, timeoutMs);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => (stdout += chunk));
    child.stderr.on("data", (chunk: string) => (stderr += chunk));
    child.on("error", (error) => {
      settle(() => reject(error));
    });
    child.on("close", (code) => {
      if (timedOut) {
        settle(() => reject(timeoutError()));
        return;
      }
      settle(() =>
        resolve({
          exitCode: code ?? 1,
          stdout,
          stderr,
          durationMs: performance.now() - startedAt,
        }),
      );
    });
    child.stdin.end(stdin);
  });
}

function extractLastUsage(stdout: string): {
  inputTokens: number | null;
  outputTokens: number | null;
} {
  let last: { inputTokens: number | null; outputTokens: number | null } = {
    inputTokens: null,
    outputTokens: null,
  };
  for (const line of stdout.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const parsed = JSON.parse(line) as unknown;
      const usage = findUsage(parsed);
      if (usage.inputTokens !== null || usage.outputTokens !== null) last = usage;
    } catch {
      // Non-JSON diagnostic output does not invalidate the structured final response.
    }
  }
  return last;
}

function findUsage(value: unknown): { inputTokens: number | null; outputTokens: number | null } {
  if (!value || typeof value !== "object") return { inputTokens: null, outputTokens: null };
  const object = value as Record<string, unknown>;
  const input = numericField(object, ["input_tokens", "inputTokens"]);
  const output = numericField(object, ["output_tokens", "outputTokens"]);
  if (input !== null || output !== null) return { inputTokens: input, outputTokens: output };
  for (const nested of Object.values(object)) {
    const usage = findUsage(nested);
    if (usage.inputTokens !== null || usage.outputTokens !== null) return usage;
  }
  return { inputTokens: null, outputTokens: null };
}

function numericField(object: Record<string, unknown>, names: readonly string[]): number | null {
  for (const name of names) {
    const value = object[name];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}
