import Anthropic from "@anthropic-ai/sdk";
import {
  ANSWER_SCHEMA,
  type AnswerBatchInput,
  type BatchItem,
  type CodexJsonResult,
  type CodexModelConfig,
  JUDGE_SCHEMA,
  type JudgeBatchInput,
  TEXT_SCHEMA,
  answerBatchPrompt,
  answerBatchSchema,
  answerPrompt,
  asObject,
  completeTextPrompt,
  judgeBatchPrompt,
  judgeBatchSchema,
  judgePrompt,
  parseAnswerContract,
  parseBatchResults,
  parseJudgeContract,
  parseSupportedEvidenceNeedsAnswer,
  policyAwareAnswerBatchSchema,
  stringValue,
  supportedEvidenceNeedsAnswerSchema,
  validateBatchInputs,
} from "./codex-json.js";
import type {
  AnswerContract,
  AnswerPolicy,
  BenchmarkQuery,
  JudgeContract,
  RetrievedEvidence,
} from "./contracts.js";
import type { JsonLlmClient } from "./llm-json-client.js";

export interface AnthropicTransportRequest {
  readonly model: string;
  readonly maxTokens: number;
  readonly prompt: string;
  readonly schema: unknown;
  readonly effort: CodexModelConfig["reasoningEffort"];
  readonly timeoutMs: number;
}

export interface AnthropicTransportResponse {
  readonly text: string;
  readonly inputTokens: number | null;
  readonly outputTokens: number | null;
  readonly latencyMs: number;
  readonly stopReason: string | null;
}

export type AnthropicTransport = (
  request: AnthropicTransportRequest,
) => Promise<AnthropicTransportResponse>;

const MAX_OUTPUT_TOKENS = 16_000;
const DEFAULT_TIMEOUT_MS = 180_000;

const UNSUPPORTED_SCHEMA_KEYWORDS = new Set([
  "minimum",
  "maximum",
  "multipleOf",
  "minLength",
  "maxLength",
  "minItems",
  "maxItems",
]);

/**
 * Removes JSON Schema constraint keywords that the Anthropic structured-output
 * compiler rejects. Range and shape validation still happens client-side in the
 * shared parse functions from codex-json.ts.
 */
export function stripUnsupportedSchemaKeywords(schema: unknown): unknown {
  if (Array.isArray(schema)) return schema.map(stripUnsupportedSchemaKeywords);
  if (!schema || typeof schema !== "object") return schema;
  return Object.fromEntries(
    Object.entries(schema)
      .filter(([key]) => !UNSUPPORTED_SCHEMA_KEYWORDS.has(key))
      .map(([key, value]) => [key, stripUnsupportedSchemaKeywords(value)]),
  );
}

/**
 * Anthropic Messages API transport with the exact prompts, schemas, and parsers
 * of the local `codex exec` backend.
 */
export class AnthropicJsonClient implements JsonLlmClient {
  private readonly transport: AnthropicTransport;
  private sdkClient: Anthropic | null = null;

  constructor(transport?: AnthropicTransport) {
    this.transport = transport ?? ((request) => this.sdkTransport(request));
  }

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
    const result = await this.execute(
      model,
      completeTextPrompt(systemPrompt, context, query),
      TEXT_SCHEMA,
    );
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
    const response = await this.transport({
      model: model.model,
      maxTokens: MAX_OUTPUT_TOKENS,
      prompt,
      schema: stripUnsupportedSchemaKeywords(schema),
      effort: model.reasoningEffort,
      timeoutMs: model.timeoutMs ?? DEFAULT_TIMEOUT_MS,
    });
    if (response.stopReason === "refusal" || response.stopReason === "max_tokens") {
      throw new Error(`Anthropic request stopped with stop_reason ${response.stopReason}`);
    }
    return {
      value: JSON.parse(response.text) as unknown,
      latencyMs: response.latencyMs,
      inputTokens: response.inputTokens,
      outputTokens: response.outputTokens,
    };
  }

  private async sdkTransport(
    request: AnthropicTransportRequest,
  ): Promise<AnthropicTransportResponse> {
    this.sdkClient ??= new Anthropic({ maxRetries: 0 });
    const startedAt = performance.now();
    const response = await this.sdkClient.messages.create(
      {
        model: request.model,
        max_tokens: request.maxTokens,
        messages: [{ role: "user", content: request.prompt }],
        output_config: {
          effort: request.effort,
          format: { type: "json_schema", schema: request.schema },
        },
      } as Anthropic.MessageCreateParamsNonStreaming,
      { timeout: request.timeoutMs, maxRetries: 0 },
    );
    const latencyMs = performance.now() - startedAt;
    return {
      text: response.content
        .filter((block): block is Anthropic.TextBlock => block.type === "text")
        .map((block) => block.text)
        .join(""),
      inputTokens: response.usage?.input_tokens ?? null,
      outputTokens: response.usage?.output_tokens ?? null,
      latencyMs,
      stopReason: response.stop_reason ?? null,
    };
  }
}
