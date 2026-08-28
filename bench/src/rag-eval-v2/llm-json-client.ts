import { AnthropicJsonClient } from "./anthropic-json.js";
import {
  type AnswerBatchInput,
  type BatchItem,
  CodexJsonClient,
  type CodexJsonResult,
  type CodexModelConfig,
  type JudgeBatchInput,
} from "./codex-json.js";
import type {
  AnswerContract,
  AnswerPolicy,
  BenchmarkQuery,
  JudgeContract,
  RetrievedEvidence,
} from "./contracts.js";

export type JsonModelConfig = CodexModelConfig;

export type JsonLlmExecution = "codex-exec" | "anthropic-api";

/** Transport-agnostic contract for the structured LLM calls used by the benchmark. */
export interface JsonLlmClient {
  answer(
    model: CodexModelConfig,
    query: BenchmarkQuery,
    evidence: readonly RetrievedEvidence[],
    answerPolicy?: AnswerPolicy,
  ): Promise<CodexJsonResult<AnswerContract>>;
  judge(
    model: CodexModelConfig,
    query: BenchmarkQuery,
    evidence: readonly RetrievedEvidence[],
    answer: AnswerContract,
  ): Promise<CodexJsonResult<JudgeContract>>;
  answerBatch(
    model: CodexModelConfig,
    inputs: readonly AnswerBatchInput[],
  ): Promise<CodexJsonResult<readonly BatchItem<AnswerContract>[]>>;
  judgeBatch(
    model: CodexModelConfig,
    inputs: readonly JudgeBatchInput[],
  ): Promise<CodexJsonResult<readonly BatchItem<JudgeContract>[]>>;
  completeText(
    model: CodexModelConfig,
    systemPrompt: string,
    context: string,
    query: string,
  ): Promise<CodexJsonResult<string>>;
}

export function createJsonLlmClient(execution: JsonLlmExecution): JsonLlmClient {
  return execution === "anthropic-api" ? new AnthropicJsonClient() : new CodexJsonClient();
}
