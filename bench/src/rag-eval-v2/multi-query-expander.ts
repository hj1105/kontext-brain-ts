import type { CodexModelConfig } from "./codex-json.js";
import type { JsonLlmClient } from "./llm-json-client.js";

const MAX_EXPANDED_QUERIES = 3;

interface MultiQueryResponse {
  readonly queries: readonly string[];
}

export interface MultiQueryExpansion {
  readonly queries: readonly string[];
  readonly latencyMs: number;
  readonly inputTokens: number | null;
  readonly outputTokens: number | null;
  readonly error: string | null;
}

/**
 * Produces search-only perspectives from the user question. The original
 * question is intentionally added by the caller so it can never be displaced.
 */
export class MultiQueryExpander {
  constructor(
    private readonly client: Pick<JsonLlmClient, "completeText">,
    private readonly model: CodexModelConfig,
  ) {}

  async expand(question: string): Promise<MultiQueryExpansion> {
    let lastError: Error | null = null;
    let latencyMs = 0;
    let inputTokens = 0;
    let outputTokens = 0;
    let hasInputUsage = true;
    let hasOutputUsage = true;
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      try {
        const response = await this.client.completeText(
          this.model,
          [
            "Generate up to three complementary standalone retrieval queries for the question.",
            "The queries should cover different evidence needs or bridge steps that a source corpus must literally support.",
            "Preserve named entities, relationships, dates, comparisons, and constraints from the question.",
            "Do not answer the question, guess an unknown entity, introduce new facts, or mention evaluation data.",
            'Return JSON only in this exact shape: {"queries":["query", ...]}.',
          ].join(" "),
          "",
          question,
        );
        latencyMs += response.latencyMs;
        if (response.inputTokens === null) hasInputUsage = false;
        else inputTokens += response.inputTokens;
        if (response.outputTokens === null) hasOutputUsage = false;
        else outputTokens += response.outputTokens;
        return {
          queries: parseQueries(response.value, question),
          latencyMs,
          inputTokens: hasInputUsage ? inputTokens : null,
          outputTokens: hasOutputUsage ? outputTokens : null,
          error: null,
        };
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
      }
    }
    return {
      queries: [],
      latencyMs,
      inputTokens: hasInputUsage ? inputTokens : null,
      outputTokens: hasOutputUsage ? outputTokens : null,
      error: lastError?.message ?? "Multi-query expansion failed without an error",
    };
  }
}

function parseQueries(value: string, originalQuestion: string): string[] {
  const parsed = JSON.parse(value) as unknown;
  if (
    !parsed ||
    typeof parsed !== "object" ||
    !Array.isArray((parsed as MultiQueryResponse).queries)
  ) {
    throw new Error("Multi-query expansion must return a queries array");
  }
  if ((parsed as MultiQueryResponse).queries.some((query) => typeof query !== "string")) {
    throw new Error("Multi-query expansion queries must contain only strings");
  }
  const originalKey = normalizedKey(originalQuestion);
  const seen = new Set<string>([originalKey]);
  const queries: string[] = [];
  for (const rawQuery of (parsed as MultiQueryResponse).queries) {
    const query = rawQuery.replace(/\s+/g, " ").trim();
    const key = normalizedKey(query);
    if (!query || query.length > 500 || seen.has(key)) continue;
    seen.add(key);
    queries.push(query);
    if (queries.length >= MAX_EXPANDED_QUERIES) break;
  }
  if (queries.length === 0) {
    throw new Error("Multi-query expansion returned no distinct usable queries");
  }
  return queries;
}

function normalizedKey(value: string): string {
  return value.toLocaleLowerCase().replace(/\s+/g, " ").trim();
}
