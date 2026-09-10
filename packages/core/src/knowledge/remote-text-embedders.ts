import { type EmbeddingInputKind, type TextEmbedder, normalizeVector } from "./text-embedder.js";

/**
 * Embedders that call a server: a local Ollama or an OpenAI-compatible API.
 * Both speak plain HTTP, so they need no dependency, and both are what a
 * person picks when the built-in model is too slow or too small for them.
 */

type FetchLike = (input: string, init: RequestInit) => Promise<Response>;

const BATCH_SIZE = 32;

async function postJson(
  fetchImpl: FetchLike,
  url: string,
  body: unknown,
  headers: Record<string, string>,
): Promise<unknown> {
  let response: Response;
  try {
    response = await fetchImpl(url, {
      method: "POST",
      headers: { "content-type": "application/json", ...headers },
      body: JSON.stringify(body),
    });
  } catch (error) {
    // Why: "fetch failed" names nothing; the address tells a person what to start or fix.
    const cause = (error as { cause?: { code?: string } }).cause?.code;
    throw new Error(`Could not reach ${url}${cause ? ` (${cause})` : ""}; is the server running?`);
  }
  if (!response.ok) {
    const text = (await response.text().catch(() => "")).slice(0, 300);
    throw new Error(`${url} answered ${response.status}${text ? `: ${text}` : ""}`);
  }
  return response.json();
}

function toVectors(rows: unknown, expected: number, where: string): Float32Array[] {
  if (!Array.isArray(rows) || rows.length !== expected) {
    throw new Error(
      `${where} returned ${Array.isArray(rows) ? rows.length : "no"} embeddings for ${expected} texts`,
    );
  }
  return rows.map((row) => {
    if (!Array.isArray(row)) throw new Error(`${where} returned a non-numeric embedding`);
    return normalizeVector(Float32Array.from(row as number[]));
  });
}

async function inBatches(
  texts: readonly string[],
  embedBatch: (batch: readonly string[]) => Promise<Float32Array[]>,
): Promise<Float32Array[]> {
  const vectors: Float32Array[] = [];
  for (let start = 0; start < texts.length; start += BATCH_SIZE) {
    vectors.push(...(await embedBatch(texts.slice(start, start + BATCH_SIZE))));
  }
  return vectors;
}

export class OllamaTextEmbedder implements TextEmbedder {
  readonly model: string;
  private readonly baseUrl: string;
  private readonly fetchImpl: FetchLike;

  constructor(options: { model: string; baseUrl: string; fetch?: FetchLike }) {
    this.model = `ollama:${options.model}`;
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.fetchImpl = options.fetch ?? ((input, init) => fetch(input, init));
  }

  embed(texts: readonly string[], _kind: EmbeddingInputKind): Promise<readonly Float32Array[]> {
    const model = this.model.slice("ollama:".length);
    return inBatches(texts, async (batch) => {
      const body = (await postJson(
        this.fetchImpl,
        `${this.baseUrl}/api/embed`,
        { model, input: batch },
        {},
      )) as { embeddings?: unknown };
      return toVectors(body.embeddings, batch.length, "Ollama");
    });
  }
}

export class OpenAITextEmbedder implements TextEmbedder {
  readonly model: string;
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly fetchImpl: FetchLike;

  constructor(options: { model: string; baseUrl: string; apiKey: string; fetch?: FetchLike }) {
    this.model = `openai:${options.model}`;
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.fetchImpl = options.fetch ?? ((input, init) => fetch(input, init));
  }

  embed(texts: readonly string[], _kind: EmbeddingInputKind): Promise<readonly Float32Array[]> {
    const model = this.model.slice("openai:".length);
    return inBatches(texts, async (batch) => {
      const body = (await postJson(
        this.fetchImpl,
        `${this.baseUrl}/embeddings`,
        { model, input: batch },
        { authorization: `Bearer ${this.apiKey}` },
      )) as { data?: unknown };
      const rows = Array.isArray(body.data)
        ? [...(body.data as { index?: number; embedding?: unknown }[])]
            .sort((left, right) => (left.index ?? 0) - (right.index ?? 0))
            .map((row) => row.embedding)
        : undefined;
      return toVectors(rows, batch.length, "The embeddings API");
    });
  }
}
