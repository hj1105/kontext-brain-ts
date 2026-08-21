export type EmbeddingTask = "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY";

export interface EmbeddingInput {
  readonly id: string;
  readonly text: string;
  readonly title?: string;
}

export interface EmbeddingOutput {
  readonly id: string;
  readonly values: readonly number[];
}

export interface EmbeddingUsage {
  readonly requests: number;
  readonly inputTokens: number;
  readonly totalTokens: number;
}

export interface EmbeddingClient {
  readonly model: string;
  readonly dimensions: number;
  embed(inputs: readonly EmbeddingInput[], task: EmbeddingTask): Promise<EmbeddingOutput[]>;
  getUsage(): EmbeddingUsage;
}

interface OpenAIEmbeddingResponse {
  readonly data?: readonly {
    readonly index?: number;
    readonly embedding?: readonly number[];
  }[];
  readonly model?: string;
  readonly usage?: {
    readonly prompt_tokens?: number;
    readonly total_tokens?: number;
  };
  readonly error?: { readonly message?: string };
}

export interface OpenAIEmbeddingClientOptions {
  readonly apiKey: string;
  readonly model?: string;
  readonly dimensions?: number;
  readonly batchSize?: number;
  readonly maxRetries?: number;
  readonly fetchImplementation?: typeof fetch;
  readonly wait?: (milliseconds: number) => Promise<void>;
}

export class OpenAIEmbeddingClient implements EmbeddingClient {
  readonly model: string;
  readonly dimensions: number;
  private readonly batchSize: number;
  private readonly maxRetries: number;
  private readonly fetchImplementation: typeof fetch;
  private readonly wait: (milliseconds: number) => Promise<void>;
  private usage: EmbeddingUsage = { requests: 0, inputTokens: 0, totalTokens: 0 };

  constructor(private readonly options: OpenAIEmbeddingClientOptions) {
    if (!options.apiKey.trim()) throw new Error("OPENAI_API_KEY is required");
    this.model = options.model ?? "text-embedding-3-small";
    this.dimensions = options.dimensions ?? 1536;
    this.batchSize = options.batchSize ?? 100;
    this.maxRetries = options.maxRetries ?? 8;
    this.fetchImplementation = options.fetchImplementation ?? fetch;
    this.wait = options.wait ?? ((milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds)));
    if (this.model !== "text-embedding-3-small") {
      throw new Error("Benchmark protocol requires text-embedding-3-small");
    }
    if (this.dimensions !== 1536) {
      throw new Error("Benchmark protocol requires 1536-dimensional OpenAI embeddings");
    }
    if (!Number.isInteger(this.batchSize) || this.batchSize <= 0 || this.batchSize > 2048) {
      throw new Error("batchSize must be an integer between 1 and 2048");
    }
  }

  getUsage(): EmbeddingUsage {
    return { ...this.usage };
  }

  async embed(inputs: readonly EmbeddingInput[], _task: EmbeddingTask): Promise<EmbeddingOutput[]> {
    const outputs: EmbeddingOutput[] = [];
    for (let offset = 0; offset < inputs.length; offset += this.batchSize) {
      const batch = inputs.slice(offset, offset + this.batchSize);
      const values = await this.embedBatch(batch);
      outputs.push(...batch.map((input, index) => ({ id: input.id, values: values[index]! })));
    }
    return outputs;
  }

  private async embedBatch(
    inputs: readonly EmbeddingInput[],
  ): Promise<readonly (readonly number[])[]> {
    const body = {
      model: this.model,
      input: inputs.map((input) => input.text),
      encoding_format: "float",
      dimensions: this.dimensions,
    };

    for (let attempt = 0; attempt <= this.maxRetries; attempt += 1) {
      let response: Response;
      try {
        response = await this.fetchImplementation("https://api.openai.com/v1/embeddings", {
          method: "POST",
          headers: {
            Authorization: `Bearer ${this.options.apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(body),
        });
      } catch (error) {
        if (attempt === this.maxRetries) {
          const message = error instanceof Error ? error.message : String(error);
          throw new Error(`OpenAI embedding network request failed after ${attempt + 1} attempts: ${message}`);
        }
        await this.wait(transientRetryDelayMilliseconds(attempt));
        continue;
      }
      const payload = await readPayload(response);
      if (response.ok) {
        const embeddings = [...(payload.data ?? [])].sort(
          (left, right) => (left.index ?? 0) - (right.index ?? 0),
        );
        if (embeddings.length !== inputs.length) {
          if (attempt === this.maxRetries) {
            throw new Error(`OpenAI returned ${embeddings.length} embeddings for ${inputs.length} inputs`);
          }
          await this.wait(transientRetryDelayMilliseconds(attempt));
          continue;
        }
        const invalidIndex = embeddings.findIndex(
          (embedding) => (embedding.embedding ?? []).length !== this.dimensions,
        );
        if (invalidIndex !== -1) {
          if (attempt === this.maxRetries) {
            const actualDimensions = embeddings[invalidIndex]!.embedding?.length ?? 0;
            throw new Error(
              `OpenAI embedding ${invalidIndex} has ${actualDimensions} dimensions; expected ${this.dimensions}`,
            );
          }
          await this.wait(transientRetryDelayMilliseconds(attempt));
          continue;
        }
        const values = embeddings.map((embedding) => normalizeVector(embedding.embedding!));
        this.usage = {
          requests: this.usage.requests + 1,
          inputTokens: this.usage.inputTokens + (payload.usage?.prompt_tokens ?? 0),
          totalTokens: this.usage.totalTokens + (payload.usage?.total_tokens ?? 0),
        };
        return values;
      }
      const retryable = response.status === 408 || response.status === 409 || response.status === 429 || response.status >= 500;
      if (!retryable || attempt === this.maxRetries) {
        throw new Error(
          `OpenAI embedding request failed (${response.status}): ${payload.error?.message ?? response.statusText}`,
        );
      }
      await this.wait(retryDelayMilliseconds(response, attempt));
    }
    throw new Error("OpenAI embedding request exhausted retries");
  }
}

async function readPayload(response: Response): Promise<OpenAIEmbeddingResponse> {
  try {
    return await response.json() as OpenAIEmbeddingResponse;
  } catch {
    return {};
  }
}

function retryDelayMilliseconds(response: Response, attempt: number): number {
  const retryAfter = response.headers.get("retry-after");
  if (retryAfter) {
    const seconds = Number(retryAfter);
    if (Number.isFinite(seconds) && seconds >= 0) return Math.ceil(seconds * 1000) + 250;
    const date = Date.parse(retryAfter);
    if (Number.isFinite(date)) return Math.max(0, date - Date.now()) + 250;
  }
  return transientRetryDelayMilliseconds(attempt);
}

function transientRetryDelayMilliseconds(attempt: number): number {
  return 500 * 2 ** attempt;
}

export function normalizeVector(values: readonly number[]): number[] {
  const magnitude = Math.sqrt(values.reduce((total, value) => total + value * value, 0));
  if (magnitude === 0) return values.map(() => 0);
  return values.map((value) => value / magnitude);
}

export function cosineSimilarity(left: ArrayLike<number>, right: ArrayLike<number>): number {
  if (left.length !== right.length) {
    throw new Error(`Vector dimension mismatch: ${left.length} != ${right.length}`);
  }
  let score = 0;
  for (let index = 0; index < left.length; index += 1) score += left[index]! * right[index]!;
  return score;
}
