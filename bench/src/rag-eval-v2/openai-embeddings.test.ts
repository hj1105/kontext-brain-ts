import { describe, expect, it, vi } from "vitest";
import { OpenAIEmbeddingClient } from "./openai-embeddings.js";

describe("OpenAIEmbeddingClient", () => {
  it("uses text-embedding-3-small symmetrically at the frozen dimensions", async () => {
    const requests: Array<{ url: string; headers: Headers; body: Record<string, unknown> }> = [];
    const fetchImplementation = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      requests.push({
        url: String(input),
        headers: new Headers(init?.headers),
        body: JSON.parse(String(init?.body)) as Record<string, unknown>,
      });
      return new Response(JSON.stringify({
        data: [{ index: 0, embedding: Array.from({ length: 1536 }, () => 1) }],
        model: "text-embedding-3-small",
        usage: { prompt_tokens: 3, total_tokens: 3 },
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }) as typeof fetch;
    const client = new OpenAIEmbeddingClient({ apiKey: "test", fetchImplementation });
    const [embedding] = await client.embed(
      [{ id: "d1", title: "Title", text: "document" }],
      "RETRIEVAL_DOCUMENT",
    );

    expect(requests[0]!.url).toBe("https://api.openai.com/v1/embeddings");
    expect(requests[0]!.headers.get("Authorization")).toBe("Bearer test");
    expect(requests[0]!.body).toEqual({
      model: "text-embedding-3-small",
      input: ["document"],
      encoding_format: "float",
      dimensions: 1536,
    });
    expect(embedding!.values).toHaveLength(1536);
    expect(Math.sqrt(embedding!.values.reduce((sum, value) => sum + value * value, 0))).toBeCloseTo(1);
    expect(client.getUsage()).toEqual({ requests: 1, inputTokens: 3, totalTokens: 3 });
  });

  it("uses the same unprefixed text for retrieval queries", async () => {
    let sentInput: unknown;
    const fetchImplementation = vi.fn(async (_input: string | URL | Request, init?: RequestInit) => {
      sentInput = (JSON.parse(String(init?.body)) as { input: unknown }).input;
      return new Response(JSON.stringify({
        data: [{ index: 0, embedding: Array.from({ length: 1536 }, () => 0.5) }],
        usage: { prompt_tokens: 1, total_tokens: 1 },
      }), { status: 200 });
    }) as typeof fetch;
    const client = new OpenAIEmbeddingClient({ apiKey: "test", fetchImplementation });

    await client.embed([{ id: "q1", text: "question" }], "RETRIEVAL_QUERY");

    expect(sentInput).toEqual(["question"]);
  });

  it("retries rate limits using Retry-After", async () => {
    let attempts = 0;
    const wait = vi.fn(async (_milliseconds: number) => undefined);
    const fetchImplementation = vi.fn(async () => {
      attempts += 1;
      return attempts === 1
        ? new Response(JSON.stringify({ error: { message: "rate limited" } }), {
            status: 429,
            headers: { "Retry-After": "2" },
          })
        : new Response(JSON.stringify({
            data: [{ index: 0, embedding: Array.from({ length: 1536 }, () => 0.5) }],
            usage: { prompt_tokens: 1, total_tokens: 1 },
          }), { status: 200 });
    }) as typeof fetch;
    const client = new OpenAIEmbeddingClient({ apiKey: "test", fetchImplementation, wait });

    await client.embed([{ id: "q1", text: "question" }], "RETRIEVAL_QUERY");

    expect(attempts).toBe(2);
    expect(wait).toHaveBeenCalledWith(2250);
  });

  it("retries transient network failures inside the current checkpoint batch", async () => {
    let attempts = 0;
    const wait = vi.fn(async (_milliseconds: number) => undefined);
    const fetchImplementation = vi.fn(async () => {
      attempts += 1;
      if (attempts === 1) throw new TypeError("fetch failed");
      return new Response(JSON.stringify({
        data: [{ index: 0, embedding: Array.from({ length: 1536 }, () => 0.5) }],
        usage: { prompt_tokens: 1, total_tokens: 1 },
      }), { status: 200 });
    }) as typeof fetch;
    const client = new OpenAIEmbeddingClient({ apiKey: "test", fetchImplementation, wait });

    await client.embed([{ id: "q1", text: "question" }], "RETRIEVAL_QUERY");

    expect(attempts).toBe(2);
    expect(wait).toHaveBeenCalledWith(500);
  });

  it("retries a transient successful response with missing embeddings", async () => {
    let attempts = 0;
    const wait = vi.fn(async (_milliseconds: number) => undefined);
    const fetchImplementation = vi.fn(async () => {
      attempts += 1;
      return attempts === 1
        ? new Response(JSON.stringify({ data: [] }), { status: 200 })
        : new Response(JSON.stringify({
            data: [{ index: 0, embedding: Array.from({ length: 1536 }, () => 0.5) }],
            usage: { prompt_tokens: 1, total_tokens: 1 },
          }), { status: 200 });
    }) as typeof fetch;
    const client = new OpenAIEmbeddingClient({ apiKey: "test", fetchImplementation, wait });

    await client.embed([{ id: "q1", text: "question" }], "RETRIEVAL_QUERY");

    expect(attempts).toBe(2);
    expect(wait).toHaveBeenCalledWith(500);
  });
});
