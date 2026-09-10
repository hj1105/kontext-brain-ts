import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  type EmbeddingInputKind,
  FileResourceContentStore,
  LocalKnowledgeSearch,
  OllamaTextEmbedder,
  OpenAITextEmbedder,
  type Principal,
  type ResourceSnapshot,
  SqliteKnowledgeGraphRepository,
  SyncResourceUseCase,
  type TextEmbedder,
  bytesToVector,
  embedMissingChunks,
  normalizeVector,
  unitCosine,
  vectorToBytes,
} from "../src/index.js";

const roots: string[] = [];
afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

const principal: Principal = { organizationId: "org-1", subjectId: "me", groupIds: [] };

/**
 * A toy semantic space: each known concept owns one axis, so "money back" and
 * "refund" land on the same axis even though they share no word.
 */
const CONCEPTS: Record<string, number> = {
  refund: 0,
  "money back": 0,
  reimburse: 0,
  oncall: 1,
  "on-call": 1,
  pager: 1,
  hiring: 2,
  interview: 2,
};

class ToyEmbedder implements TextEmbedder {
  readonly model = "toy:v1";
  readonly calls: { texts: readonly string[]; kind: EmbeddingInputKind }[] = [];
  async embed(texts: readonly string[], kind: EmbeddingInputKind) {
    this.calls.push({ texts, kind });
    return texts.map((text) => {
      const vector = new Float32Array(4);
      const lower = text.toLowerCase();
      for (const [concept, axis] of Object.entries(CONCEPTS)) {
        if (lower.includes(concept)) vector[axis] += 1;
      }
      // Why a constant axis: a text with no known concept still has a direction.
      vector[3] = 0.1;
      return normalizeVector(vector);
    });
  }
}

function snapshot(
  externalId: string,
  title: string,
  chunks: readonly string[],
  version = "",
): ResourceSnapshot {
  return {
    organizationId: principal.organizationId,
    source: { connectorId: "handbook", externalId, type: "local" },
    title,
    contentHash: `hash:${externalId}${version}`,
    body: chunks.join("\n\n"),
    acl: { organizationWide: true },
    ontologyNodeIds: ["Ops"],
    chunks: chunks.map((text, index) => ({
      id: `c${index}`,
      contentHash: `chunk:${externalId}${version}:${index}`,
      text,
      position: index,
    })),
  };
}

async function graph() {
  const data = await mkdtemp(path.join(tmpdir(), "kontext-embeddings-"));
  roots.push(data);
  const repository = await SqliteKnowledgeGraphRepository.open(data);
  const contentStore = new FileResourceContentStore(path.join(data, "knowledge-content"));
  const sync = new SyncResourceUseCase(repository, contentStore);
  await sync.execute(
    snapshot("refunds.md", "Refund policy", [
      "A refund is issued within 14 days.",
      "Partial refunds need a manager.",
    ]),
  );
  await sync.execute(snapshot("oncall.md", "On-call rota", ["The pager rotates on Wednesday."]));
  await sync.execute(snapshot("hiring.md", "Hiring", ["Interviews take two rounds."]));
  return { repository, contentStore, sync };
}

describe("vector helpers", () => {
  it("round-trips a vector through bytes and measures unit cosine", () => {
    const vector = normalizeVector(Float32Array.from([3, 4, 0]));
    expect(Array.from(bytesToVector(vectorToBytes(vector)))).toEqual(Array.from(vector));
    expect(unitCosine(vector, vector)).toBeCloseTo(1, 5);
    expect(unitCosine(vector, normalizeVector(Float32Array.from([0, 0, 1])))).toBeCloseTo(0, 5);
    expect(unitCosine(vector, Float32Array.from([1]))).toBe(0);
  });
});

describe("remote embedders", () => {
  it("posts batches to Ollama and to an OpenAI-compatible API and normalizes what comes back", async () => {
    const requests: { url: string; body: unknown; auth: string | undefined }[] = [];
    const fetchImpl = async (url: string, init: RequestInit) => {
      const body = JSON.parse(String(init.body)) as { input: string[] };
      requests.push({
        url,
        body,
        auth: (init.headers as Record<string, string>).authorization,
      });
      const rows = body.input.map((_text, index) => [index + 1, 0, 0]);
      const payload = url.endsWith("/api/embed")
        ? { embeddings: rows }
        : { data: rows.map((embedding, index) => ({ index, embedding })).reverse() };
      return new Response(JSON.stringify(payload), { status: 200 });
    };
    const ollama = new OllamaTextEmbedder({
      model: "nomic-embed-text",
      baseUrl: "http://127.0.0.1:11434/",
      fetch: fetchImpl,
    });
    const vectors = await ollama.embed(["a", "b"], "passage");
    expect(ollama.model).toBe("ollama:nomic-embed-text");
    expect(requests[0]?.url).toBe("http://127.0.0.1:11434/api/embed");
    expect(Array.from(vectors[1] ?? [])).toEqual([1, 0, 0]);

    const openai = new OpenAITextEmbedder({
      model: "text-embedding-3-small",
      baseUrl: "https://api.openai.com/v1",
      apiKey: "sk-test",
      fetch: fetchImpl,
    });
    const [first, second] = await openai.embed(["a", "b"], "query");
    expect(requests[1]?.auth).toBe("Bearer sk-test");
    // Why: the API may answer out of order; vectors must line up with the inputs by index.
    expect(Array.from(first ?? [])).toEqual([1, 0, 0]);
    expect(Array.from(second ?? [])).toEqual([1, 0, 0]);
  });

  it("names the address when the server cannot be reached at all", async () => {
    const embedder = new OllamaTextEmbedder({
      model: "x",
      baseUrl: "http://127.0.0.1:1",
      fetch: async () => {
        throw Object.assign(new TypeError("fetch failed"), { cause: { code: "ECONNREFUSED" } });
      },
    });
    await expect(embedder.embed(["a"], "query")).rejects.toThrow(
      /Could not reach http:\/\/127\.0\.0\.1:1\/api\/embed \(ECONNREFUSED\); is the server running\?/,
    );
  });

  it("names the endpoint and status when a server refuses", async () => {
    const embedder = new OllamaTextEmbedder({
      model: "x",
      baseUrl: "http://127.0.0.1:1",
      fetch: async () => new Response("model not found", { status: 404 }),
    });
    await expect(embedder.embed(["a"], "query")).rejects.toThrow(
      /api\/embed answered 404: model not found/,
    );
  });
});

describe("embedMissingChunks and hybrid search", () => {
  it("embeds only chunks without a vector, drops orphans, and finds a paraphrase lexical search misses", async () => {
    const { repository, contentStore, sync } = await graph();
    const embedder = new ToyEmbedder();
    const first = await embedMissingChunks(
      repository,
      contentStore,
      embedder,
      principal.organizationId,
      {
        batchSize: 2,
      },
    );
    expect(first).toEqual({ model: "toy:v1", chunksEmbedded: 4, chunksTotal: 4 });
    expect(embedder.calls.every((call) => call.kind === "passage")).toBe(true);
    expect(embedder.calls.flatMap((call) => call.texts)).toContain(
      "Refund policy\nA refund is issued within 14 days.",
    );

    const again = await embedMissingChunks(
      repository,
      contentStore,
      embedder,
      principal.organizationId,
    );
    expect(again.chunksEmbedded).toBe(0);

    // Re-syncing a resource with new content retires its chunks; their vectors go with them.
    await sync.execute(snapshot("hiring.md", "Hiring", ["Three interview rounds now."], "@2"));
    const afterResync = await embedMissingChunks(
      repository,
      contentStore,
      embedder,
      principal.organizationId,
    );
    expect(afterResync.chunksEmbedded).toBe(1);
    const stored = await repository.listChunkVectors(principal.organizationId, "toy:v1");
    expect(stored.size).toBe(4);

    const lexical = new LocalKnowledgeSearch(repository, contentStore);
    const miss = await lexical.search({ question: "money back rules", principal });
    expect(miss.mode).toBe("lexical");
    expect(miss.hits.map((hit) => hit.title)).not.toContain("Refund policy");

    const hybrid = new LocalKnowledgeSearch(repository, contentStore, embedder);
    const hit = await hybrid.search({ question: "money back rules", principal, limit: 2 });
    expect(hit.mode).toBe("hybrid");
    expect(hit.hits[0]?.title).toBe("Refund policy");
    expect(hit.hits[0]?.similarity).toBeGreaterThan(0.9);
    expect(hit.hits.map((entry) => entry.title)).not.toContain("Hiring");

    // A term match still counts: the pager chunk wins on both words and meaning.
    const both = await hybrid.search({
      question: "when does the pager rotate",
      principal,
      limit: 1,
    });
    expect(both.hits[0]?.title).toBe("On-call rota");
    expect(both.hits[0]?.matchedTerms).toContain("pager");
  });

  it("falls back to lexical, saying why, when vectors are missing or the embedder fails", async () => {
    const { repository, contentStore } = await graph();
    const noVectors = new LocalKnowledgeSearch(repository, contentStore, new ToyEmbedder());
    const result = await noVectors.search({ question: "refund", principal });
    expect(result.mode).toBe("lexical");
    expect(result.embeddingError).toMatch(/No chunk vectors for toy:v1/);
    expect(result.hits[0]?.title).toBe("Refund policy");

    await embedMissingChunks(repository, contentStore, new ToyEmbedder(), principal.organizationId);
    const broken: TextEmbedder = {
      model: "toy:v1",
      embed: async () => {
        throw new Error("model file missing");
      },
    };
    const fallback = new LocalKnowledgeSearch(repository, contentStore, broken);
    const answer = await fallback.search({ question: "refund", principal });
    expect(answer.mode).toBe("lexical");
    expect(answer.embeddingError).toBe("model file missing");
    expect(answer.hits[0]?.title).toBe("Refund policy");
  });
});
