import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { Embeddings } from "@langchain/core/embeddings";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import {
  DefaultPromptTemplates,
  type LLMAdapter,
  type PromptTemplates,
  type VectorStore,
  cosineSimilarity,
} from "@kontext-brain/core";

/** Adapter that wraps a LangChain.js BaseChatModel as an LLMAdapter. */
export class LangChainLLMAdapter implements LLMAdapter {
  constructor(
    private readonly model: BaseChatModel,
    private readonly templates: PromptTemplates = DefaultPromptTemplates,
  ) {}

  async complete(systemPrompt: string, context: string, query: string): Promise<string> {
    const response = await this.model.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(this.templates.formatUserMessage(context, query)),
    ]);
    const content = response.content;
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content
        .map((c) => (typeof c === "string" ? c : "text" in c ? String(c.text ?? "") : ""))
        .join("");
    }
    return String(content);
  }
}

/** VectorStore backed by a LangChain.js Embeddings model + in-memory cosine search. */
export class LangChainVectorStore implements VectorStore {
  private readonly index = new Map<
    string,
    { embedding: Float32Array; metadata: Record<string, string> }
  >();

  constructor(private readonly embeddings: Embeddings) {}

  async embed(text: string): Promise<Float32Array> {
    const vec = await this.embeddings.embedQuery(text);
    return Float32Array.from(vec);
  }

  async upsert(
    key: string,
    embedding: Float32Array,
    metadata: Record<string, string> = {},
  ): Promise<void> {
    this.index.set(key, { embedding, metadata });
  }

  async similaritySearch(query: string, topK: number): Promise<string[]> {
    return this.similaritySearchWithPrefix(query, "", topK);
  }

  async similaritySearchWithPrefix(
    query: string,
    prefix: string,
    topK: number,
    threshold = 0,
  ): Promise<string[]> {
    const queryVec = await this.embed(query);
    const scored: Array<{ key: string; score: number }> = [];
    for (const [key, { embedding }] of this.index) {
      if (prefix && !key.startsWith(prefix)) continue;
      const score = cosineSimilarity(queryVec, embedding);
      if (threshold > 0 && score < threshold) continue;
      scored.push({ key, score });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map((s) => {
      const idx = s.key.lastIndexOf(":");
      return idx >= 0 ? s.key.slice(idx + 1) : s.key;
    });
  }
}
