import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { Embeddings } from "@langchain/core/embeddings";

export interface LLMProviderConfig {
  readonly provider: string;
  readonly model: string;
  readonly apiKey?: string;
  readonly baseUrl?: string;
}

/** Resolves ${ENV_VAR} placeholders in apiKey. */
export function resolveApiKey(config: LLMProviderConfig): string {
  const key = config.apiKey ?? "";
  return key.replace(/\$\{(.+?)\}/g, (_, name: string) => {
    const val = process.env[name];
    if (!val) throw new Error(`Environment variable '${name}' is not set`);
    return val;
  });
}

export interface LLMProviderFactory {
  readonly providerName: string;
  createChat(config: LLMProviderConfig): BaseChatModel;
  createEmbedding(config: LLMProviderConfig): Embeddings;
}

// ── Concrete factories ────────────────────────────────────────

export class ClaudeProviderFactory implements LLMProviderFactory {
  readonly providerName = "claude";

  createChat(config: LLMProviderConfig): BaseChatModel {
    // Lazy import so this module doesn't fail to load if @langchain/anthropic is absent.
    // biome-ignore lint/suspicious/noExplicitAny: require is untyped here
    const { ChatAnthropic } = require("@langchain/anthropic") as any;
    return new ChatAnthropic({
      apiKey: resolveApiKey(config),
      model: config.model,
    });
  }

  createEmbedding(_config: LLMProviderConfig): Embeddings {
    // Claude has no first-party embedding model; fall back to OpenAI-compatible.
    throw new Error(
      "ClaudeProviderFactory does not supply embeddings. Use OpenAI or Ollama for embeddings.",
    );
  }
}

export class OpenAIProviderFactory implements LLMProviderFactory {
  readonly providerName = "openai";

  createChat(config: LLMProviderConfig): BaseChatModel {
    // biome-ignore lint/suspicious/noExplicitAny: require is untyped here
    const { ChatOpenAI } = require("@langchain/openai") as any;
    return new ChatOpenAI({
      apiKey: resolveApiKey(config),
      model: config.model,
    });
  }

  createEmbedding(config: LLMProviderConfig): Embeddings {
    // biome-ignore lint/suspicious/noExplicitAny: require is untyped here
    const { OpenAIEmbeddings } = require("@langchain/openai") as any;
    return new OpenAIEmbeddings({
      apiKey: resolveApiKey(config),
      model: "text-embedding-3-small",
    });
  }
}

export class OllamaProviderFactory implements LLMProviderFactory {
  readonly providerName = "ollama";

  createChat(config: LLMProviderConfig): BaseChatModel {
    // biome-ignore lint/suspicious/noExplicitAny: require is untyped here
    const { ChatOllama } = require("@langchain/ollama") as any;
    return new ChatOllama({
      baseUrl: config.baseUrl ?? "http://localhost:11434",
      model: config.model,
    });
  }

  createEmbedding(config: LLMProviderConfig): Embeddings {
    // biome-ignore lint/suspicious/noExplicitAny: require is untyped here
    const { OllamaEmbeddings } = require("@langchain/ollama") as any;
    return new OllamaEmbeddings({
      baseUrl: config.baseUrl ?? "http://localhost:11434",
      model: config.model,
    });
  }
}

export class LLMProviderRegistry {
  private readonly factories = new Map<string, LLMProviderFactory>();

  constructor() {
    this.register(new ClaudeProviderFactory());
    this.register(new OpenAIProviderFactory());
    this.register(new OllamaProviderFactory());
  }

  register(factory: LLMProviderFactory): void {
    this.factories.set(factory.providerName, factory);
  }

  createChat(config: LLMProviderConfig): BaseChatModel {
    return this.resolve(config.provider).createChat(config);
  }

  createEmbedding(config: LLMProviderConfig): Embeddings {
    return this.resolve(config.provider).createEmbedding(config);
  }

  private resolve(name: string): LLMProviderFactory {
    const f = this.factories.get(name);
    if (!f) {
      throw new Error(
        `Unsupported LLM provider: '${name}'. Registered: ${Array.from(this.factories.keys()).join(",")}`,
      );
    }
    return f;
  }
}
