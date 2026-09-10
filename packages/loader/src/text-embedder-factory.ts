import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import path from "node:path";
import { OllamaTextEmbedder, OpenAITextEmbedder, type TextEmbedder } from "@kontext-brain/core";
import {
  BuiltinTextEmbedder,
  DEFAULT_BUILTIN_EMBEDDING_MODEL,
  type ModelDownloadEvent,
} from "./builtin-text-embedder.js";
import type { EmbeddingConfigDto } from "./kontext-config.js";

/**
 * One place that turns the `embedding:` section of kontext.yaml into an
 * embedder, fills in the defaults a person did not write, and remembers the
 * choice in the data directory so the Task sidecar — which sees the data
 * directory but not the workspace's kontext.yaml — searches in the same space
 * the build wrote.
 */

export type EmbeddingProvider = EmbeddingConfigDto["provider"];

export interface EmbeddingSettings {
  readonly provider: EmbeddingProvider;
  readonly model: string;
  readonly baseUrl: string | null;
  readonly apiKeyEnv: string | null;
}

export const DEFAULT_EMBEDDING_MODELS: Record<EmbeddingProvider, string> = {
  builtin: DEFAULT_BUILTIN_EMBEDDING_MODEL,
  ollama: "nomic-embed-text",
  openai: "text-embedding-3-small",
  none: "",
};

const DEFAULT_BASE_URLS: Record<EmbeddingProvider, string | null> = {
  builtin: null,
  ollama: "http://127.0.0.1:11434",
  openai: "https://api.openai.com/v1",
  none: null,
};

export function resolveEmbeddingSettings(
  config: EmbeddingConfigDto | undefined,
): EmbeddingSettings {
  const provider = config?.provider ?? "builtin";
  return {
    provider,
    model: config?.model?.trim() || DEFAULT_EMBEDDING_MODELS[provider],
    baseUrl: config?.baseUrl?.trim() || DEFAULT_BASE_URLS[provider],
    apiKeyEnv: provider === "openai" ? config?.apiKeyEnv?.trim() || "OPENAI_API_KEY" : null,
  };
}

/** The config entry that reproduces these settings, without the defaults it would fill in anyway. */
export function toEmbeddingConfig(settings: EmbeddingSettings): EmbeddingConfigDto {
  const config: {
    provider: EmbeddingProvider;
    model?: string;
    baseUrl?: string;
    apiKeyEnv?: string;
  } = { provider: settings.provider };
  if (settings.model && settings.model !== DEFAULT_EMBEDDING_MODELS[settings.provider]) {
    config.model = settings.model;
  }
  if (settings.baseUrl && settings.baseUrl !== DEFAULT_BASE_URLS[settings.provider]) {
    config.baseUrl = settings.baseUrl;
  }
  if (settings.apiKeyEnv && settings.apiKeyEnv !== "OPENAI_API_KEY") {
    config.apiKeyEnv = settings.apiKeyEnv;
  }
  return config;
}

export interface CreateTextEmbedderOptions {
  readonly dataDirectory: string;
  readonly env?: NodeJS.ProcessEnv;
  readonly onDownload?: (event: ModelDownloadEvent) => void;
}

/** Null for `none`; throws when a hosted provider is missing what it needs to be called at all. */
export function createTextEmbedder(
  settings: EmbeddingSettings,
  options: CreateTextEmbedderOptions,
): TextEmbedder | null {
  const env = options.env ?? process.env;
  switch (settings.provider) {
    case "none":
      return null;
    case "builtin":
      return new BuiltinTextEmbedder({
        model: settings.model,
        modelsDirectory: path.join(options.dataDirectory, "models"),
        ...(options.onDownload ? { onDownload: options.onDownload } : {}),
      });
    case "ollama":
      return new OllamaTextEmbedder({
        model: settings.model,
        baseUrl: settings.baseUrl ?? (DEFAULT_BASE_URLS.ollama as string),
      });
    case "openai": {
      const variable = settings.apiKeyEnv ?? "OPENAI_API_KEY";
      const apiKey = env[variable]?.trim();
      if (!apiKey) {
        throw new Error(
          `Embedding provider openai needs the ${variable} environment variable; set it, or choose another provider.`,
        );
      }
      return new OpenAITextEmbedder({
        model: settings.model,
        baseUrl: settings.baseUrl ?? (DEFAULT_BASE_URLS.openai as string),
        apiKey,
      });
    }
  }
}

function settingsFile(dataDirectory: string): string {
  return path.join(dataDirectory, "embedding.json");
}

/** Recorded by every build and embed run; the Task sidecar reads it to search in the same space. */
export function writeEmbeddingSettings(dataDirectory: string, settings: EmbeddingSettings): void {
  const file = settingsFile(dataDirectory);
  mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  writeFileSync(temporary, `${JSON.stringify(settings, null, 2)}\n`);
  renameSync(temporary, file);
}

export function readEmbeddingSettings(dataDirectory: string): EmbeddingSettings | null {
  try {
    const parsed = JSON.parse(
      readFileSync(settingsFile(dataDirectory), "utf8"),
    ) as Partial<EmbeddingSettings>;
    if (!parsed.provider || !(parsed.provider in DEFAULT_EMBEDDING_MODELS)) return null;
    return resolveEmbeddingSettings({
      provider: parsed.provider,
      ...(parsed.model ? { model: parsed.model } : {}),
      ...(parsed.baseUrl ? { baseUrl: parsed.baseUrl } : {}),
      ...(parsed.apiKeyEnv ? { apiKeyEnv: parsed.apiKeyEnv } : {}),
    });
  } catch {
    return null;
  }
}
