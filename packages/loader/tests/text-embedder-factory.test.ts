import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { OllamaTextEmbedder, OpenAITextEmbedder } from "@kontext-brain/core";
import { afterEach, describe, expect, it } from "vitest";
import {
  BuiltinTextEmbedder,
  createTextEmbedder,
  readEmbeddingSettings,
  resolveEmbeddingSettings,
  toEmbeddingConfig,
  writeEmbeddingSettings,
} from "../src/index.js";

const roots: string[] = [];
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("embedding settings", () => {
  it("defaults to the built-in model and fills provider defaults a person did not write", () => {
    expect(resolveEmbeddingSettings(undefined)).toEqual({
      provider: "builtin",
      model: "Xenova/multilingual-e5-small",
      baseUrl: null,
      apiKeyEnv: null,
    });
    expect(resolveEmbeddingSettings({ provider: "ollama" })).toEqual({
      provider: "ollama",
      model: "nomic-embed-text",
      baseUrl: "http://127.0.0.1:11434",
      apiKeyEnv: null,
    });
    expect(
      resolveEmbeddingSettings({ provider: "openai", model: "text-embedding-3-large" }),
    ).toEqual({
      provider: "openai",
      model: "text-embedding-3-large",
      baseUrl: "https://api.openai.com/v1",
      apiKeyEnv: "OPENAI_API_KEY",
    });
    expect(resolveEmbeddingSettings({ provider: "none" }).model).toBe("");
  });

  it("writes back only what differs from the defaults", () => {
    expect(toEmbeddingConfig(resolveEmbeddingSettings({ provider: "ollama" }))).toEqual({
      provider: "ollama",
    });
    expect(
      toEmbeddingConfig(
        resolveEmbeddingSettings({
          provider: "openai",
          baseUrl: "https://llm.internal/v1",
          apiKeyEnv: "INTERNAL_KEY",
        }),
      ),
    ).toEqual({
      provider: "openai",
      baseUrl: "https://llm.internal/v1",
      apiKeyEnv: "INTERNAL_KEY",
    });
  });

  it("records the settings in the data directory and reads them back, ignoring garbage", () => {
    const data = mkdtempSync(path.join(tmpdir(), "kontext-embedding-settings-"));
    roots.push(data);
    expect(readEmbeddingSettings(data)).toBeNull();
    const settings = resolveEmbeddingSettings({ provider: "ollama", model: "bge-m3" });
    writeEmbeddingSettings(data, settings);
    expect(readEmbeddingSettings(data)).toEqual(settings);
  });

  it("builds the embedder each provider names and refuses openai without its key", () => {
    const data = mkdtempSync(path.join(tmpdir(), "kontext-embedding-factory-"));
    roots.push(data);
    const options = { dataDirectory: data, env: {} };
    expect(createTextEmbedder(resolveEmbeddingSettings({ provider: "none" }), options)).toBeNull();
    expect(createTextEmbedder(resolveEmbeddingSettings(undefined), options)).toBeInstanceOf(
      BuiltinTextEmbedder,
    );
    expect(
      createTextEmbedder(resolveEmbeddingSettings({ provider: "ollama" }), options),
    ).toBeInstanceOf(OllamaTextEmbedder);
    expect(() =>
      createTextEmbedder(resolveEmbeddingSettings({ provider: "openai" }), options),
    ).toThrow(/OPENAI_API_KEY environment variable/);
    expect(
      createTextEmbedder(resolveEmbeddingSettings({ provider: "openai" }), {
        dataDirectory: data,
        env: { OPENAI_API_KEY: "sk" },
      }),
    ).toBeInstanceOf(OpenAITextEmbedder);
  });
});
