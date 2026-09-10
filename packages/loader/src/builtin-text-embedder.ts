import { createWriteStream, existsSync } from "node:fs";
import { mkdir, readFile, rename, stat } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { type EmbeddingInputKind, type TextEmbedder, normalizeVector } from "@kontext-brain/core";

/**
 * Embeds text with an ONNX sentence model run on the WebAssembly build of ONNX
 * Runtime, inside the sidecar process. Nothing to install: the model files are
 * fetched once from the Hugging Face hub into the data directory. WebAssembly
 * is slower than a native runtime but ships the same bytes on every platform,
 * which is what a desktop app that packages itself for three of them needs.
 */

export const DEFAULT_BUILTIN_EMBEDDING_MODEL = "Xenova/multilingual-e5-small";
const MAX_TOKENS = 512;
const MODEL_FILES = [
  { remote: "onnx/model_quantized.onnx", local: "model_quantized.onnx" },
  { remote: "tokenizer.json", local: "tokenizer.json" },
  { remote: "tokenizer_config.json", local: "tokenizer_config.json" },
] as const;

export interface ModelDownloadEvent {
  readonly file: string;
  readonly receivedBytes: number;
  readonly totalBytes: number | null;
}

export interface BuiltinTextEmbedderOptions {
  /** Hub repository id, e.g. Xenova/multilingual-e5-small; must ship onnx/model_quantized.onnx. */
  readonly model?: string;
  /** Where model files live; each model gets its own directory below it. */
  readonly modelsDirectory: string;
  readonly onDownload?: (event: ModelDownloadEvent) => void;
  readonly fetch?: (input: string) => Promise<Response>;
  /** Hub base URL; HF_ENDPOINT-style mirrors go here. */
  readonly hubUrl?: string;
}

interface OnnxRuntime {
  env: { wasm: { wasmPaths?: string; numThreads?: number; proxy?: boolean } };
  InferenceSession: {
    create(
      pathOrBuffer: string,
      options: { executionProviders: string[] },
    ): Promise<{
      inputNames: readonly string[];
      outputNames: readonly string[];
      run(feeds: Record<string, unknown>): Promise<Record<string, OnnxTensor>>;
    }>;
  };
  Tensor: new (type: string, data: BigInt64Array, dims: number[]) => unknown;
}

interface OnnxTensor {
  readonly dims: readonly number[];
  readonly data: ArrayLike<number>;
}

interface LoadedModel {
  readonly ort: OnnxRuntime;
  readonly session: Awaited<ReturnType<OnnxRuntime["InferenceSession"]["create"]>>;
  readonly tokenizer: { encode(text: string): { ids: number[] } };
}

/** Hub ids contain a slash; on disk the model gets one flat directory name. */
export function builtinModelDirectory(modelsDirectory: string, model: string): string {
  return path.join(modelsDirectory, model.replace(/[^A-Za-z0-9._-]+/g, "__"));
}

export class BuiltinTextEmbedder implements TextEmbedder {
  readonly model: string;
  private readonly hubModel: string;
  private readonly directory: string;
  private loaded: Promise<LoadedModel> | undefined;

  constructor(private readonly options: BuiltinTextEmbedderOptions) {
    this.hubModel = options.model ?? DEFAULT_BUILTIN_EMBEDDING_MODEL;
    this.model = `builtin:${this.hubModel}`;
    this.directory = builtinModelDirectory(options.modelsDirectory, this.hubModel);
  }

  async embed(
    texts: readonly string[],
    kind: EmbeddingInputKind,
  ): Promise<readonly Float32Array[]> {
    const loaded = await this.load();
    const vectors: Float32Array[] = [];
    for (const text of texts) {
      vectors.push(await this.embedOne(loaded, this.prefixed(text, kind)));
    }
    return vectors;
  }

  /** Downloads what is missing; a no-op once the files are on disk. */
  async ensureDownloaded(): Promise<void> {
    await mkdir(this.directory, { recursive: true });
    for (const file of MODEL_FILES) {
      const target = path.join(this.directory, file.local);
      if (await exists(target)) continue;
      await this.download(file.remote, target);
    }
  }

  // Why: e5 models were trained with these prefixes; without them similarity drops noticeably.
  private prefixed(text: string, kind: EmbeddingInputKind): string {
    if (!/e5/i.test(this.hubModel)) return text;
    return `${kind}: ${text}`;
  }

  private load(): Promise<LoadedModel> {
    this.loaded ??= this.loadNow().catch((error) => {
      this.loaded = undefined;
      throw error;
    });
    return this.loaded;
  }

  private async loadNow(): Promise<LoadedModel> {
    await this.ensureDownloaded();
    const require = createRequire(import.meta.url);
    const ort = (await import("onnxruntime-web")) as unknown as OnnxRuntime;
    const { Tokenizer } = (await import("@huggingface/tokenizers")) as unknown as {
      Tokenizer: new (
        json: unknown,
        config: unknown,
      ) => { encode(text: string): { ids: number[] } };
    };
    ort.env.wasm.wasmPaths = `${wasmDirectory(require)}/`;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    const [tokenizerJson, tokenizerConfig] = await Promise.all([
      readFile(path.join(this.directory, "tokenizer.json"), "utf8"),
      readFile(path.join(this.directory, "tokenizer_config.json"), "utf8"),
    ]);
    const tokenizer = new Tokenizer(JSON.parse(tokenizerJson), JSON.parse(tokenizerConfig));
    const session = await ort.InferenceSession.create(
      path.join(this.directory, "model_quantized.onnx"),
      { executionProviders: ["wasm"] },
    );
    return { ort, session, tokenizer };
  }

  private async embedOne(loaded: LoadedModel, text: string): Promise<Float32Array> {
    const { ort, session, tokenizer } = loaded;
    let ids = tokenizer.encode(text).ids;
    if (ids.length > MAX_TOKENS) {
      // Why keep the last token: it is the end-of-sequence marker the model expects.
      ids = [...ids.slice(0, MAX_TOKENS - 1), ids[ids.length - 1] as number];
    }
    const length = ids.length;
    const feeds: Record<string, unknown> = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, length]),
      attention_mask: new ort.Tensor("int64", BigInt64Array.from(ids.map(() => 1n)), [1, length]),
    };
    if (session.inputNames.includes("token_type_ids")) {
      feeds.token_type_ids = new ort.Tensor("int64", new BigInt64Array(length), [1, length]);
    }
    const output = await session.run(feeds);
    const hidden = output[session.outputNames[0] ?? ""];
    if (!hidden) throw new Error(`${this.hubModel} produced no output`);
    const [, tokens = 0, dimensions = 0] = hidden.dims;
    const vector = new Float32Array(dimensions);
    for (let token = 0; token < tokens; token += 1) {
      for (let axis = 0; axis < dimensions; axis += 1) {
        vector[axis] = (vector[axis] ?? 0) + (hidden.data[token * dimensions + axis] ?? 0);
      }
    }
    for (let axis = 0; axis < dimensions; axis += 1) vector[axis] = (vector[axis] ?? 0) / tokens;
    return normalizeVector(vector);
  }

  private async download(remote: string, target: string): Promise<void> {
    const hub = (
      this.options.hubUrl ??
      process.env.HF_ENDPOINT ??
      "https://huggingface.co"
    ).replace(/\/+$/, "");
    const url = `${hub}/${this.hubModel}/resolve/main/${remote}`;
    const fetchImpl = this.options.fetch ?? ((input: string) => fetch(input));
    const response = await fetchImpl(url);
    if (!response.ok || !response.body) {
      throw new Error(
        `Could not download ${url} (${response.status}); check the network or set embedding.provider to none.`,
      );
    }
    const total = Number(response.headers.get("content-length"));
    const totalBytes = Number.isFinite(total) && total > 0 ? total : null;
    let received = 0;
    const temporary = `${target}.${process.pid}.part`;
    const progress = new (await import("node:stream")).Transform({
      transform: (chunk: Buffer, _encoding, callback) => {
        received += chunk.length;
        this.options.onDownload?.({ file: remote, receivedBytes: received, totalBytes });
        callback(null, chunk);
      },
    });
    await pipeline(
      Readable.fromWeb(response.body as import("node:stream/web").ReadableStream),
      progress,
      createWriteStream(temporary),
    );
    await rename(temporary, target);
  }
}

/**
 * Where ort-wasm-simd-threaded.wasm lives: beside the installed onnxruntime-web
 * package in a checkout, or beside this file when the single-file bundle ships
 * the two WebAssembly assets next to itself.
 */
function wasmDirectory(require: NodeJS.Require): string {
  try {
    return path.dirname(require.resolve("onnxruntime-web"));
  } catch {
    const beside = path.dirname(fileURLToPath(import.meta.url));
    if (existsSync(path.join(beside, "ort-wasm-simd-threaded.wasm"))) return beside;
    throw new Error(
      `The built-in embedding runtime (ort-wasm-simd-threaded.wasm) was not found beside ${beside}; reinstall, or set embedding.provider to ollama, openai or none.`,
    );
  }
}

async function exists(file: string): Promise<boolean> {
  try {
    return (await stat(file)).size > 0;
  } catch {
    return false;
  }
}
