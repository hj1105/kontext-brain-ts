import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { unitCosine } from "@kontext-brain/core";
import { afterEach, describe, expect, it } from "vitest";
import { BuiltinTextEmbedder, builtinModelDirectory } from "../src/index.js";

/**
 * The real model is 118 MB, so this runs only where KONTEXT_TEST_MODEL_DIR
 * points at a directory holding model_quantized.onnx, tokenizer.json and
 * tokenizer_config.json for Xenova/multilingual-e5-small. The "download" is
 * served from that directory through an injected fetch, so the test covers the
 * download path, the WebAssembly runtime and the pooling without the network.
 */
const modelDirectory = process.env.KONTEXT_TEST_MODEL_DIR;
const roots: string[] = [];
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("builtinModelDirectory", () => {
  it("flattens a hub id into one directory name", () => {
    expect(builtinModelDirectory("/data/models", "Xenova/multilingual-e5-small")).toBe(
      path.join("/data/models", "Xenova__multilingual-e5-small"),
    );
  });
});

describe.skipIf(!modelDirectory)("BuiltinTextEmbedder (live model)", () => {
  it("downloads once, embeds with e5 prefixes, and ranks a Korean paraphrase above an unrelated passage", async () => {
    const data = mkdtempSync(path.join(tmpdir(), "kontext-builtin-embedder-"));
    roots.push(data);
    const fetched: string[] = [];
    const embedder = new BuiltinTextEmbedder({
      modelsDirectory: path.join(data, "models"),
      hubUrl: "https://hub.invalid",
      fetch: async (url) => {
        fetched.push(url);
        const file = path.join(modelDirectory as string, path.basename(url));
        const bytes = readFileSync(file);
        return new Response(bytes, {
          status: 200,
          headers: { "content-length": String(bytes.byteLength) },
        });
      },
    });
    const [query, refund, oncall] = await embedder
      .embed(["환불은 어떻게 처리되나요"], "query")
      .then(async ([q]) => [
        q as Float32Array,
        ...(await embedder.embed(
          [
            "Refunds are issued within 14 days; partial refunds need a manager approval.",
            "On-call rotates weekly on Wednesday.",
          ],
          "passage",
        )),
      ]);
    expect(fetched).toEqual([
      "https://hub.invalid/Xenova/multilingual-e5-small/resolve/main/onnx/model_quantized.onnx",
      "https://hub.invalid/Xenova/multilingual-e5-small/resolve/main/tokenizer.json",
      "https://hub.invalid/Xenova/multilingual-e5-small/resolve/main/tokenizer_config.json",
    ]);
    expect(
      existsSync(
        path.join(
          builtinModelDirectory(path.join(data, "models"), "Xenova/multilingual-e5-small"),
          "model_quantized.onnx",
        ),
      ),
    ).toBe(true);
    expect(query.length).toBe(384);
    expect(unitCosine(query, refund as Float32Array)).toBeGreaterThan(
      unitCosine(query, oncall as Float32Array),
    );
    // A second embedder over the same directory downloads nothing.
    fetched.length = 0;
    const again = new BuiltinTextEmbedder({
      modelsDirectory: path.join(data, "models"),
      fetch: async () => {
        throw new Error("network must not be used");
      },
    });
    await again.embed(["x"], "query");
    expect(fetched).toEqual([]);
  }, 120_000);
});
