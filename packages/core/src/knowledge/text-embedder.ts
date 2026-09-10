/**
 * Turns text into vectors for semantic search over the local knowledge graph.
 * Lexical scoring alone misses a question phrased differently from the answer
 * ("money back rules" against a page titled "Refund policy"); an embedder
 * closes that gap. Implementations range from an in-process model to a local
 * Ollama or a hosted API, so the interface carries only what search needs.
 */

/** e5-style models want to know which side of the match a text is on. */
export type EmbeddingInputKind = "query" | "passage";

export interface TextEmbedder {
  /** Identifies the vectors' space; vectors from different models never mix. */
  readonly model: string;
  embed(texts: readonly string[], kind: EmbeddingInputKind): Promise<readonly Float32Array[]>;
}

export function normalizeVector(vector: Float32Array): Float32Array {
  let sum = 0;
  for (const value of vector) sum += value * value;
  const norm = Math.sqrt(sum);
  if (norm === 0) return vector;
  for (let index = 0; index < vector.length; index += 1)
    vector[index] = (vector[index] ?? 0) / norm;
  return vector;
}

/** Both inputs are expected to be unit vectors; the result is then their cosine. */
export function unitCosine(left: Float32Array, right: Float32Array): number {
  if (left.length !== right.length) return 0;
  let dot = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += (left[index] ?? 0) * (right[index] ?? 0);
  }
  return dot;
}

export function vectorToBytes(vector: Float32Array): Uint8Array {
  return new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength).slice();
}

export function bytesToVector(bytes: Uint8Array): Float32Array {
  const aligned = bytes.byteOffset % 4 === 0 ? bytes : bytes.slice();
  return new Float32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4);
}
