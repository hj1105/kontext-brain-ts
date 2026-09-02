import { describe, expect, it } from "vitest";
import type {
  EmbeddingClient,
  EmbeddingInput,
  EmbeddingOutput,
  EmbeddingTask,
} from "../../rag-eval-v2/openai-embeddings.js";
import { renderLargeScaleContext, retrieveLargeScaleContext } from "./retrieval.js";

class SubsystemEmbeddingClient implements EmbeddingClient {
  readonly model = "fixture";
  readonly dimensions = 4;

  async embed(inputs: readonly EmbeddingInput[], _task: EmbeddingTask): Promise<EmbeddingOutput[]> {
    return inputs.map((input) => ({ id: input.id, values: vector(input) }));
  }

  getUsage() {
    return { requests: 0, inputTokens: 0, totalTokens: 0 };
  }
}

describe("large-scale retrieval control", () => {
  it("searches long source documents per visible subsystem", async () => {
    const result = await retrieveLargeScaleContext(new SubsystemEmbeddingClient(), 1);
    expect(result.documents.map((document) => document.documentId)).toEqual([
      "spec:billing-retry-recovery",
      "spec:media-retry",
      "spec:notify-retry",
      "spec:sync-retry",
    ]);
    expect(result.governingRetrieved).toBe(1);
    expect(result.governingTotal).toBe(1);
    expect(renderLargeScaleContext(result)).toContain("Billing retry recovery specification");
  });
});

function vector(input: EmbeddingInput): readonly number[] {
  const value = input.id.toLowerCase();
  const names = ["billing", "notify", "sync", "media"];
  return names.map((name) => Number(value.includes(name)));
}
