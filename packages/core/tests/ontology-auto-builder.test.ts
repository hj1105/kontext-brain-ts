import { describe, expect, it } from "vitest";
import { InMemoryDocumentSource, type LLMAdapter, OntologyAutoBuilder } from "../src/index.js";

class CapturingLLM implements LLMAdapter {
  readonly nodeDesignPrompts: string[] = [];

  constructor(private readonly categories = ["Backend", "Frontend", "Security", "Operations"]) {}

  async complete(systemPrompt: string): Promise<string> {
    if (systemPrompt.includes("extract common topic categories")) {
      return JSON.stringify(this.categories);
    }
    if (systemPrompt.includes("design approximately")) {
      this.nodeDesignPrompts.push(systemPrompt);
      return JSON.stringify({
        nodes: [
          { id: "Engineering", description: "software systems", level: 0 },
          { id: "Operations", description: "deployments incidents", level: 0 },
        ],
      });
    }
    return JSON.stringify({ edges: [] });
  }
}

function createDocuments(count: number) {
  return Array.from({ length: count }, (_, index) => ({
    id: `doc-${index}`,
    title: `Document ${index}`,
    metadata: {},
  }));
}

describe("OntologyAutoBuilder node-count selection", () => {
  it("infers the target from corpus size and topic diversity by default", async () => {
    const llm = new CapturingLLM();
    const builder = new OntologyAutoBuilder(llm);

    await builder.build([new InMemoryDocumentSource(createDocuments(16))]);

    expect(llm.nodeDesignPrompts).toHaveLength(1);
    expect(llm.nodeDesignPrompts[0]).toContain("design approximately 4 ontology nodes");
  });

  it("does not infer more nodes than a small corpus can support", async () => {
    const llm = new CapturingLLM();
    const builder = new OntologyAutoBuilder(llm);

    await builder.build([new InMemoryDocumentSource(createDocuments(2))]);

    expect(llm.nodeDesignPrompts[0]).toContain("design approximately 2 ontology nodes");
  });

  it("increases the target when topic diversity exceeds the corpus-size signal", async () => {
    const categories = Array.from({ length: 24 }, (_, index) => `Topic ${index}`);
    const llm = new CapturingLLM(categories);
    const builder = new OntologyAutoBuilder(llm);

    await builder.build([new InMemoryDocumentSource(createDocuments(100))]);

    expect(llm.nodeDesignPrompts[0]).toContain("design approximately 12 ontology nodes");
  });

  it("caps the automatically inferred target for large corpora", async () => {
    const llm = new CapturingLLM();
    const builder = new OntologyAutoBuilder(llm);

    await builder.build([new InMemoryDocumentSource(createDocuments(625))]);

    expect(llm.nodeDesignPrompts[0]).toContain("design approximately 20 ontology nodes");
  });

  it("keeps an explicit target node-count override", async () => {
    const llm = new CapturingLLM();
    const builder = new OntologyAutoBuilder(llm, 8);

    await builder.build([new InMemoryDocumentSource(createDocuments(16))]);

    expect(llm.nodeDesignPrompts).toHaveLength(1);
    expect(llm.nodeDesignPrompts[0]).toContain("design approximately 8 ontology nodes");
  });
});
