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

  it("gives each extracted topic its own node rather than merging pairs", async () => {
    const categories = Array.from({ length: 24 }, (_, index) => `Topic ${index}`);
    const llm = new CapturingLLM(categories);
    const builder = new OntologyAutoBuilder(llm);

    await builder.build([new InMemoryDocumentSource(createDocuments(100))]);

    // Halving the topic count merges distinct areas. For a governance ontology
    // a merged node hands a symbol its neighbour's approved decision with the
    // same authority as its own, so one node too many is the cheaper mistake.
    expect(llm.nodeDesignPrompts[0]).toContain("design approximately 24 ontology nodes");
  });

  it("still grows sublinearly with corpus size", async () => {
    const llm = new CapturingLLM();
    const builder = new OntologyAutoBuilder(llm);

    await builder.build([new InMemoryDocumentSource(createDocuments(625))]);

    // 625 documents and four topics: the square-root term decides, so the node
    // count stays far below the document count without a fixed ceiling.
    expect(llm.nodeDesignPrompts[0]).toContain("design approximately 25 ontology nodes");
  });

  it("lets a Codebase have more areas than the old fixed ceiling of twenty", async () => {
    const categories = Array.from({ length: 40 }, (_, index) => `Subsystem ${index}`);
    const llm = new CapturingLLM(categories);
    const builder = new OntologyAutoBuilder(llm);

    await builder.build([new InMemoryDocumentSource(createDocuments(600))]);

    expect(llm.nodeDesignPrompts[0]).toContain("design approximately 40 ontology nodes");
  });

  it("keeps an absolute safety rail on prompt size", async () => {
    const categories = Array.from({ length: 500 }, (_, index) => `Topic ${index}`);
    const llm = new CapturingLLM(categories);
    const builder = new OntologyAutoBuilder(llm);

    await builder.build([new InMemoryDocumentSource(createDocuments(1000))]);

    expect(llm.nodeDesignPrompts[0]).toContain("design approximately 200 ontology nodes");
  });

  it("rejects an explicit override outside the supported range", async () => {
    for (const invalid of [0, -5, 2.5, 201]) {
      const builder = new OntologyAutoBuilder(new CapturingLLM(), invalid);
      await expect(
        builder.build([new InMemoryDocumentSource(createDocuments(16))]),
      ).rejects.toThrow(RangeError);
    }
  });

  it("keeps an explicit target node-count override", async () => {
    const llm = new CapturingLLM();
    const builder = new OntologyAutoBuilder(llm, 8);

    await builder.build([new InMemoryDocumentSource(createDocuments(16))]);

    expect(llm.nodeDesignPrompts).toHaveLength(1);
    expect(llm.nodeDesignPrompts[0]).toContain("design approximately 8 ontology nodes");
  });
});
