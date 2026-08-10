import { describe, expect, it } from "vitest";
import {
  DataSource,
  DocumentClassifier,
  InMemoryOntologyProposalQueue,
  type LLMAdapter,
  PublishOntologyProposalsUseCase,
  createNode,
} from "../src/index.js";

class ProposalLLM implements LLMAdapter {
  async complete(systemPrompt: string): Promise<string> {
    if (systemPrompt.includes("Classify each document")) {
      return JSON.stringify({ mappings: {}, unmapped: [0] });
    }
    return JSON.stringify({
      nodes: [{ id: "refund", description: "Customer refunds" }],
      mappings: { refund: [0] },
    });
  }
}

describe("ontology proposal flow", () => {
  it("keeps unmatched documents out of the active ontology and emits proposal drafts", async () => {
    const classifier = new DocumentClassifier(new ProposalLLM());
    const resource = {
      id: "notion:refund-policy",
      title: "Refund policy",
      description: "How customer refunds are approved",
      source: DataSource.NOTION,
      connectorName: "notion",
    };

    const result = await classifier.classify(
      [resource],
      new Map([["order", createNode({ id: "order", description: "Customer orders" })]]),
    );

    expect(result.newNodes).toEqual([]);
    expect(result.unmapped).toEqual([resource]);
    expect(result.proposals).toEqual([
      {
        suggestedNodeId: "refund",
        description: "Customer refunds",
        resourceIds: ["notion:notion%3Arefund-policy"],
      },
    ]);
  });

  it("deduplicates repeated suggestions in the organization proposal queue", async () => {
    const queue = new InMemoryOntologyProposalQueue();
    const proposal = {
      suggestedNodeId: "refund",
      description: "Customer refunds",
      resourceIds: ["notion:page-1"],
    };

    await queue.enqueue("acme", [proposal]);
    await queue.enqueue("acme", [{ ...proposal, resourceIds: ["notion:page-2"] }]);

    expect(await queue.listOpen("acme")).toMatchObject([
      {
        suggestedNodeId: "refund",
        occurrences: 2,
        resourceIds: ["notion:page-1", "notion:page-2"],
        status: "open",
      },
    ]);
  });

  it("publishes all open proposals as one YAML update and marks them published afterward", async () => {
    const queue = new InMemoryOntologyProposalQueue();
    await queue.enqueue("acme", [
      { suggestedNodeId: "refund", description: "Refunds", resourceIds: ["notion:p1"] },
      { suggestedNodeId: "tax", description: "Taxes", resourceIds: ["notion:p2"] },
    ]);
    const published: string[] = [];
    const useCase = new PublishOntologyProposalsUseCase(
      queue,
      {
        async update(yaml, proposals) {
          return `${yaml}\n${proposals.map((p) => p.suggestedNodeId).join(",")}`;
        },
      },
      {
        async upsert(input) {
          published.push(input.yaml);
          return { url: "https://example.test/pr/1" };
        },
      },
    );

    const result = await useCase.execute("acme", "ontology:");

    expect(result).toMatchObject({ changed: true, url: "https://example.test/pr/1" });
    expect(published).toEqual(["ontology:\nrefund,tax"]);
    expect(await queue.listOpen("acme")).toEqual([]);
  });
});
