import { describe, expect, it } from "vitest";
import {
  DocumentClassifier,
  type LLMAdapter,
  type MCPResourceInfo,
  type OntologyNode,
} from "../src/index.js";

function document(index: number, description = `module ${index}`): MCPResourceInfo {
  return {
    id: `doc-${index}`,
    title: `Document ${index}`,
    description,
    source: "CUSTOM" as MCPResourceInfo["source"],
    connectorName: "repo",
  };
}

const nodes = new Map<string, OntologyNode>([
  ["Billing", { id: "Billing", description: "invoices", weight: 1 } as OntologyNode],
  ["HR", { id: "HR", description: "people", weight: 1 } as OntologyNode],
]);

/** Answers every batch by mapping even positions to Billing and odd ones to HR, last one unmapped. */
class BatchAwareAdapter implements LLMAdapter {
  readonly prompts: string[] = [];
  inFlight = 0;
  peakInFlight = 0;

  async complete(_system: string, context: string, query: string): Promise<string> {
    this.inFlight += 1;
    this.peakInFlight = Math.max(this.peakInFlight, this.inFlight);
    await new Promise((resolve) => setTimeout(resolve, 5));
    this.inFlight -= 1;
    if (query.startsWith("Create new ontology nodes")) {
      return JSON.stringify({ nodes: [{ id: "Misc", description: "leftovers" }], mappings: {} });
    }
    this.prompts.push(context);
    const count = context.split("\n").filter((line) => /^\[\d+\]/.test(line)).length;
    const billing: number[] = [];
    const hr: number[] = [];
    for (let i = 0; i < count - 1; i += 1) (i % 2 === 0 ? billing : hr).push(i);
    return JSON.stringify({ mappings: { Billing: billing, HR: hr }, unmapped: [count - 1] });
  }
}

describe("DocumentClassifier", () => {
  it("classifies in batches and merges the answers by document, not by prompt index", async () => {
    const adapter = new BatchAwareAdapter();
    const classifier = new DocumentClassifier(adapter, undefined, {
      maxDocumentsPerBatch: 4,
      concurrency: 2,
    });
    const documents = Array.from({ length: 10 }, (_, index) => document(index));

    const result = await classifier.classify(documents, nodes);

    // 10 documents in batches of 4 → 3 classification calls; every batch's last document is unmapped.
    expect(adapter.prompts).toHaveLength(3);
    expect(adapter.peakInFlight).toBeLessThanOrEqual(2);
    const ids = (nodeId: string) => (result.mappings.get(nodeId) ?? []).map((d) => d.id);
    expect(ids("Billing")).toEqual(["doc-0", "doc-2", "doc-4", "doc-6", "doc-8"]);
    expect(ids("HR")).toEqual(["doc-1", "doc-5"]);
    expect(result.unmapped.map((d) => d.id)).toEqual(["doc-3", "doc-7", "doc-9"]);
    expect(result.proposals.map((proposal) => proposal.suggestedNodeId)).toEqual(["Misc"]);
  });

  it("keeps each batch's prompt small by trimming long descriptions", async () => {
    const adapter = new BatchAwareAdapter();
    const classifier = new DocumentClassifier(adapter, undefined, { maxDescriptionChars: 20 });
    await classifier.classify([document(0, "x".repeat(500))], nodes);
    expect(adapter.prompts[0]).toContain(`[0] Document 0 — ${"x".repeat(20)}`);
    expect(adapter.prompts[0]).not.toContain("x".repeat(21));
  });
});
