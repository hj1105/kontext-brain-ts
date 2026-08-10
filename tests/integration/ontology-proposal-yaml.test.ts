import type { OntologyProposal } from "@kontext-brain/core";
import { YamlOntologyProposalUpdater } from "@kontext-brain/loader";
import { describe, expect, it } from "vitest";

describe("YamlOntologyProposalUpdater", () => {
  it("adds proposed coarse nodes without replacing manual YAML content", async () => {
    const yaml = "# company ontology\nontology:\n  - id: order\n    description: Orders\n";
    const proposal: OntologyProposal = {
      organizationId: "acme",
      proposalKey: "refund",
      suggestedNodeId: "refund",
      description: "Customer refunds",
      resourceIds: ["notion:p1"],
      occurrences: 1,
      status: "open",
      updatedAt: new Date().toISOString(),
    };

    const updated = await new YamlOntologyProposalUpdater().update(yaml, [proposal]);

    expect(updated).toContain("# company ontology");
    expect(updated).toContain("id: order");
    expect(updated).toContain("id: refund");
    expect(updated).toContain("description: Customer refunds");
  });
});
