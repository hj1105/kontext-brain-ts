import type { OntologyProposal, OntologyYamlUpdater } from "@kontext-brain/core";
import { isMap, isSeq, parseDocument } from "yaml";

export class YamlOntologyProposalUpdater implements OntologyYamlUpdater {
  async update(yaml: string, proposals: readonly OntologyProposal[]): Promise<string> {
    const document = parseDocument(yaml);
    if (document.errors.length > 0) throw document.errors[0];
    const ontology = document.get("ontology", true);
    if (!isSeq(ontology)) throw new Error('Ontology YAML requires an "ontology" sequence');
    const existingIds = new Set<string>();
    for (const item of ontology.items) {
      if (!isMap(item)) continue;
      const id = item.get("id");
      if (typeof id === "string") existingIds.add(id);
    }
    for (const proposal of proposals) {
      if (existingIds.has(proposal.suggestedNodeId)) continue;
      ontology.add({
        id: proposal.suggestedNodeId,
        description: proposal.description,
        weight: 0.7,
      });
      existingIds.add(proposal.suggestedNodeId);
    }
    return document.toString();
  }
}
