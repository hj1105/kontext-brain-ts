import type { Edge, OntologyNode } from "@kontext-brain/core";

export const OntologyYamlWriter = {
  write(nodes: readonly OntologyNode[], edges: readonly Edge[]): string {
    const lines: string[] = [];
    lines.push("# Auto-generated ontology");
    lines.push(`# OntologyAutoBuilder — ${nodes.length} nodes, ${edges.length} edges`);
    lines.push("");
    lines.push("ontology:");

    const edgesByFrom = new Map<string, Edge[]>();
    for (const e of edges) {
      const list = edgesByFrom.get(e.from) ?? [];
      list.push(e);
      edgesByFrom.set(e.from, list);
    }

    for (const node of nodes) {
      lines.push(`  - id: ${node.id}`);
      lines.push(`    description: ${node.description}`);
      lines.push(`    weight: ${node.weight}`);
      if (node.parentId) lines.push(`    parentId: ${node.parentId}`);
      if (node.level > 0) lines.push(`    level: ${node.level}`);
      const nodeEdges = edgesByFrom.get(node.id) ?? [];
      if (nodeEdges.length > 0) {
        lines.push("    relates:");
        nodeEdges
          .slice()
          .sort((a, b) => b.weight - a.weight)
          .forEach((edge) => {
            lines.push(`      - to: ${edge.to}`);
            lines.push(`        weight: ${edge.weight}`);
          });
      }
      lines.push("");
    }
    return lines.join("\n");
  },
};
