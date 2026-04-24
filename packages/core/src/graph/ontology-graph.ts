import type { Edge, GraphConfig, OntologyNode } from "./ontology-node.js";
import { DEFAULT_GRAPH_CONFIG } from "./ontology-node.js";

export class OntologyGraph {
  constructor(
    public readonly nodes: ReadonlyMap<string, OntologyNode>,
    public readonly edges: readonly Edge[],
    public readonly config: GraphConfig = DEFAULT_GRAPH_CONFIG,
  ) {}

  edgesFrom(nodeId: string): Edge[] {
    return this.edges.filter((e) => e.from === nodeId);
  }

  /** Direct children of a node, sorted by weight desc. */
  childrenOf(nodeId: string): OntologyNode[] {
    const children: OntologyNode[] = [];
    for (const node of this.nodes.values()) {
      if (node.parentId === nodeId) children.push(node);
    }
    return children.sort((a, b) => b.weight - a.weight);
  }

  /** Nodes at a specific hierarchy level. */
  nodesAtLevel(level: number): OntologyNode[] {
    return Array.from(this.nodes.values()).filter((n) => n.level === level);
  }

  /** Root nodes (no parent), sorted by weight desc. */
  roots(): OntologyNode[] {
    return Array.from(this.nodes.values())
      .filter((n) => !n.parentId)
      .sort((a, b) => b.weight - a.weight);
  }

  /** Ancestor path from root to the given node (inclusive). */
  ancestorPath(nodeId: string): OntologyNode[] {
    const path: OntologyNode[] = [];
    let current = this.nodes.get(nodeId);
    while (current) {
      path.unshift(current);
      current = current.parentId ? this.nodes.get(current.parentId) : undefined;
    }
    return path;
  }

  /** All descendants recursively. */
  descendants(nodeId: string): OntologyNode[] {
    const result: OntologyNode[] = [];
    const dfs = (id: string) => {
      for (const child of this.childrenOf(id)) {
        result.push(child);
        dfs(child.id);
      }
    };
    dfs(nodeId);
    return result;
  }

  describeHierarchy(): string {
    const lines: string[] = ["=== Ontology Hierarchy ==="];
    const printNode = (node: OntologyNode, indent: number) => {
      lines.push(`${"  ".repeat(indent)}- ${node.id} [${node.nodeType}] w=${node.weight}`);
      for (const child of this.childrenOf(node.id)) {
        printNode(child, indent + 1);
      }
    };
    for (const root of this.roots()) {
      printNode(root, 0);
    }
    lines.push("");
    lines.push(`Total: ${this.nodes.size} nodes / ${this.edges.length} edges`);
    return lines.join("\n");
  }
}
