import type { OntologyGraph } from "../graph/ontology-graph.js";
import type { OntologyNode } from "../graph/ontology-node.js";
import type { NodeMappingStrategy } from "./node-mapping-strategy.js";

/**
 * Wraps any inner mapping strategy and expands the result via ontology edges.
 *
 * If the inner strategy maps a query to node A, and node A has an outgoing
 * edge A → B with weight >= edgeThreshold, then B is appended to the start
 * nodes too. Useful when queries cross-cut ontology silos:
 *
 *   "How do I secure my JWT API?"  →  inner finds "backend"  →
 *   edge backend → security (w=0.7) is followed → adds "security"
 *
 * Edge weights act as confidence; lower the threshold to be more inclusive.
 */
export class EdgeAwareMappingStrategy implements NodeMappingStrategy {
  readonly strategyName: string;

  constructor(
    private readonly inner: NodeMappingStrategy,
    private readonly graph: OntologyGraph,
    private readonly edgeThreshold = 0.5,
    private readonly maxAdditional = 2,
  ) {
    this.strategyName = `edge-aware(${inner.strategyName})`;
  }

  async findStartNodes(
    query: string,
    nodes: ReadonlyMap<string, OntologyNode>,
  ): Promise<string[]> {
    const seeds = await this.inner.findStartNodes(query, nodes);
    const result = new Set(seeds);
    let added = 0;

    for (const seedId of seeds) {
      // Outgoing edges
      for (const edge of this.graph.edgesFrom(seedId)) {
        if (added >= this.maxAdditional) break;
        if (edge.weight < this.edgeThreshold) continue;
        if (result.has(edge.to)) continue;
        if (!nodes.has(edge.to)) continue;
        result.add(edge.to);
        added++;
      }

      // Incoming edges (related nodes pointing at this seed)
      for (const edge of this.graph.edges) {
        if (added >= this.maxAdditional) break;
        if (edge.to !== seedId) continue;
        if (edge.weight < this.edgeThreshold) continue;
        if (result.has(edge.from)) continue;
        if (!nodes.has(edge.from)) continue;
        result.add(edge.from);
        added++;
      }
    }

    return Array.from(result);
  }
}
