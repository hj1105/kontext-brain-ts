import type { OntologyGraph } from "./ontology-graph.js";
import type { Edge, OntologyNode } from "./ontology-node.js";
import { TraversalStrategy } from "./ontology-node.js";
import type { TraversalResult, TraversedNode } from "./layered-models.js";

/**
 * Graph traverser with BFS/DFS/WEIGHTED_DFS strategies.
 *
 * Handles hierarchical traversal (children) + horizontal edges (relates).
 */
export class GraphTraverser {
  constructor(private readonly graph: OntologyGraph) {}

  traverse(startNodeIds: readonly string[]): TraversalResult {
    const visited = new Set<string>();
    const resultNodes: TraversedNode[] = [];
    const path: Edge[] = [];
    const maxDepth = this.graph.config.maxDepth;

    const strategy = this.graph.config.strategy;

    switch (strategy) {
      case TraversalStrategy.BFS:
        this.bfs(startNodeIds, maxDepth, visited, resultNodes, path);
        break;
      case TraversalStrategy.DFS:
      case TraversalStrategy.WEIGHTED_DFS:
        for (const startId of startNodeIds) {
          this.dfs(startId, 0, 1.0, maxDepth, visited, resultNodes, path, strategy);
        }
        break;
    }

    return { nodes: resultNodes, path };
  }

  private dfs(
    nodeId: string,
    depth: number,
    cumulativeWeight: number,
    maxDepth: number,
    visited: Set<string>,
    result: TraversedNode[],
    path: Edge[],
    strategy: TraversalStrategy,
  ): void {
    if (depth > maxDepth || visited.has(nodeId)) return;
    const node = this.graph.nodes.get(nodeId);
    if (!node) return;

    visited.add(nodeId);
    result.push({ node, depth, cumulativeWeight });

    // Collect next candidates: hierarchical children + horizontal edges
    const next: Array<{ id: string; weight: number; edge?: Edge }> = [];
    for (const child of this.graph.childrenOf(nodeId)) {
      next.push({ id: child.id, weight: child.weight });
    }
    for (const edge of this.graph.edgesFrom(nodeId)) {
      next.push({ id: edge.to, weight: edge.weight, edge });
    }

    // WEIGHTED_DFS sorts by weight desc
    if (strategy === TraversalStrategy.WEIGHTED_DFS) {
      next.sort((a, b) => b.weight - a.weight);
    }

    for (const { id, weight, edge } of next) {
      if (edge) path.push(edge);
      this.dfs(id, depth + 1, cumulativeWeight * weight, maxDepth, visited, result, path, strategy);
    }
  }

  private bfs(
    startIds: readonly string[],
    maxDepth: number,
    visited: Set<string>,
    result: TraversedNode[],
    path: Edge[],
  ): void {
    interface QueueItem {
      id: string;
      depth: number;
      weight: number;
    }
    const queue: QueueItem[] = startIds.map((id) => ({ id, depth: 0, weight: 1.0 }));

    while (queue.length > 0) {
      const item = queue.shift();
      if (!item) break;
      const { id, depth, weight } = item;
      if (depth > maxDepth || visited.has(id)) continue;
      const node: OntologyNode | undefined = this.graph.nodes.get(id);
      if (!node) continue;
      visited.add(id);
      result.push({ node, depth, cumulativeWeight: weight });

      for (const child of this.graph.childrenOf(id)) {
        queue.push({ id: child.id, depth: depth + 1, weight: weight * child.weight });
      }
      for (const edge of this.graph.edgesFrom(id)) {
        path.push(edge);
        queue.push({ id: edge.to, depth: depth + 1, weight: weight * edge.weight });
      }
    }
  }
}
