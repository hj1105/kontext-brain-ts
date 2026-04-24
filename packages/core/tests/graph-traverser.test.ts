import { describe, expect, it } from "vitest";
import {
  type Edge,
  GraphTraverser,
  OntologyGraph,
  type OntologyNode,
  TraversalStrategy,
  createNode,
} from "../src/index.js";

function buildGraph(): OntologyGraph {
  const nodes = new Map<string, OntologyNode>([
    ["a", createNode({ id: "a", description: "alpha", weight: 1.0 })],
    ["b", createNode({ id: "b", description: "beta", weight: 0.8 })],
    ["c", createNode({ id: "c", description: "gamma", weight: 0.9, parentId: "a", level: 1 })],
    ["d", createNode({ id: "d", description: "delta", weight: 0.7, parentId: "a", level: 1 })],
  ]);
  const edges: Edge[] = [
    { from: "a", to: "b", weight: 0.9 },
  ];
  return new OntologyGraph(nodes, edges, {
    maxDepth: 3,
    maxTokens: 1000,
    strategy: TraversalStrategy.WEIGHTED_DFS,
  });
}

describe("GraphTraverser", () => {
  it("visits all reachable nodes from start", () => {
    const graph = buildGraph();
    const result = new GraphTraverser(graph).traverse(["a"]);
    const ids = result.nodes.map((n) => n.node.id);
    expect(ids).toContain("a");
    expect(ids).toContain("b");
    expect(ids).toContain("c");
    expect(ids).toContain("d");
  });

  it("respects maxDepth", () => {
    const graph = new OntologyGraph(buildGraph().nodes, buildGraph().edges, {
      maxDepth: 0,
      maxTokens: 1000,
      strategy: TraversalStrategy.WEIGHTED_DFS,
    });
    const result = new GraphTraverser(graph).traverse(["a"]);
    expect(result.nodes.map((n) => n.node.id)).toEqual(["a"]);
  });

  it("does not revisit nodes (cycle safety)", () => {
    const nodes = new Map<string, OntologyNode>([
      ["a", createNode({ id: "a", description: "", weight: 1.0 })],
      ["b", createNode({ id: "b", description: "", weight: 1.0 })],
    ]);
    const edges: Edge[] = [
      { from: "a", to: "b", weight: 1.0 },
      { from: "b", to: "a", weight: 1.0 },
    ];
    const graph = new OntologyGraph(nodes, edges);
    const result = new GraphTraverser(graph).traverse(["a"]);
    const visitCount = result.nodes.filter((n) => n.node.id === "a").length;
    expect(visitCount).toBe(1);
  });
});
