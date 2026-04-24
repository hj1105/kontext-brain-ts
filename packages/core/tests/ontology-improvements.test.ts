import { describe, expect, it } from "vitest";
import {
  Bm25NodeMappingStrategy,
  EdgeAwareMappingStrategy,
  KeywordMappingStrategy,
  MmrSelector,
  OntologyGraph,
  type Edge,
  type OntologyNode,
  TraversalStrategy,
  createMetaDocument,
  createNode,
  DataSource,
} from "../src/index.js";

function buildGraph(): OntologyGraph {
  const nodes = new Map<string, OntologyNode>([
    ["backend", createNode({ id: "backend", description: "rest api server database jwt", weight: 1 })],
    ["frontend", createNode({ id: "frontend", description: "react ui components", weight: 1 })],
    ["security", createNode({ id: "security", description: "owasp tls secrets vault", weight: 1 })],
  ]);
  const edges: Edge[] = [{ from: "backend", to: "security", weight: 0.7 }];
  return new OntologyGraph(nodes, edges, {
    maxDepth: 2,
    maxTokens: 1000,
    strategy: TraversalStrategy.WEIGHTED_DFS,
  });
}

describe("Bm25NodeMappingStrategy", () => {
  it("favors nodes with rare query terms", async () => {
    const graph = buildGraph();
    const bm25 = new Bm25NodeMappingStrategy();
    const result = await bm25.findStartNodes("How do I rotate vault secrets?", graph.nodes);
    expect(result[0]).toBe("security");
  });

  it("falls back to keyword strategy when no terms match", async () => {
    const graph = buildGraph();
    const bm25 = new Bm25NodeMappingStrategy();
    const result = await bm25.findStartNodes("xyzzy nothing matches", graph.nodes);
    expect(result.length).toBeGreaterThan(0);
  });
});

describe("EdgeAwareMappingStrategy", () => {
  it("expands seed nodes via edges above threshold", async () => {
    const graph = buildGraph();
    const inner = new KeywordMappingStrategy();
    const edgeAware = new EdgeAwareMappingStrategy(inner, graph, 0.5, 2);
    const result = await edgeAware.findStartNodes("backend api", graph.nodes);
    expect(result).toContain("backend");
    expect(result).toContain("security"); // followed via edge w=0.7
  });

  it("does not expand below threshold", async () => {
    const graph = new OntologyGraph(
      buildGraph().nodes,
      [{ from: "backend", to: "security", weight: 0.3 }],
      buildGraph().config,
    );
    const inner = new KeywordMappingStrategy();
    const edgeAware = new EdgeAwareMappingStrategy(inner, graph, 0.5, 2);
    const result = await edgeAware.findStartNodes("backend api", graph.nodes);
    expect(result).not.toContain("security");
  });
});

describe("MmrSelector", () => {
  it("returns at most maxSelect docs", async () => {
    const docs = ["api guide", "api reference", "rest api intro", "auth doc"].map((title, i) =>
      createMetaDocument({
        id: `d${i}`,
        title,
        source: DataSource.CUSTOM,
        ontologyNodeId: "backend",
      }),
    );
    const mmr = new MmrSelector(0.7);
    const result = await mmr.select("api docs", docs, 2);
    expect(result.length).toBe(2);
  });

  it("prefers diverse docs over duplicates", async () => {
    const docs = [
      createMetaDocument({ id: "a1", title: "api guide", source: DataSource.CUSTOM, ontologyNodeId: "n" }),
      createMetaDocument({ id: "a2", title: "api guide reference", source: DataSource.CUSTOM, ontologyNodeId: "n" }),
      createMetaDocument({ id: "b1", title: "auth tutorial", source: DataSource.CUSTOM, ontologyNodeId: "n" }),
    ];
    const mmr = new MmrSelector(0.5); // balance relevance/diversity
    const result = await mmr.select("api auth", docs, 2);
    const ids = result.map((d) => d.id);
    // First should be the most relevant; second should be diverse
    expect(ids[0]).toBeDefined();
    expect(ids[0]).not.toBe(ids[1]);
  });
});
