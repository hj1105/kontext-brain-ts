import type { Edge, OntologyNode, VectorStore } from "@kontext-brain/core";
import { OntologyGraph, OntologyNodeType, TraversalStrategy, createNode } from "@kontext-brain/core";
import type { GraphConfigDtoSchema, OntologyNodeConfig } from "./kontext-config.js";
import type { z } from "zod";

type GraphConfigDto = z.infer<typeof GraphConfigDtoSchema>;

export class YamlNodeParser {
  parse(configs: readonly OntologyNodeConfig[]): {
    nodes: Map<string, OntologyNode>;
    edges: Edge[];
  } {
    const nodes = new Map<string, OntologyNode>();
    const edges: Edge[] = [];
    this.parseRecursive(configs, null, nodes, edges);
    return { nodes, edges };
  }

  private parseRecursive(
    configs: readonly OntologyNodeConfig[],
    parentId: string | null,
    nodes: Map<string, OntologyNode>,
    edges: Edge[],
  ): void {
    for (const cfg of configs) {
      const nodeTypeStr = (cfg.nodeType ?? "DOMAIN").toUpperCase();
      const nodeType =
        nodeTypeStr in OntologyNodeType
          ? (OntologyNodeType as Record<string, OntologyNodeType>)[nodeTypeStr]!
          : OntologyNodeType.DOMAIN;

      nodes.set(
        cfg.id,
        createNode({
          id: cfg.id,
          description: cfg.description,
          weight: cfg.weight ?? 1.0,
          mcpSource: cfg.mcpSource ?? null,
          webSearch: cfg.webSearch ?? false,
          parentId: parentId ?? cfg.parentId ?? null,
          level: cfg.level ?? 0,
          nodeType,
          keywords: cfg.keywords ?? [],
        }),
      );

      for (const rel of cfg.relates ?? []) {
        edges.push({ from: cfg.id, to: rel.to, weight: rel.weight ?? 1.0 });
      }
      if (cfg.children && cfg.children.length > 0) {
        this.parseRecursive(cfg.children, cfg.id, nodes, edges);
      }
    }
  }
}

export class OntologyEmbedder {
  constructor(private readonly vectorStore: VectorStore) {}
  async embed(nodes: Iterable<OntologyNode>): Promise<void> {
    for (const node of nodes) {
      try {
        const embedding = await this.vectorStore.embed(node.description);
        await this.vectorStore.upsert(node.id, embedding);
      } catch {
        // ignore embedding errors
      }
    }
  }
}

export class OntologyGraphBuilder {
  constructor(
    private readonly embedder: OntologyEmbedder,
    private readonly parser = new YamlNodeParser(),
  ) {}

  async build(
    yamlConfigs: readonly OntologyNodeConfig[],
    graphConfig: GraphConfigDto,
  ): Promise<OntologyGraph> {
    const { nodes, edges } = this.parser.parse(yamlConfigs);
    await this.embedder.embed(nodes.values());
    const strategyStr = graphConfig.strategy.toUpperCase();
    const strategy =
      strategyStr in TraversalStrategy
        ? (TraversalStrategy as Record<string, TraversalStrategy>)[strategyStr]!
        : TraversalStrategy.WEIGHTED_DFS;
    return new OntologyGraph(nodes, edges, {
      maxDepth: graphConfig.maxDepth,
      maxTokens: graphConfig.maxTokens,
      strategy,
    });
  }
}
