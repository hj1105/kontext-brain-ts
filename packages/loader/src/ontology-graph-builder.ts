import type { Edge, OntologyNode, VectorStore } from "@kontext-brain/core";
import {
  OntologyGraph,
  OntologyNodeType,
  TraversalStrategy,
  createNode,
} from "@kontext-brain/core";
import type { z } from "zod";
import type { GraphConfigDtoSchema, OntologyNodeConfig } from "./kontext-config.js";

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
        (OntologyNodeType as Record<string, OntologyNodeType | undefined>)[nodeTypeStr] ??
        OntologyNodeType.DOMAIN;

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
        edges.push({
          from: cfg.id,
          to: rel.to,
          weight: rel.weight ?? 1.0,
          type: rel.type,
        });
      }
      if (cfg.children && cfg.children.length > 0) {
        this.parseRecursive(cfg.children, cfg.id, nodes, edges);
      }
    }
  }
}

export function validateOntologyConfiguration(configs: readonly OntologyNodeConfig[]): void {
  const all = new Map<string, { node: OntologyNodeConfig; parentId: string | null }>();
  const collect = (nodes: readonly OntologyNodeConfig[], implicitParent: string | null) => {
    for (const node of nodes) {
      if (all.has(node.id)) throw new Error(`Duplicate ontology node "${node.id}"`);
      all.set(node.id, { node, parentId: implicitParent ?? node.parentId ?? null });
      collect(node.children ?? [], node.id);
    }
  };
  collect(configs, null);

  for (const [nodeId, entry] of all) {
    if (entry.parentId && !all.has(entry.parentId)) {
      throw new Error(`Ontology node "${nodeId}" references unknown parent "${entry.parentId}"`);
    }
    for (const relation of entry.node.relates ?? []) {
      if (!all.has(relation.to)) {
        throw new Error(
          `Ontology node "${nodeId}" relates to unknown ontology node "${relation.to}"`,
        );
      }
    }
  }

  for (const nodeId of all.keys()) {
    const seen = new Set<string>();
    let current: string | null = nodeId;
    while (current) {
      if (seen.has(current)) throw new Error(`Ontology parent cycle contains "${current}"`);
      seen.add(current);
      current = all.get(current)?.parentId ?? null;
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
      (TraversalStrategy as Record<string, TraversalStrategy | undefined>)[strategyStr] ??
      TraversalStrategy.WEIGHTED_DFS;
    return new OntologyGraph(nodes, edges, {
      maxDepth: graphConfig.maxDepth,
      maxTokens: graphConfig.maxTokens,
      strategy,
    });
  }
}
