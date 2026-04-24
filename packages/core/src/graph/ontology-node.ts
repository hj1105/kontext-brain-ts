/**
 * Ontology graph node.
 *
 * Hierarchical structure:
 *   - parentId: parent node ID (null for root)
 *   - level: hierarchy depth (0 = root)
 */

export enum OntologyNodeType {
  DOMAIN = "DOMAIN",
  CATEGORY = "CATEGORY",
  SUBCATEGORY = "SUBCATEGORY",
  LEAF = "LEAF",
  CUSTOM = "CUSTOM",
}

export interface OntologyNode {
  readonly id: string;
  readonly description: string;
  readonly weight: number;
  readonly mcpSource?: string | null;
  readonly webSearch: boolean;
  readonly refBlock?: (() => Promise<unknown>) | null;
  readonly parentId?: string | null;
  readonly level: number;
  readonly nodeType: OntologyNodeType;
  readonly keywords?: readonly string[];
  /**
   * Schema for attribute values of entities that are instances of this node.
   * Example: { version: "string", supports_json: "boolean", released: "number" }
   *
   * When present, entities with `nodeId === this.id` should carry matching
   * attributes. Validated by `validateEntityAttributes` from `./entity.js`.
   */
  readonly attributeSchema?: Readonly<Record<string, "string" | "number" | "boolean" | "string[]">>;
}

export function createNode(init: Partial<OntologyNode> & { id: string; description: string }): OntologyNode {
  return {
    id: init.id,
    description: init.description,
    weight: init.weight ?? 1.0,
    mcpSource: init.mcpSource ?? null,
    webSearch: init.webSearch ?? false,
    refBlock: init.refBlock ?? null,
    parentId: init.parentId ?? null,
    level: init.level ?? 0,
    nodeType: init.nodeType ?? OntologyNodeType.DOMAIN,
    keywords: init.keywords ?? [],
    attributeSchema: init.attributeSchema,
  };
}

export interface Edge {
  readonly from: string;
  readonly to: string;
  readonly weight: number;
  /**
   * Optional typed relation label ("uses", "part_of", "alternative_to", ...).
   * When absent, the edge is treated as a generic "related" link.
   */
  readonly type?: string;
}

export enum TraversalStrategy {
  BFS = "BFS",
  DFS = "DFS",
  WEIGHTED_DFS = "WEIGHTED_DFS",
}

export interface GraphConfig {
  readonly maxDepth: number;
  readonly maxTokens: number;
  readonly strategy: TraversalStrategy;
}

export const DEFAULT_GRAPH_CONFIG: GraphConfig = {
  maxDepth: 3,
  maxTokens: 8000,
  strategy: TraversalStrategy.WEIGHTED_DFS,
};

export interface ParsedGraph {
  readonly nodes: ReadonlyMap<string, OntologyNode>;
  readonly edges: readonly Edge[];
}
