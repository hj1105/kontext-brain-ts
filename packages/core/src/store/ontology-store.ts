import { randomUUID } from "node:crypto";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { DataSource, MetaDocument } from "../graph/layered-models.js";
import type { OntologyGraph } from "../graph/ontology-graph.js";
import type { Edge, GraphConfig, OntologyNode } from "../graph/ontology-node.js";
import {
  DEFAULT_GRAPH_CONFIG,
  OntologyNodeType,
  TraversalStrategy,
} from "../graph/ontology-node.js";

export interface SerializableNode {
  readonly id: string;
  readonly description: string;
  readonly weight: number;
  readonly mcpSource?: string | null;
  readonly webSearch: boolean;
  readonly parentId?: string | null;
  readonly level?: number;
  readonly nodeType?: OntologyNodeType;
  readonly keywords?: readonly string[];
  readonly attributeSchema?: Readonly<Record<string, "string" | "number" | "boolean" | "string[]">>;
}

export interface SerializableEdge {
  readonly from: string;
  readonly to: string;
  readonly weight: number;
  readonly type?: string;
}

export interface SerializableMetaDocument {
  readonly id: string;
  readonly title: string;
  readonly source: DataSource;
  readonly ontologyNodeId: string;
  readonly url?: string | null;
  readonly score: number;
  readonly metadata: Readonly<Record<string, string>>;
  readonly fetchedAt: string;
}

/**
 * Persisted MCP resource assignment.
 *
 * Resources are not ontology nodes. `nodeIds` records the classification
 * relationship and is the source of truth used to rebuild the meta index.
 */
export interface SerializableResourceRecord {
  readonly connectorName: string;
  readonly resourceId: string;
  readonly title: string;
  readonly description: string;
  readonly source: DataSource;
  readonly nodeIds: readonly string[];
  readonly signature: string;
  readonly lastSeenAt: string;
}

export interface UserOntologyGraph {
  readonly userId: string;
  readonly nodes: Readonly<Record<string, SerializableNode>>;
  readonly edges: readonly SerializableEdge[];
  readonly graphConfig?: GraphConfig;
  readonly metaDocuments?: Readonly<Record<string, readonly SerializableMetaDocument[]>>;
  readonly resources?: readonly SerializableResourceRecord[];
  readonly mcpSources?: readonly string[];
  readonly lastUpdated?: string;
  readonly ontologyContentHash?: string;
}

export function toOntologyNodes(graph: UserOntologyGraph): Map<string, OntologyNode> {
  const out = new Map<string, OntologyNode>();
  for (const [id, n] of Object.entries(graph.nodes)) {
    out.set(id, {
      id: n.id,
      description: n.description,
      weight: n.weight,
      mcpSource: n.mcpSource ?? null,
      webSearch: n.webSearch,
      refBlock: null,
      parentId: n.parentId ?? null,
      level: n.level ?? 0,
      nodeType: n.nodeType ?? OntologyNodeType.DOMAIN,
      keywords: n.keywords ?? [],
      attributeSchema: n.attributeSchema,
    });
  }
  return out;
}

export function toEdges(graph: UserOntologyGraph): Edge[] {
  return graph.edges.map((e) => ({
    from: e.from,
    to: e.to,
    weight: e.weight,
    type: e.type,
  }));
}

export function toGraphConfig(graph: UserOntologyGraph): GraphConfig {
  const config = graph.graphConfig;
  if (!config) return DEFAULT_GRAPH_CONFIG;
  const strategy = Object.values(TraversalStrategy).includes(config.strategy)
    ? config.strategy
    : TraversalStrategy.WEIGHTED_DFS;
  return {
    maxDepth: config.maxDepth,
    maxTokens: config.maxTokens,
    strategy,
  };
}

export function serializeMetaDocument(document: MetaDocument): SerializableMetaDocument {
  return {
    ...document,
    fetchedAt: document.fetchedAt.toISOString(),
  };
}

export function deserializeMetaDocument(document: SerializableMetaDocument): MetaDocument {
  return {
    ...document,
    fetchedAt: new Date(document.fetchedAt),
  };
}

export function createPersistedGraphState(
  userId: string,
  graph: OntologyGraph,
  metaDocuments: ReadonlyMap<string, readonly MetaDocument[]>,
  resources: readonly SerializableResourceRecord[],
  ontologyContentHash?: string,
): UserOntologyGraph {
  const nodes: Record<string, SerializableNode> = {};
  for (const [id, node] of graph.nodes) {
    nodes[id] = {
      id: node.id,
      description: node.description,
      weight: node.weight,
      mcpSource: node.mcpSource ?? null,
      webSearch: node.webSearch,
      parentId: node.parentId ?? null,
      level: node.level,
      nodeType: node.nodeType,
      keywords: node.keywords ?? [],
      attributeSchema: node.attributeSchema,
    };
  }

  const serializedMeta: Record<string, SerializableMetaDocument[]> = {};
  for (const [nodeId, documents] of metaDocuments) {
    serializedMeta[nodeId] = documents.map(serializeMetaDocument);
  }

  return {
    userId,
    nodes,
    edges: graph.edges.map((edge) => ({
      from: edge.from,
      to: edge.to,
      weight: edge.weight,
      type: edge.type,
    })),
    graphConfig: graph.config,
    metaDocuments: serializedMeta,
    resources: [...resources],
    mcpSources: Array.from(new Set(resources.map((resource) => resource.connectorName))),
    lastUpdated: new Date().toISOString(),
    ontologyContentHash,
  };
}

// ── OntologyStore port ────────────────────────────────────────

export interface OntologyStore {
  load(userId: string): Promise<UserOntologyGraph>;
  save(userId: string, graph: UserOntologyGraph): Promise<void>;
  delete(userId: string): Promise<void>;
}

export interface StorageConfig {
  readonly type: string;
  readonly path?: string | null;
  readonly url?: string | null;
}

export const DEFAULT_STORAGE_CONFIG: StorageConfig = {
  type: "memory",
  path: null,
  url: null,
};

export interface OntologyStoreFactory {
  readonly storeType: string;
  create(config: StorageConfig): OntologyStore;
}

// ── In-memory store ───────────────────────────────────────────

export class InMemoryOntologyStore implements OntologyStore {
  private readonly store = new Map<string, UserOntologyGraph>();

  async load(userId: string): Promise<UserOntologyGraph> {
    return this.store.get(userId) ?? { userId, nodes: {}, edges: [] };
  }

  async save(userId: string, graph: UserOntologyGraph): Promise<void> {
    this.store.set(userId, graph);
  }

  async delete(userId: string): Promise<void> {
    this.store.delete(userId);
  }
}

// ── File-based store ──────────────────────────────────────────

export class FileOntologyStore implements OntologyStore {
  constructor(private readonly dir: string) {}

  private async ensureDir(): Promise<void> {
    await fs.mkdir(this.dir, { recursive: true });
  }

  async load(userId: string): Promise<UserOntologyGraph> {
    try {
      const file = path.join(this.dir, `${userId}.json`);
      const data = await fs.readFile(file, "utf-8");
      return JSON.parse(data) as UserOntologyGraph;
    } catch (error) {
      if (isFileNotFound(error)) return { userId, nodes: {}, edges: [] };
      throw error;
    }
  }

  async save(userId: string, graph: UserOntologyGraph): Promise<void> {
    await this.ensureDir();
    const file = path.join(this.dir, `${userId}.json`);
    const temporaryFile = path.join(this.dir, `.${userId}.${process.pid}.${randomUUID()}.tmp`);
    try {
      await fs.writeFile(temporaryFile, JSON.stringify(graph, null, 2), "utf-8");
      await fs.rename(temporaryFile, file);
    } finally {
      await fs.unlink(temporaryFile).catch(() => undefined);
    }
  }

  async delete(userId: string): Promise<void> {
    try {
      await fs.unlink(path.join(this.dir, `${userId}.json`));
    } catch (error) {
      if (!isFileNotFound(error)) throw error;
    }
  }
}

function isFileNotFound(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error as { code?: unknown }).code === "ENOENT"
  );
}

// ── Factories ─────────────────────────────────────────────────

export class InMemoryStoreFactory implements OntologyStoreFactory {
  readonly storeType = "memory";
  create(): OntologyStore {
    return new InMemoryOntologyStore();
  }
}

export class FileStoreFactory implements OntologyStoreFactory {
  readonly storeType = "file";
  create(config: StorageConfig): OntologyStore {
    return new FileOntologyStore(config.path ?? "./kontext-store");
  }
}

// ── Registry ──────────────────────────────────────────────────

export class OntologyStoreRegistry {
  private readonly factories = new Map<string, OntologyStoreFactory>();

  constructor() {
    this.register(new InMemoryStoreFactory());
    this.register(new FileStoreFactory());
  }

  register(factory: OntologyStoreFactory): void {
    this.factories.set(factory.storeType, factory);
  }

  create(config: StorageConfig): OntologyStore {
    const factory = this.factories.get(config.type);
    if (!factory) {
      throw new Error(
        `Unsupported storage type: '${config.type}'. Registered: ${Array.from(this.factories.keys()).join(",")}`,
      );
    }
    return factory.create(config);
  }
}
