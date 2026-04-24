import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { Edge, OntologyNode } from "../graph/ontology-node.js";
import { OntologyNodeType } from "../graph/ontology-node.js";

export interface SerializableNode {
  readonly id: string;
  readonly description: string;
  readonly weight: number;
  readonly mcpSource?: string | null;
  readonly webSearch: boolean;
}

export interface SerializableEdge {
  readonly from: string;
  readonly to: string;
  readonly weight: number;
}

export interface UserOntologyGraph {
  readonly userId: string;
  readonly nodes: Readonly<Record<string, SerializableNode>>;
  readonly edges: readonly SerializableEdge[];
  readonly mcpSources?: readonly string[];
  readonly lastUpdated?: string;
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
      parentId: null,
      level: 0,
      nodeType: OntologyNodeType.DOMAIN,
      keywords: [],
    });
  }
  return out;
}

export function toEdges(graph: UserOntologyGraph): Edge[] {
  return graph.edges.map((e) => ({ from: e.from, to: e.to, weight: e.weight }));
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
    } catch {
      return { userId, nodes: {}, edges: [] };
    }
  }

  async save(userId: string, graph: UserOntologyGraph): Promise<void> {
    await this.ensureDir();
    const file = path.join(this.dir, `${userId}.json`);
    await fs.writeFile(file, JSON.stringify(graph, null, 2), "utf-8");
  }

  async delete(userId: string): Promise<void> {
    try {
      await fs.unlink(path.join(this.dir, `${userId}.json`));
    } catch {
      // ignore
    }
  }
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
