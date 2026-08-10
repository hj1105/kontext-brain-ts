import type { DataSource, MetaDocument } from "../graph/layered-models.js";
import type { VectorStore } from "./vector-store.js";

/**
 * L2 meta document index port.
 * Stores document titles grouped by ontology node, supports search within a node.
 */
export interface MetaIndexStore {
  index(nodeId: string, documents: readonly MetaDocument[]): Promise<void>;
  replace(nodeId: string, documents: readonly MetaDocument[]): Promise<void>;
  remove(nodeId: string): Promise<void>;
  search(nodeId: string, query: string, topK: number): Promise<MetaDocument[]>;
  listBySource(nodeId: string, source: DataSource): Promise<MetaDocument[]>;
  list(nodeId: string): Promise<MetaDocument[]>;
  all(): Promise<ReadonlyMap<string, readonly MetaDocument[]>>;
}

/** Stable identity for a document across connectors that may reuse resource IDs. */
export function resourceDocumentIdentity(connectorName: string, resourceId: string): string {
  return `${connectorName}\u0000${resourceId}`;
}

export function metaDocumentIdentity(document: MetaDocument): string {
  return resourceDocumentIdentity(document.metadata.connector ?? "", document.id);
}

/** In-memory store with simple keyword scoring. */
export class InMemoryMetaIndexStore implements MetaIndexStore {
  private readonly byNode = new Map<string, MetaDocument[]>();

  async index(nodeId: string, documents: readonly MetaDocument[]): Promise<void> {
    const existing = this.byNode.get(nodeId) ?? [];
    const merged = new Map<string, MetaDocument>();
    for (const d of existing) merged.set(metaDocumentIdentity(d), d);
    for (const d of documents) merged.set(metaDocumentIdentity(d), d);
    this.byNode.set(nodeId, Array.from(merged.values()));
  }

  async replace(nodeId: string, documents: readonly MetaDocument[]): Promise<void> {
    this.byNode.set(nodeId, [...documents]);
  }

  async remove(nodeId: string): Promise<void> {
    this.byNode.delete(nodeId);
  }

  async search(nodeId: string, query: string, topK: number): Promise<MetaDocument[]> {
    const docs = this.byNode.get(nodeId) ?? [];
    if (docs.length === 0) return [];

    const q = query.toLowerCase();
    const queryWords = new Set(q.split(/\s+/).filter((w) => w.length > 1));

    return docs
      .map((doc) => {
        const titleWords = doc.title.toLowerCase().split(/\s+/);
        let overlap = 0;
        for (const w of titleWords) if (queryWords.has(w)) overlap++;
        const score = overlap / (queryWords.size + 1) + doc.score * 0.3;
        return { doc, score };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map((x) => ({ ...x.doc, score: x.score }));
  }

  async listBySource(nodeId: string, source: DataSource): Promise<MetaDocument[]> {
    const docs = this.byNode.get(nodeId) ?? [];
    return docs.filter((d) => d.source === source);
  }

  async list(nodeId: string): Promise<MetaDocument[]> {
    return [...(this.byNode.get(nodeId) ?? [])];
  }

  async all(): Promise<ReadonlyMap<string, readonly MetaDocument[]>> {
    return new Map(Array.from(this.byNode, ([nodeId, documents]) => [nodeId, [...documents]]));
  }
}

/** Vector-based store — uses VectorStore embeddings for ranking. */
export class VectorMetaIndexStore implements MetaIndexStore {
  private readonly byNode = new Map<string, MetaDocument[]>();

  constructor(private readonly vectorStore: VectorStore) {}

  async index(nodeId: string, documents: readonly MetaDocument[]): Promise<void> {
    const existing = this.byNode.get(nodeId) ?? [];
    const merged = new Map<string, MetaDocument>();
    for (const d of existing) merged.set(metaDocumentIdentity(d), d);
    for (const d of documents) merged.set(metaDocumentIdentity(d), d);
    this.byNode.set(nodeId, Array.from(merged.values()));

    // Embed titles for vector search
    for (const doc of documents) {
      const embedding = await this.vectorStore.embed(doc.title);
      await this.vectorStore.upsert(`meta:${nodeId}:${metaDocumentIdentity(doc)}`, embedding, {
        nodeId,
        docId: doc.id,
        title: doc.title,
      });
    }
  }

  async replace(nodeId: string, documents: readonly MetaDocument[]): Promise<void> {
    await this.vectorStore.deleteByPrefix?.(`meta:${nodeId}:`);
    this.byNode.set(nodeId, []);
    await this.index(nodeId, documents);
  }

  async remove(nodeId: string): Promise<void> {
    this.byNode.delete(nodeId);
    await this.vectorStore.deleteByPrefix?.(`meta:${nodeId}:`);
  }

  async search(nodeId: string, query: string, topK: number): Promise<MetaDocument[]> {
    const docs = this.byNode.get(nodeId) ?? [];
    if (docs.length === 0) return [];

    const matches = await this.vectorStore.similaritySearchWithPrefix(
      query,
      `meta:${nodeId}:`,
      topK,
    );
    const byId = new Map(docs.map((d) => [metaDocumentIdentity(d), d]));
    const ordered: MetaDocument[] = [];
    for (const docId of matches) {
      const doc = byId.get(docId);
      if (doc) ordered.push(doc);
    }
    // Fill remaining if short
    if (ordered.length < topK) {
      for (const d of docs) {
        if (!ordered.includes(d)) ordered.push(d);
        if (ordered.length >= topK) break;
      }
    }
    return ordered.slice(0, topK);
  }

  async listBySource(nodeId: string, source: DataSource): Promise<MetaDocument[]> {
    const docs = this.byNode.get(nodeId) ?? [];
    return docs.filter((d) => d.source === source);
  }

  async list(nodeId: string): Promise<MetaDocument[]> {
    return [...(this.byNode.get(nodeId) ?? [])];
  }

  async all(): Promise<ReadonlyMap<string, readonly MetaDocument[]>> {
    return new Map(Array.from(this.byNode, ([nodeId, documents]) => [nodeId, [...documents]]));
  }
}
