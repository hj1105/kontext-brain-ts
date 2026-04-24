import type { DataSource, MetaDocument } from "../graph/layered-models.js";
import type { VectorStore } from "./vector-store.js";

/**
 * L2 meta document index port.
 * Stores document titles grouped by ontology node, supports search within a node.
 */
export interface MetaIndexStore {
  index(nodeId: string, documents: readonly MetaDocument[]): Promise<void>;
  search(nodeId: string, query: string, topK: number): Promise<MetaDocument[]>;
  listBySource(nodeId: string, source: DataSource): Promise<MetaDocument[]>;
}

/** In-memory store with simple keyword scoring. */
export class InMemoryMetaIndexStore implements MetaIndexStore {
  private readonly byNode = new Map<string, MetaDocument[]>();

  async index(nodeId: string, documents: readonly MetaDocument[]): Promise<void> {
    const existing = this.byNode.get(nodeId) ?? [];
    const merged = new Map<string, MetaDocument>();
    for (const d of existing) merged.set(d.id, d);
    for (const d of documents) merged.set(d.id, d);
    this.byNode.set(nodeId, Array.from(merged.values()));
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
}

/** Vector-based store — uses VectorStore embeddings for ranking. */
export class VectorMetaIndexStore implements MetaIndexStore {
  private readonly byNode = new Map<string, MetaDocument[]>();

  constructor(private readonly vectorStore: VectorStore) {}

  async index(nodeId: string, documents: readonly MetaDocument[]): Promise<void> {
    const existing = this.byNode.get(nodeId) ?? [];
    const merged = new Map<string, MetaDocument>();
    for (const d of existing) merged.set(d.id, d);
    for (const d of documents) merged.set(d.id, d);
    this.byNode.set(nodeId, Array.from(merged.values()));

    // Embed titles for vector search
    for (const doc of documents) {
      const embedding = await this.vectorStore.embed(doc.title);
      await this.vectorStore.upsert(`meta:${nodeId}:${doc.id}`, embedding, {
        nodeId,
        docId: doc.id,
        title: doc.title,
      });
    }
  }

  async search(nodeId: string, query: string, topK: number): Promise<MetaDocument[]> {
    const docs = this.byNode.get(nodeId) ?? [];
    if (docs.length === 0) return [];

    const matches = await this.vectorStore.similaritySearchWithPrefix(
      query,
      `meta:${nodeId}:`,
      topK,
    );
    // matches return the trimmed key (after last ':') = docId
    const byId = new Map(docs.map((d) => [d.id, d]));
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
}
