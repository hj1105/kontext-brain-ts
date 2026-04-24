import type { Entity } from "../graph/entity.js";
import type { EntityIndex } from "../graph/entity-index.js";
import type { MetaDocument } from "../graph/layered-models.js";
import type { VectorStore } from "./vector-store.js";

/**
 * Hybrid retriever that combines semantic vector similarity with entity
 * mention overlap.
 *
 *   final_score(doc) = entityWeight * entity_score(doc) +
 *                      vectorWeight * vector_similarity(query, doc)
 *
 * Covers the weaknesses of each approach on its own:
 *   - Pure entity matching misses semantic paraphrases ("force unit equal
 *     to 1000 newtons" → doesn't match "sthène" unless sthène is in vocab)
 *   - Pure vector similarity on short queries is noisy and loses precision
 *     when multiple docs are topically close
 *
 * On SQuAD-style generic QA, the hybrid path typically recovers most of
 * the baseline vector-RAG accuracy while retaining entity retrieval's
 * precision boost on queries with strong named-entity anchors.
 *
 * Requires doc-level embeddings to be present in the vectorStore under
 * the key scheme `docPrefix + docId` (default: `"doc:"`).
 */
export class HybridRetriever {
  constructor(
    private readonly entityIndex: EntityIndex,
    private readonly vectorStore: VectorStore,
    private readonly metaDocsByDocId: () => Promise<Map<string, MetaDocument>>,
    private readonly entityWeight = 0.5,
    private readonly vectorWeight = 0.5,
    private readonly expansionDepth = 1,
    private readonly docPrefix = "doc:",
    private readonly vectorTopK = 20,
  ) {}

  async retrieve(
    query: string,
    topK = 5,
  ): Promise<Array<{ doc: MetaDocument; score: number; matched: Entity[] }>> {
    const metaByDoc = await this.metaDocsByDocId();

    // ── entity scoring ──
    const queryEntities = await this.entityIndex.findEntitiesInText(query);
    const entityScores = new Map<string, { score: number; matched: Set<string> }>();

    const seedWeights = new Map<string, number>();
    for (const e of queryEntities) seedWeights.set(e.id, 1.0);
    if (this.expansionDepth > 0) {
      for (const seed of queryEntities) {
        const related = await this.entityIndex.relatedEntities(seed.id, this.expansionDepth);
        for (const { entity, depth } of related) {
          const w = 0.4 / depth;
          const prev = seedWeights.get(entity.id) ?? 0;
          if (w > prev) seedWeights.set(entity.id, w);
        }
      }
    }
    for (const [entityId, w] of seedWeights) {
      const docs = await this.entityIndex.docsForEntity(entityId);
      for (const docId of docs) {
        const cur = entityScores.get(docId) ?? { score: 0, matched: new Set<string>() };
        cur.score += w;
        cur.matched.add(entityId);
        entityScores.set(docId, cur);
      }
    }

    // Normalize entity scores to [0, 1]
    let maxEntity = 0;
    for (const { score } of entityScores.values()) if (score > maxEntity) maxEntity = score;
    if (maxEntity === 0) maxEntity = 1;

    // ── vector scoring ──
    const vectorHits = await this.vectorStore.similaritySearchWithPrefix(
      query,
      this.docPrefix,
      this.vectorTopK,
    );
    // similaritySearchWithPrefix returns the substring after the last ':',
    // which equals docId under our key scheme.
    const vectorScores = new Map<string, number>();
    vectorHits.forEach((docId, idx) => {
      vectorScores.set(docId, 1 - idx / this.vectorTopK);
    });

    // ── merge ──
    const allDocIds = new Set<string>();
    for (const id of entityScores.keys()) allDocIds.add(id);
    for (const id of vectorScores.keys()) allDocIds.add(id);

    const merged: Array<{ doc: MetaDocument; score: number; matched: Entity[] }> = [];
    for (const docId of allDocIds) {
      const doc = metaByDoc.get(docId);
      if (!doc) continue;
      const ent = entityScores.get(docId);
      const entNorm = ent ? ent.score / maxEntity : 0;
      const vec = vectorScores.get(docId) ?? 0;
      const score = this.entityWeight * entNorm + this.vectorWeight * vec;
      const matchedIds = Array.from(ent?.matched ?? []);
      const matched = matchedIds
        .map((id) => queryEntities.find((e) => e.id === id))
        .filter((e): e is Entity => e !== undefined);
      merged.push({ doc, score, matched });
    }

    merged.sort((a, b) => b.score - a.score);
    return merged.slice(0, topK);
  }
}
