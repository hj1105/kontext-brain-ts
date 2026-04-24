import type { Entity } from "../graph/entity.js";
import type { EntityIndex } from "../graph/entity-index.js";
import type { MetaDocument } from "../graph/layered-models.js";

/**
 * Retrieval strategy that ranks docs by entity mentions.
 *
 * Flow:
 *   1. Find entities mentioned in the query
 *   2. Optionally expand via entity relations (depth N BFS)
 *   3. Score each doc by how many of those entities it mentions
 *   4. Return top-K docs
 *
 * Useful when the ontology node routing is too coarse — entity matching
 * cuts across categories to pull in exactly the docs that discuss the
 * named things in the query.
 */
export class EntityRetriever {
  constructor(
    private readonly entityIndex: EntityIndex,
    private readonly metaDocsByDocId: () => Promise<Map<string, MetaDocument>>,
    /** Follow entity→entity relations up to this depth for query expansion. */
    private readonly expansionDepth = 1,
    /** Weight for direct vs expanded entity matches (0..1). */
    private readonly directBoost = 1.0,
    private readonly expandedBoost = 0.4,
  ) {}

  /** Returns docs ranked by entity-match score, highest first. */
  async retrieve(
    query: string,
    topK = 5,
  ): Promise<Array<{ doc: MetaDocument; score: number; matched: Entity[] }>> {
    const queryEntities = await this.entityIndex.findEntitiesInText(query);
    if (queryEntities.length === 0) return [];

    // Expand through relations
    const entityScores = new Map<string, number>();
    for (const e of queryEntities) entityScores.set(e.id, this.directBoost);
    if (this.expansionDepth > 0) {
      for (const seed of queryEntities) {
        const related = await this.entityIndex.relatedEntities(seed.id, this.expansionDepth);
        for (const { entity, depth } of related) {
          const boost = this.expandedBoost / depth;
          const prev = entityScores.get(entity.id) ?? 0;
          if (boost > prev) entityScores.set(entity.id, boost);
        }
      }
    }

    // Score docs
    const docScores = new Map<string, { score: number; matched: Set<string> }>();
    for (const [entityId, score] of entityScores) {
      const docs = await this.entityIndex.docsForEntity(entityId);
      for (const docId of docs) {
        const cur = docScores.get(docId) ?? { score: 0, matched: new Set() };
        cur.score += score;
        cur.matched.add(entityId);
        docScores.set(docId, cur);
      }
    }

    const metaByDoc = await this.metaDocsByDocId();
    const ranked = Array.from(docScores.entries())
      .map(([docId, { score, matched }]) => {
        const doc = metaByDoc.get(docId);
        if (!doc) return null;
        const matchedEntities = Array.from(matched)
          .map((id) => queryEntities.find((e) => e.id === id))
          .filter((e): e is Entity => e !== undefined);
        return { doc, score, matched: matchedEntities };
      })
      .filter((x): x is { doc: MetaDocument; score: number; matched: Entity[] } => x !== null)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    return ranked;
  }
}
