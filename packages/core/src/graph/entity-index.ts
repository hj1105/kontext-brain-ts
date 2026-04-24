import type { AttrValue, Entity, EntityMention, EntityRelation } from "./entity.js";

/** A predicate for filtering entity attribute values. */
export type AttrPredicate =
  | { op: "eq"; value: AttrValue }
  | { op: "ne"; value: AttrValue }
  | { op: "gt"; value: number }
  | { op: "gte"; value: number }
  | { op: "lt"; value: number }
  | { op: "lte"; value: number }
  | { op: "in"; values: readonly AttrValue[] }
  | { op: "contains"; value: string } // string contains substring
  | { op: "has"; value: string }; // string[] includes value

/**
 * Query entities by attribute values (instance-of-node use case).
 * Example: find all databases that support JSON released after 2010:
 *   { nodeId: "Database", where: { supports_json: { op: "eq", value: true },
 *                                   released: { op: "gte", value: 2010 } } }
 */
export interface AttributeQuery {
  /** Restrict to instances of this node (class). */
  readonly nodeId?: string;
  /** attribute_key → predicate. All predicates must match (AND). */
  readonly where?: Readonly<Record<string, AttrPredicate>>;
}

/**
 * EntityIndex port — stores entities, their document mentions, typed
 * entity-to-entity relations, and (when entities carry `nodeId` +
 * `attributes`) supports structured attribute queries.
 */
export interface EntityIndex {
  addEntity(entity: Entity): Promise<void>;
  addMention(mention: EntityMention): Promise<void>;
  addRelation(relation: EntityRelation): Promise<void>;

  getEntity(id: string): Promise<Entity | null>;
  allEntities(): Promise<Entity[]>;

  /** Match entities in text via name + alias substring (case-insensitive). */
  findEntitiesInText(text: string): Promise<Entity[]>;

  /** Doc IDs that mention this entity. */
  docsForEntity(entityId: string): Promise<string[]>;

  /** All entities mentioned in this doc. */
  entitiesForDoc(docId: string): Promise<Entity[]>;

  /**
   * BFS over the entity relation graph up to the given depth.
   * Returns { entity, depth } so callers can weight by distance.
   */
  relatedEntities(
    entityId: string,
    depth?: number,
    relationTypes?: readonly string[],
  ): Promise<Array<{ entity: Entity; depth: number; relation: string }>>;

  // ── Instance-of-node queries ────────────────────────────────

  /** All entities that are instances of the given node. */
  entitiesForNode(nodeId: string): Promise<Entity[]>;

  /** Filter entities by attribute predicates. Only considers entities with `nodeId` set. */
  findByAttributes(query: AttributeQuery): Promise<Entity[]>;
}

/** Pure in-memory implementation — good default, swap for DB-backed on large corpora. */
export class InMemoryEntityIndex implements EntityIndex {
  private readonly entities = new Map<string, Entity>();
  private readonly docToEntities = new Map<string, Set<string>>();
  private readonly entityToDocs = new Map<string, Set<string>>();
  private readonly mentionsByEntity = new Map<string, EntityMention[]>();
  private readonly outgoing = new Map<string, EntityRelation[]>();
  private readonly incoming = new Map<string, EntityRelation[]>();
  private patternCache: { patterns: Array<{ entity: Entity; re: RegExp }>; stamp: number } | null = null;

  async addEntity(entity: Entity): Promise<void> {
    this.entities.set(entity.id, entity);
    this.patternCache = null;
  }

  async addMention(mention: EntityMention): Promise<void> {
    if (!this.entities.has(mention.entityId)) return;

    const set1 = this.docToEntities.get(mention.docId) ?? new Set<string>();
    set1.add(mention.entityId);
    this.docToEntities.set(mention.docId, set1);

    const set2 = this.entityToDocs.get(mention.entityId) ?? new Set<string>();
    set2.add(mention.docId);
    this.entityToDocs.set(mention.entityId, set2);

    const list = this.mentionsByEntity.get(mention.entityId) ?? [];
    list.push(mention);
    this.mentionsByEntity.set(mention.entityId, list);
  }

  async addRelation(relation: EntityRelation): Promise<void> {
    const out = this.outgoing.get(relation.from) ?? [];
    out.push(relation);
    this.outgoing.set(relation.from, out);

    const inc = this.incoming.get(relation.to) ?? [];
    inc.push(relation);
    this.incoming.set(relation.to, inc);
  }

  async getEntity(id: string): Promise<Entity | null> {
    return this.entities.get(id) ?? null;
  }

  async allEntities(): Promise<Entity[]> {
    return Array.from(this.entities.values());
  }

  async findEntitiesInText(text: string): Promise<Entity[]> {
    const lc = text.toLowerCase();
    const found = new Map<string, Entity>();
    for (const { entity, re } of this.patterns()) {
      if (re.test(lc)) found.set(entity.id, entity);
    }
    return Array.from(found.values());
  }

  async docsForEntity(entityId: string): Promise<string[]> {
    return Array.from(this.entityToDocs.get(entityId) ?? []);
  }

  async entitiesForDoc(docId: string): Promise<Entity[]> {
    const ids = this.docToEntities.get(docId) ?? new Set<string>();
    return Array.from(ids)
      .map((id) => this.entities.get(id))
      .filter((e): e is Entity => e !== undefined);
  }

  async entitiesForNode(nodeId: string): Promise<Entity[]> {
    const out: Entity[] = [];
    for (const e of this.entities.values()) {
      if (e.nodeId === nodeId) out.push(e);
    }
    return out;
  }

  async findByAttributes(query: AttributeQuery): Promise<Entity[]> {
    const out: Entity[] = [];
    for (const e of this.entities.values()) {
      if (query.nodeId && e.nodeId !== query.nodeId) continue;
      if (!e.nodeId) continue; // attribute queries only apply to typed instances
      if (query.where && !matchesPredicates(e.attributes ?? {}, query.where)) continue;
      out.push(e);
    }
    return out;
  }

  async relatedEntities(
    entityId: string,
    depth = 1,
    relationTypes?: readonly string[],
  ): Promise<Array<{ entity: Entity; depth: number; relation: string }>> {
    const typeFilter = relationTypes && relationTypes.length > 0 ? new Set(relationTypes) : null;
    const result: Array<{ entity: Entity; depth: number; relation: string }> = [];
    const visited = new Set<string>([entityId]);
    interface Frontier { id: string; depth: number; relation: string }
    const queue: Frontier[] = [{ id: entityId, depth: 0, relation: "" }];

    while (queue.length > 0) {
      const cur = queue.shift()!;
      if (cur.depth >= depth) continue;

      const outs = this.outgoing.get(cur.id) ?? [];
      const ins = this.incoming.get(cur.id) ?? [];
      for (const rel of [...outs, ...ins.map((r) => ({ ...r, from: r.to, to: r.from }))]) {
        if (typeFilter && !typeFilter.has(rel.type)) continue;
        if (visited.has(rel.to)) continue;
        const next = this.entities.get(rel.to);
        if (!next) continue;
        visited.add(rel.to);
        result.push({ entity: next, depth: cur.depth + 1, relation: rel.type });
        queue.push({ id: rel.to, depth: cur.depth + 1, relation: rel.type });
      }
    }
    return result;
  }

  private patterns(): Array<{ entity: Entity; re: RegExp }> {
    if (this.patternCache) return this.patternCache.patterns;
    const items: Array<{ entity: Entity; re: RegExp }> = [];
    for (const entity of this.entities.values()) {
      const names = [entity.name, ...(entity.aliases ?? [])];
      for (const n of names) {
        const escaped = n.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        // Word boundary helps avoid partial matches ("api" matching "apis")
        const re = new RegExp(`\\b${escaped}\\b`, "i");
        items.push({ entity, re });
      }
    }
    this.patternCache = { patterns: items, stamp: Date.now() };
    return items;
  }
}

function matchesPredicates(
  attrs: Record<string, AttrValue>,
  where: Record<string, AttrPredicate>,
): boolean {
  for (const [key, pred] of Object.entries(where)) {
    const v = attrs[key];
    if (v === undefined) return false;
    if (!matchesPredicate(v, pred)) return false;
  }
  return true;
}

function matchesPredicate(v: AttrValue, pred: AttrPredicate): boolean {
  switch (pred.op) {
    case "eq":
      return deepEquals(v, pred.value);
    case "ne":
      return !deepEquals(v, pred.value);
    case "gt":
      return typeof v === "number" && v > pred.value;
    case "gte":
      return typeof v === "number" && v >= pred.value;
    case "lt":
      return typeof v === "number" && v < pred.value;
    case "lte":
      return typeof v === "number" && v <= pred.value;
    case "in":
      return pred.values.some((x) => deepEquals(v, x));
    case "contains":
      return typeof v === "string" && v.toLowerCase().includes(pred.value.toLowerCase());
    case "has":
      return Array.isArray(v) && v.includes(pred.value);
  }
}

function deepEquals(a: AttrValue, b: AttrValue): boolean {
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    return a.every((x, i) => x === b[i]);
  }
  return a === b;
}
