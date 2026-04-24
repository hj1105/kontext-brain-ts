/**
 * Entities in kontext-brain support two complementary interpretations:
 *
 * 1. **Instance-of-node** (proper ontological sense):
 *    An Entity is an instance of an OntologyNode (a class), carrying
 *    attribute values defined by the node's schema.
 *      OntologyNode "Database" with schema { version: string, supports_json: boolean }
 *      Entity { nodeId: "Database", name: "PostgreSQL", attributes: { version: "15", supports_json: true } }
 *    Used for structured queries ("find databases where supports_json=true"),
 *    knowledge-graph navigation, and schema-aware retrieval.
 *
 * 2. **Named mention** (NER-style):
 *    An Entity is just a named thing appearing in text, with no class
 *    or schema, used for retrieval by name/alias match.
 *      Entity { name: "JWT", type: "concept", aliases: ["JSON Web Token"] }
 *    Used by `EntityRetriever`, `HybridRetriever`, and alias-matching
 *    extractors.
 *
 * Both uses coexist on the same type via optional fields: `nodeId` and
 * `attributes` are populated for instance-of-node entities; left empty
 * for pure NER-style mentions. Existing code that used only name/alias
 * matching continues to work unchanged.
 */

export type AttrValue = string | number | boolean | readonly string[];

export interface Entity {
  readonly id: string;
  /** Canonical name used in prose (e.g. "JWT", "PostgreSQL"). */
  readonly name: string;
  /**
   * Entity type. For instance-of-node entities this usually matches the
   * parent node's role (e.g. "Database", "Person"). For mentions it's a
   * free-form tag like "concept" / "tool" / "person".
   */
  readonly type: string;
  /** Alternate names that should resolve to this entity. */
  readonly aliases?: readonly string[];
  readonly description?: string;
  /** Priority weight for ranking when multiple entities match. */
  readonly weight?: number;

  // ── Instance-of-node fields (optional) ────────────────────

  /**
   * The OntologyNode this entity is an instance of. When set, the entity
   * is treated as a typed instance of that node's class. When absent,
   * the entity is a standalone mention with no class affiliation.
   */
  readonly nodeId?: string;

  /**
   * Attribute values for this entity. The keys and value types should
   * match the parent node's `attributeSchema` if one is defined.
   * Example: { version: "15", supports_json: true, license: "PostgreSQL" }
   */
  readonly attributes?: Readonly<Record<string, AttrValue>>;
}

export function createEntity(
  init: Partial<Entity> & { id: string; name: string; type: string },
): Entity {
  return {
    id: init.id,
    name: init.name,
    type: init.type,
    aliases: init.aliases ?? [],
    description: init.description,
    weight: init.weight ?? 1.0,
    nodeId: init.nodeId,
    attributes: init.attributes ?? {},
  };
}

/** A document → entity mention edge. */
export interface EntityMention {
  readonly entityId: string;
  readonly docId: string;
  readonly context?: string;
  readonly confidence?: number;
}

/**
 * A typed relation between two entities.
 * Relation types are free-form strings; use consistent vocab per corpus.
 *
 * Common types:
 *   - "uses"          (React USES TypeScript)
 *   - "implements"    (JWT IMPLEMENTS OAuth-bearer-token)
 *   - "alternative_to" (RabbitMQ ALTERNATIVE_TO Kafka)
 *   - "part_of"       (Redis PART_OF Backend stack)
 *   - "depends_on"    (gRPC DEPENDS_ON HTTP/2)
 *   - "instance_of"   (PostgreSQL INSTANCE_OF Database)
 */
export interface EntityRelation {
  readonly from: string;
  readonly to: string;
  readonly type: string;
  readonly weight?: number;
}

// ── Attribute schema types (for OntologyNode.attributeSchema) ────

export type AttrType = "string" | "number" | "boolean" | "string[]";

export interface AttributeSchema {
  readonly [key: string]: AttrType;
}

/**
 * Validates that an entity's attribute values match the schema defined
 * by its parent node. Returns a list of issue strings (empty = valid).
 */
export function validateEntityAttributes(
  attributes: Record<string, AttrValue> | undefined,
  schema: AttributeSchema | undefined,
): string[] {
  if (!schema) return [];
  if (!attributes) return Object.keys(schema).map((k) => `missing attribute: ${k}`);
  const issues: string[] = [];
  for (const [key, type] of Object.entries(schema)) {
    const v = attributes[key];
    if (v === undefined) {
      issues.push(`missing attribute: ${key}`);
      continue;
    }
    if (!matchesType(v, type)) {
      issues.push(`attribute '${key}' should be ${type}, got ${typeof v}`);
    }
  }
  return issues;
}

function matchesType(v: AttrValue, type: AttrType): boolean {
  switch (type) {
    case "string":
      return typeof v === "string";
    case "number":
      return typeof v === "number";
    case "boolean":
      return typeof v === "boolean";
    case "string[]":
      return Array.isArray(v) && v.every((x) => typeof x === "string");
  }
}
