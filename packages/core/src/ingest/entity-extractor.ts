import type { Entity, EntityRelation } from "../graph/entity.js";
import { createEntity } from "../graph/entity.js";
import type { LLMAdapter } from "../query/llm-adapter.js";

/**
 * Extracts entities (and optionally typed relations) from text.
 *
 * Two built-in strategies:
 *   - `AliasEntityExtractor`: match a known vocabulary of entity names/aliases
 *     against the text. Fast, deterministic, no LLM cost. Good for curated
 *     corpora where the entity set is known up front.
 *   - `LLMEntityExtractor`: LLM generates entity + relation JSON for arbitrary
 *     text. More flexible, slower, costs tokens. Good for bootstrap / new
 *     corpora where you don't yet know the vocabulary.
 */
export interface ExtractedEntities {
  readonly entities: readonly Entity[];
  readonly relations: readonly EntityRelation[];
}

export interface EntityExtractor {
  extract(text: string): Promise<ExtractedEntities>;
}

/**
 * Matches against a known entity vocabulary.
 * Zero LLM cost; returns the subset of known entities mentioned in the text.
 * Doesn't produce new relations (use LLM extractor for that).
 */
export class AliasEntityExtractor implements EntityExtractor {
  constructor(private readonly vocabulary: readonly Entity[]) {}

  async extract(text: string): Promise<ExtractedEntities> {
    const lc = text.toLowerCase();
    const found: Entity[] = [];
    for (const entity of this.vocabulary) {
      const names = [entity.name, ...(entity.aliases ?? [])];
      for (const n of names) {
        const escaped = n.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        const re = new RegExp(`\\b${escaped}\\b`, "i");
        if (re.test(lc)) {
          found.push(entity);
          break;
        }
      }
    }
    return { entities: found, relations: [] };
  }
}

/**
 * LLM-based extraction. Returns JSON with entities and (optionally) typed
 * relations. Pair with deduplication logic in the caller if you extract
 * from many docs.
 */
export class LLMEntityExtractor implements EntityExtractor {
  constructor(
    private readonly adapter: LLMAdapter,
    private readonly maxCharsPerCall = 3000,
  ) {}

  async extract(text: string): Promise<ExtractedEntities> {
    const snippet = text.slice(0, this.maxCharsPerCall);
    const system = `
Extract named entities (tools, concepts, frameworks, protocols, people, orgs)
from the text and their typed relations.

Rules:
- Each entity: id (lowercase kebab-case), name (canonical), type, aliases (optional)
- Each relation: from (entity id), to (entity id), type (lowercase: uses, depends_on,
  alternative_to, part_of, implements, extends)
- Only include entities actually mentioned in the text
- 3-10 entities per call

Return JSON only:
{
  "entities": [
    {"id": "jwt", "name": "JWT", "type": "concept", "aliases": ["JSON Web Token"]}
  ],
  "relations": [
    {"from": "jwt", "to": "oauth2", "type": "alternative_to"}
  ]
}
    `.trim();

    const response = await this.adapter.complete(system, "", snippet);
    try {
      const clean = response.trim().replace(/^```json/, "").replace(/```$/, "").trim();
      const parsed = JSON.parse(clean) as {
        entities?: Array<{ id: string; name: string; type?: string; aliases?: string[] }>;
        relations?: Array<{ from: string; to: string; type: string; weight?: number }>;
      };
      const entities = (parsed.entities ?? []).map((e) =>
        createEntity({
          id: e.id,
          name: e.name,
          type: e.type ?? "concept",
          aliases: e.aliases ?? [],
        }),
      );
      const relations: EntityRelation[] = (parsed.relations ?? []).map((r) => ({
        from: r.from,
        to: r.to,
        type: r.type,
        weight: r.weight ?? 0.8,
      }));
      return { entities, relations };
    } catch {
      return { entities: [], relations: [] };
    }
  }
}

/**
 * Combine alias matching (cheap, known vocab) with LLM fallback for
 * sentences that had no alias hit. Gets most of the speed of alias with
 * LLM's flexibility for unknown terms.
 */
export class HybridEntityExtractor implements EntityExtractor {
  constructor(
    private readonly alias: AliasEntityExtractor,
    private readonly llm: LLMEntityExtractor,
    private readonly llmFallbackThreshold = 1,
  ) {}

  async extract(text: string): Promise<ExtractedEntities> {
    const aliasResult = await this.alias.extract(text);
    if (aliasResult.entities.length >= this.llmFallbackThreshold) return aliasResult;
    const llmResult = await this.llm.extract(text);
    // Merge — prefer alias matches (canonical vocab) and add novel LLM entities
    const merged = new Map<string, Entity>();
    for (const e of aliasResult.entities) merged.set(e.id, e);
    for (const e of llmResult.entities) if (!merged.has(e.id)) merged.set(e.id, e);
    return {
      entities: Array.from(merged.values()),
      relations: llmResult.relations,
    };
  }
}
