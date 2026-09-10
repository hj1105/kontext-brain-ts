import { DefaultAccessPolicy } from "./access-policy.js";
import type { Principal, ResourceRecord, ResourceSource } from "./domain.js";
import type { ResourceContentStore } from "./ports.js";
import type { SqliteKnowledgeGraphRepository } from "./sqlite-knowledge-graph.js";

/**
 * Keyword search over the local knowledge graph: every active chunk the
 * principal may read, scored by the question's terms with a title boost, each
 * hit carrying its Evidence id so an agent can cite what it found. It reads the
 * same SQLite graph and content store the ontology build and the Task sidecar
 * write, so a question about a decision or a module is answered from the
 * connected repositories and documents instead of a fresh GitHub crawl.
 */

export interface KnowledgeSearchRequest {
  readonly question: string;
  readonly principal: Principal;
  readonly limit?: number;
  /** Only chunks whose resource sits on one of these ontology nodes. */
  readonly ontologyNodeIds?: readonly string[];
  /** Only resources from these connectors (source names, or "code"). */
  readonly connectorIds?: readonly string[];
}

export interface KnowledgeSearchHit {
  readonly evidenceId: string;
  readonly resourceId: string;
  readonly chunkId: string;
  readonly title: string;
  readonly source: ResourceSource;
  readonly ontologyNodeIds: readonly string[];
  readonly text: string;
  readonly score: number;
  readonly matchedTerms: readonly string[];
}

export interface KnowledgeSearchResult {
  readonly hits: readonly KnowledgeSearchHit[];
  readonly resourcesScanned: number;
  readonly chunksScanned: number;
}

interface IndexedChunk {
  readonly resource: ResourceRecord;
  readonly chunkId: string;
  readonly text: string;
  readonly terms: ReadonlyMap<string, number>;
}

interface IndexedResource {
  readonly contentHash: string;
  readonly chunks: readonly IndexedChunk[];
}

const DEFAULT_LIMIT = 10;
const MAX_LIMIT = 50;
const SNIPPET_CHARS = 700;
/** A title match is a hint, not a chunk that answers; keep it below one body occurrence. */
const TITLE_BOOST = 0.5;

/** Lowercased word-ish tokens; Hangul and CJK runs count as words too. */
export function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? []).filter((token) => token.length >= 2);
}

function termFrequencies(text: string): Map<string, number> {
  const frequencies = new Map<string, number>();
  for (const token of tokenize(text)) {
    frequencies.set(token, (frequencies.get(token) ?? 0) + 1);
  }
  return frequencies;
}

export class LocalKnowledgeSearch {
  private readonly policy = new DefaultAccessPolicy();
  /** Chunk text per resource, keyed by resource id, valid while its contentHash is unchanged. */
  private readonly indexed = new Map<string, IndexedResource>();

  constructor(
    private readonly repository: SqliteKnowledgeGraphRepository,
    private readonly contentStore: ResourceContentStore,
  ) {}

  async search(request: KnowledgeSearchRequest): Promise<KnowledgeSearchResult> {
    const limit = Math.max(1, Math.min(request.limit ?? DEFAULT_LIMIT, MAX_LIMIT));
    const question = termFrequencies(request.question);
    const nodeFilter = request.ontologyNodeIds?.length ? new Set(request.ontologyNodeIds) : null;
    const connectorFilter = request.connectorIds?.length ? new Set(request.connectorIds) : null;
    const organizationId = request.principal.organizationId;

    const resources = (await this.repository.listResources(organizationId)).filter(
      (resource) =>
        resource.status === "active" &&
        this.policy.canAccess(request.principal, resource.acl) &&
        (!nodeFilter || resource.ontologyNodeIds.some((nodeId) => nodeFilter.has(nodeId))) &&
        (!connectorFilter || connectorFilter.has(resource.source.connectorId)),
    );
    const chunks: IndexedChunk[] = [];
    for (const resource of resources) {
      chunks.push(...(await this.chunksOf(organizationId, resource)));
    }
    if (question.size === 0 || chunks.length === 0) {
      return { hits: [], resourcesScanned: resources.length, chunksScanned: chunks.length };
    }

    // Why idf: a term that appears in every chunk of an organization ("the", a
    // repository name) says nothing about which chunk answers the question.
    const documentFrequency = new Map<string, number>();
    for (const chunk of chunks) {
      for (const term of question.keys()) {
        if (chunk.terms.has(term))
          documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1);
      }
    }
    const idf = (term: string) =>
      Math.log(1 + chunks.length / (1 + (documentFrequency.get(term) ?? 0)));

    const scored: KnowledgeSearchHit[] = [];
    for (const chunk of chunks) {
      const titleTerms = termFrequencies(chunk.resource.title);
      let score = 0;
      const matched: string[] = [];
      for (const [term, weight] of question) {
        const inChunk = chunk.terms.get(term) ?? 0;
        const inTitle = titleTerms.get(term) ?? 0;
        if (inChunk === 0 && inTitle === 0) continue;
        matched.push(term);
        score +=
          weight * idf(term) * ((inChunk > 0 ? 1 + Math.log(inChunk) : 0) + TITLE_BOOST * inTitle);
      }
      if (score <= 0) continue;
      scored.push({
        evidenceId: `${chunk.resource.resourceId}|source|${chunk.chunkId}`,
        resourceId: chunk.resource.resourceId,
        chunkId: chunk.chunkId,
        title: chunk.resource.title,
        source: chunk.resource.source,
        ontologyNodeIds: chunk.resource.ontologyNodeIds,
        text:
          chunk.text.length > SNIPPET_CHARS ? `${chunk.text.slice(0, SNIPPET_CHARS)}…` : chunk.text,
        score,
        matchedTerms: matched,
      });
    }
    scored.sort(
      (left, right) =>
        right.score - left.score ||
        right.matchedTerms.length - left.matchedTerms.length ||
        left.evidenceId.localeCompare(right.evidenceId),
    );
    return {
      hits: scored.slice(0, limit),
      resourcesScanned: resources.length,
      chunksScanned: chunks.length,
    };
  }

  private async chunksOf(
    organizationId: string,
    resource: ResourceRecord,
  ): Promise<readonly IndexedChunk[]> {
    const cached = this.indexed.get(resource.resourceId);
    if (cached && cached.contentHash === resource.contentHash) return cached.chunks;
    const records = await this.repository.listChunks(organizationId, resource.resourceId);
    const content = await this.contentStore.get(resource.contentObjectKey);
    const chunks: IndexedChunk[] = [];
    for (const record of records) {
      if (record.status !== "active") continue;
      const text = content?.chunks[record.sourceChunkId];
      if (text === undefined || text.trim() === "") continue;
      chunks.push({ resource, chunkId: record.chunkId, text, terms: termFrequencies(text) });
    }
    this.indexed.set(resource.resourceId, { contentHash: resource.contentHash, chunks });
    return chunks;
  }
}
