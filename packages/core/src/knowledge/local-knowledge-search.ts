import { DefaultAccessPolicy } from "./access-policy.js";
import type { Principal, ResourceRecord, ResourceSource } from "./domain.js";
import type { ResourceContentStore } from "./ports.js";
import type { SqliteKnowledgeGraphRepository } from "./sqlite-knowledge-graph.js";
import { type TextEmbedder, unitCosine } from "./text-embedder.js";

/**
 * Search over the local knowledge graph: every active chunk the principal may
 * read, scored by the question's terms with a title boost and, when an embedder
 * is configured, by vector similarity as well. Each hit carries its Evidence id
 * so an agent can cite what it found. It reads the same SQLite graph and content
 * store the ontology build and the Task sidecar write, so a question about a
 * decision or a module is answered from the connected repositories and
 * documents instead of a fresh GitHub crawl.
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
  /** Cosine similarity to the question in the embedder's space; absent without vectors. */
  readonly similarity?: number;
}

export interface KnowledgeSearchResult {
  readonly hits: readonly KnowledgeSearchHit[];
  readonly resourcesScanned: number;
  readonly chunksScanned: number;
  /** hybrid when an embedder scored this search; lexical otherwise. */
  readonly mode: "lexical" | "hybrid";
  /** Set when an embedder was configured but could not be used; the search fell back to lexical. */
  readonly embeddingError?: string;
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
/** Hybrid: half the rank from words in common, half from meaning; neither alone decides. */
const LEXICAL_WEIGHT = 0.5;
/** Cosine below this is noise for the models in use; it must not lift a chunk into the hits. */
const MIN_SIMILARITY = 0.6;

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
    private readonly embedder: TextEmbedder | null = null,
  ) {}

  async search(request: KnowledgeSearchRequest): Promise<KnowledgeSearchResult> {
    const limit = Math.max(1, Math.min(request.limit ?? DEFAULT_LIMIT, MAX_LIMIT));
    const question = termFrequencies(request.question);
    const semantic = await this.questionVector(request.question, request.principal.organizationId);
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
    if ((question.size === 0 && !semantic.vector) || chunks.length === 0) {
      return {
        hits: [],
        resourcesScanned: resources.length,
        chunksScanned: chunks.length,
        ...semantic.summary,
      };
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

    const candidates: {
      chunk: IndexedChunk;
      lexical: number;
      matched: string[];
      similarity: number | undefined;
    }[] = [];
    let maxLexical = 0;
    for (const chunk of chunks) {
      const titleTerms = termFrequencies(chunk.resource.title);
      let lexical = 0;
      const matched: string[] = [];
      for (const [term, weight] of question) {
        const inChunk = chunk.terms.get(term) ?? 0;
        const inTitle = titleTerms.get(term) ?? 0;
        if (inChunk === 0 && inTitle === 0) continue;
        matched.push(term);
        lexical +=
          weight * idf(term) * ((inChunk > 0 ? 1 + Math.log(inChunk) : 0) + TITLE_BOOST * inTitle);
      }
      const vector = semantic.vectors?.get(chunk.chunkId);
      const similarity =
        semantic.vector && vector ? unitCosine(semantic.vector, vector) : undefined;
      if (lexical <= 0 && (similarity === undefined || similarity < MIN_SIMILARITY)) continue;
      maxLexical = Math.max(maxLexical, lexical);
      candidates.push({ chunk, lexical, matched, similarity });
    }
    // Why normalize: idf sums and cosines live on different scales; each is
    // scaled to its best candidate so neither side can drown the other.
    const maxSimilarity = Math.max(
      MIN_SIMILARITY,
      ...candidates.map((candidate) => candidate.similarity ?? 0),
    );
    const scored: KnowledgeSearchHit[] = candidates.map(
      ({ chunk, lexical, matched, similarity }) => {
        const lexicalPart = maxLexical > 0 ? lexical / maxLexical : 0;
        const semanticPart =
          similarity === undefined
            ? 0
            : Math.max(0, similarity - MIN_SIMILARITY) /
              Math.max(1e-9, maxSimilarity - MIN_SIMILARITY);
        const score = semantic.vector
          ? LEXICAL_WEIGHT * lexicalPart + (1 - LEXICAL_WEIGHT) * semanticPart
          : lexical;
        return {
          evidenceId: `${chunk.resource.resourceId}|source|${chunk.chunkId}`,
          resourceId: chunk.resource.resourceId,
          chunkId: chunk.chunkId,
          title: chunk.resource.title,
          source: chunk.resource.source,
          ontologyNodeIds: chunk.resource.ontologyNodeIds,
          text:
            chunk.text.length > SNIPPET_CHARS
              ? `${chunk.text.slice(0, SNIPPET_CHARS)}…`
              : chunk.text,
          score,
          matchedTerms: matched,
          ...(similarity === undefined ? {} : { similarity }),
        };
      },
    );
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
      ...semantic.summary,
    };
  }

  /** The question's vector and every stored chunk vector in the same space, or a reason there are none. */
  private async questionVector(
    question: string,
    organizationId: string,
  ): Promise<{
    vector: Float32Array | null;
    vectors: ReadonlyMap<string, Float32Array> | null;
    summary: Pick<KnowledgeSearchResult, "mode" | "embeddingError">;
  }> {
    if (!this.embedder) return { vector: null, vectors: null, summary: { mode: "lexical" } };
    try {
      const vectors = await this.repository.listChunkVectors(organizationId, this.embedder.model);
      if (vectors.size === 0) {
        return {
          vector: null,
          vectors: null,
          summary: {
            mode: "lexical",
            embeddingError: `No chunk vectors for ${this.embedder.model}; run an ontology build or 'embed' first.`,
          },
        };
      }
      const [vector] = await this.embedder.embed([question], "query");
      return { vector: vector ?? null, vectors, summary: { mode: "hybrid" } };
    } catch (error) {
      // Why fall back: a model that failed to load must not turn every question into an error.
      return {
        vector: null,
        vectors: null,
        summary: {
          mode: "lexical",
          embeddingError: error instanceof Error ? error.message : String(error),
        },
      };
    }
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
