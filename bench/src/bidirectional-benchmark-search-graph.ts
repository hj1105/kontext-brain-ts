import type {
  EvidenceHit,
  Principal,
  SearchEdge,
  SearchEdgeObservations,
  SearchGraphPort,
  SearchNode,
  SearchSeed,
} from "@kontext-brain/core";
import { fuseSearchSeeds } from "@kontext-brain/core";
import type { BenchDoc } from "./corpus.js";
import type { KGEdge, KGStore } from "./kg-builder.js";
import { findSeedEntities } from "./kg-retriever.js";

export interface PrecomputedChunkSeeds {
  readonly question: string;
  readonly chunkIds: readonly string[];
}

export interface BenchmarkGraphFanout {
  readonly seedChunks?: number;
  readonly lexicalSeedChunks?: number;
  readonly queryAware?: boolean;
  readonly resourceChunks?: number;
  readonly entityChunks?: number;
  readonly entityFacts?: number;
  readonly chunkEntities?: number;
  readonly chunkFacts?: number;
  /** Query-local rank fusion that rewards agreement across available chunk seed providers. */
  readonly providerConsensus?: boolean;
}

interface IndexedFact {
  readonly id: string;
  readonly edge: KGEdge;
}

/**
 * GraphRAG-Bench projection of the production SearchGraphPort contract.
 *
 * Stored KG triples become fact nodes, entity mentions connect entities to
 * chunks, and chunks lift to their original source resource. Precomputed
 * vector hits implement the optional chunk seed provider used by production.
 */
export class BidirectionalBenchmarkSearchGraph implements SearchGraphPort {
  private readonly docsById: ReadonlyMap<string, BenchDoc>;
  private readonly resourceByChunk = new Map<string, string>();
  private readonly chunksByResource = new Map<string, string[]>();
  private readonly chunksByEntity = new Map<string, string[]>();
  private readonly factsByEntity = new Map<string, IndexedFact[]>();
  private readonly factsByChunk = new Map<string, IndexedFact[]>();
  private readonly factsById = new Map<string, IndexedFact>();
  private readonly chunkSeedsByQuestion: ReadonlyMap<string, readonly string[]>;
  private readonly lexicalRanker: CorpusLexicalRanker;

  constructor(
    private readonly graph: KGStore,
    docs: readonly BenchDoc[],
    precomputedSeeds: readonly PrecomputedChunkSeeds[] = [],
    private readonly fanout: BenchmarkGraphFanout = {},
  ) {
    this.docsById = new Map(docs.map((doc) => [doc.id, doc]));
    this.lexicalRanker = new CorpusLexicalRanker(docs);
    this.chunkSeedsByQuestion = new Map(
      precomputedSeeds.map((seed) => [seed.question, seed.chunkIds]),
    );
    for (const doc of docs) {
      const resourceId = sourceDocumentId(doc);
      this.resourceByChunk.set(doc.id, resourceId);
      append(this.chunksByResource, resourceId, doc.id);
    }
    for (const [chunkId, entityIds] of graph.chunkToEntities) {
      for (const entityId of entityIds) append(this.chunksByEntity, entityId, chunkId);
    }
    graph.edges.forEach((edge, index) => {
      const fact = { id: `kg-edge:${index}`, edge };
      this.factsById.set(fact.id, fact);
      append(this.factsByEntity, edge.src, fact);
      append(this.factsByEntity, edge.dst, fact);
      append(this.factsByChunk, edge.chunkId, fact);
    });
  }

  async seed(question: string, _principal: Principal): Promise<readonly SearchSeed[]> {
    const output: SearchSeed[] = [];
    const lexicalChunkIds = this.lexicalRanker.seedChunkIds(
      question,
      this.fanout.lexicalSeedChunks ?? 0,
    );
    for (let index = 0; index < lexicalChunkIds.length; index++) {
      const chunkId = lexicalChunkIds[index];
      if (!chunkId) continue;
      output.push({
        node: { kind: "chunk", id: chunkId },
        observations: {
          providers: ["benchmark-lexical"],
          query: {
            lexical: { rank: index + 1, candidateCount: lexicalChunkIds.length },
          },
        },
      });
    }
    const chunkIds = this.chunkSeedsByQuestion.get(question) ?? [];
    for (let index = 0; index < Math.min(this.fanout.seedChunks ?? 5, chunkIds.length); index++) {
      const chunkId = chunkIds[index];
      if (!chunkId || !this.docsById.has(chunkId)) continue;
      output.push({
        node: { kind: "chunk", id: chunkId },
        observations: {
          providers: ["benchmark-vector"],
          query: {
            vector: {
              rank: index + 1,
              candidateCount: Math.min(this.fanout.seedChunks ?? 5, chunkIds.length),
            },
          },
        },
      });
    }
    const entitySeeds = Array.from(findSeedEntities(this.graph, question))
      .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
      .slice(0, 10);
    for (const [entityId, score] of entitySeeds) {
      output.push({
        node: { kind: "entity", id: entityId },
        observations: {
          providers: ["benchmark-entity-lexical"],
          query: {
            lexical: {
              rank: entitySeeds.findIndex(([id]) => id === entityId) + 1,
              candidateCount: entitySeeds.length,
              normalizedScore: Math.min(1, Math.max(0, score)),
            },
          },
        },
      });
    }
    const fused = fuseSearchSeeds(output);
    return this.fanout.providerConsensus ? addChunkProviderConsensus(fused) : fused;
  }

  /**
   * Reorders an already bounded graph result without dropping evidence.
   * This is chunk-level ranking only: it never performs BM25 sentence
   * compression or changes the source text supplied to the answer model.
   */
  rankContextChunkIds(question: string, chunkIds: readonly string[]): string[] {
    const lexicalRank = new Map(
      this.lexicalRanker.rankChunkIds(question, chunkIds).map((id, index) => [id, index]),
    );
    return chunkIds
      .map((id, graphRank) => ({
        id,
        graphRank,
        lexicalRank: lexicalRank.get(id) ?? chunkIds.length,
      }))
      .sort((left, right) => {
        const leftRankSum = left.graphRank + left.lexicalRank;
        const rightRankSum = right.graphRank + right.lexicalRank;
        return leftRankSum - rightRankSum || left.graphRank - right.graphRank;
      })
      .map((item) => item.id);
  }

  async neighbors(
    node: SearchNode,
    _question: string,
    _principal: Principal,
  ): Promise<readonly SearchEdge[]> {
    switch (node.kind) {
      case "ontology":
        return [];
      case "resource":
        return this.rankedChunkEdges(
          node,
          this.chunksByResource.get(node.id) ?? [],
          _question,
          this.fanout.resourceChunks ?? 50,
          "deterministic",
        );
      case "chunk":
        return this.chunkNeighbors(node, _question);
      case "entity":
        return this.entityNeighbors(node, _question);
      case "fact":
        return this.factNeighbors(node);
    }
  }

  async evidence(node: SearchNode, _principal: Principal): Promise<readonly EvidenceHit[]> {
    if (node.kind !== "chunk") return [];
    const doc = this.docsById.get(node.id);
    const resourceId = this.resourceByChunk.get(node.id);
    if (!doc || !resourceId) return [];
    return [
      {
        evidenceId: `chunk:${node.id}`,
        chunkId: node.id,
        resourceId,
        text: doc.body,
        observations: {
          origin: "derived",
          support: {
            activeEvidenceCount: this.factsByChunk.get(node.id)?.length ?? 1,
            derivedEvidenceCount: 1,
            curatedEvidenceCount: 0,
            distinctResourceCount: 1,
            conflictCount: 0,
            staleEvidenceCount: 0,
          },
          confidenceApplicability: "not-applicable",
          freshnessApplicability: "not-applicable",
        },
      },
    ];
  }

  private chunkNeighbors(node: SearchNode, question: string): SearchEdge[] {
    const output: SearchEdge[] = [];
    const resourceId = this.resourceByChunk.get(node.id);
    if (resourceId) {
      output.push(
        edge(node, { kind: "resource", id: resourceId }, "lift", {
          structural: { kind: "deterministic" },
          queryApplicability: "not-applicable",
          supportApplicability: "not-applicable",
        }),
      );
    }
    const rankedEntityIds = this.rankEntityIds(
      this.graph.chunkToEntities.get(node.id) ?? [],
      question,
    );
    const entityIds = rankedEntityIds.slice(0, this.fanout.chunkEntities ?? 30);
    for (const [index, entityId] of entityIds.entries()) {
      output.push(
        edge(node, { kind: "entity", id: entityId }, "lift", {
          structural: { kind: "extracted" },
          query: rankedQueryObservation(question, entityId, index + 1, rankedEntityIds.length),
          fanout: { returnedCount: entityIds.length, candidateCount: rankedEntityIds.length },
          supportApplicability: "not-applicable",
        }),
      );
    }
    const rankedFacts = this.rankFacts(this.factsByChunk.get(node.id) ?? [], question);
    const facts = rankedFacts.slice(0, this.fanout.chunkFacts ?? 30);
    for (const [index, fact] of facts.entries()) {
      output.push(
        edge(node, { kind: "fact", id: fact.id }, "lift", {
          structural: { kind: "extracted" },
          query: rankedQueryObservation(question, factText(fact), index + 1, rankedFacts.length),
          fanout: { returnedCount: facts.length, candidateCount: rankedFacts.length },
          support: benchmarkFactSupport(fact),
        }),
      );
    }
    return deduplicateEdges(output);
  }

  private entityNeighbors(node: SearchNode, question: string): SearchEdge[] {
    const output: SearchEdge[] = [];
    output.push(
      ...this.rankedChunkEdges(
        node,
        this.chunksByEntity.get(node.id) ?? [],
        question,
        this.fanout.entityChunks ?? 30,
        "extracted",
      ),
    );
    const rankedFacts = this.rankFacts(this.factsByEntity.get(node.id) ?? [], question);
    const facts = rankedFacts.slice(0, this.fanout.entityFacts ?? 30);
    for (const [index, fact] of facts.entries()) {
      output.push(
        edge(node, { kind: "fact", id: fact.id }, "expand", {
          structural: { kind: "extracted" },
          query: rankedQueryObservation(question, factText(fact), index + 1, rankedFacts.length),
          fanout: { returnedCount: facts.length, candidateCount: rankedFacts.length },
          support: benchmarkFactSupport(fact),
        }),
      );
    }
    return deduplicateEdges(output);
  }

  private factNeighbors(node: SearchNode): SearchEdge[] {
    const fact = this.factsById.get(node.id);
    if (!fact) return [];
    return deduplicateEdges([
      edge(node, { kind: "chunk", id: fact.edge.chunkId }, "ground", {
        structural: { kind: "deterministic" },
        support: benchmarkFactSupport(fact),
      }),
      edge(node, { kind: "entity", id: fact.edge.src }, "expand", {
        structural: { kind: "extracted" },
        support: benchmarkFactSupport(fact),
      }),
      edge(node, { kind: "entity", id: fact.edge.dst }, "expand", {
        structural: { kind: "extracted" },
        support: benchmarkFactSupport(fact),
      }),
    ]);
  }

  private rankChunkIds(chunkIds: readonly string[], question: string): string[] {
    if (!this.fanout.queryAware) return [...chunkIds];
    return this.lexicalRanker.rankChunkIds(question, chunkIds);
  }

  private rankedChunkEdges(
    from: SearchNode,
    chunkIds: readonly string[],
    question: string,
    limit: number,
    structuralKind: "deterministic" | "extracted",
  ): SearchEdge[] {
    const ranked = this.rankChunkIds(chunkIds, question);
    const selected = ranked.slice(0, limit);
    return selected.map((chunkId, index) =>
      edge(from, { kind: "chunk", id: chunkId }, "ground", {
        structural: { kind: structuralKind },
        query: rankedQueryObservation(
          question,
          this.docsById.get(chunkId)?.body ?? chunkId,
          index + 1,
          ranked.length,
        ),
        fanout: { returnedCount: selected.length, candidateCount: ranked.length },
        supportApplicability: "not-applicable",
      }),
    );
  }

  private rankEntityIds(entityIds: readonly string[], question: string): string[] {
    if (!this.fanout.queryAware) return [...entityIds];
    return rankByQuestion(entityIds, question, (entityId) => entityId);
  }

  private rankFacts(facts: readonly IndexedFact[], question: string): IndexedFact[] {
    if (!this.fanout.queryAware) return [...facts];
    return rankByQuestion(
      facts,
      question,
      (fact) => `${fact.edge.src} ${fact.edge.predicate} ${fact.edge.dst}`,
    );
  }
}

function addChunkProviderConsensus(seeds: readonly SearchSeed[]): SearchSeed[] {
  const availableProviders = new Set(
    seeds
      .filter((seed) => seed.node.kind === "chunk")
      .flatMap((seed) => seed.observations?.providers ?? []),
  );
  if (availableProviders.size <= 1) return [...seeds];
  return seeds.map((seed) => {
    if (seed.node.kind !== "chunk") return seed;
    const query = seed.observations?.query;
    if (!query) return seed;
    const providerScores = [
      query.lexical ? rankedConsensusScore(query.lexical.rank, query.lexical.candidateCount) : 0,
      query.vector ? rankedConsensusScore(query.vector.rank, query.vector.candidateCount) : 0,
    ];
    const consensus =
      providerScores.reduce((sum, score) => sum + score, 0) / availableProviders.size;
    return {
      ...seed,
      observations: {
        ...seed.observations,
        query: { ...query, rerankerScore: consensus },
      },
    };
  });
}

function rankedConsensusScore(rank: number, candidateCount: number): number {
  const candidates = Math.max(1, Math.floor(candidateCount));
  const boundedRank = Math.min(candidates, Math.max(1, Math.floor(rank)));
  return candidates === 1 ? 1 : (candidates - boundedRank) / (candidates - 1);
}

interface IndexedLexicalDocument {
  readonly id: string;
  readonly terms: ReadonlySet<string>;
}

/** Exact-token/IDF candidate generator used to protect rare names and terms. */
class CorpusLexicalRanker {
  private readonly documents: readonly IndexedLexicalDocument[];
  private readonly documentFrequency = new Map<string, number>();

  constructor(docs: readonly BenchDoc[]) {
    this.documents = docs.map((doc) => ({
      id: doc.id,
      terms: new Set(lexicalTerms(`${doc.title} ${doc.body}`)),
    }));
    for (const doc of this.documents) {
      for (const term of doc.terms) {
        this.documentFrequency.set(term, (this.documentFrequency.get(term) ?? 0) + 1);
      }
    }
  }

  seedChunkIds(question: string, limit: number): string[] {
    if (limit <= 0) return [];
    const queryTerms = lexicalTerms(question);
    if (queryTerms.length === 0) return [];
    const output: string[] = [];

    // A once-mentioned term such as a product name or proper noun can be the
    // most important query signal. Reserve seed slots for those terms before
    // aggregate overlap ranking so common descriptive words cannot bury them.
    const rareTerms = [...queryTerms]
      .filter((term) => (this.documentFrequency.get(term) ?? 0) <= 3)
      .sort(
        (left, right) =>
          (this.documentFrequency.get(left) ?? 0) - (this.documentFrequency.get(right) ?? 0) ||
          left.localeCompare(right),
      );
    for (const term of rareTerms) {
      const match = this.documents.find((doc) => doc.terms.has(term));
      if (match && !output.includes(match.id)) output.push(match.id);
      if (output.length >= limit) return output;
    }

    for (const id of this.rankChunkIds(
      question,
      this.documents.map((doc) => doc.id),
    )) {
      if (!output.includes(id)) output.push(id);
      if (output.length >= limit) break;
    }
    return output;
  }

  rankChunkIds(question: string, chunkIds: readonly string[]): string[] {
    const queryTerms = lexicalTerms(question);
    if (queryTerms.length === 0) return [...chunkIds];
    const allowed = new Set(chunkIds);
    return this.documents
      .filter((doc) => allowed.has(doc.id))
      .map((doc, index) => ({
        id: doc.id,
        index,
        score: this.score(queryTerms, doc.terms),
      }))
      .sort((left, right) => right.score - left.score || left.index - right.index)
      .map((item) => item.id);
  }

  private score(queryTerms: readonly string[], documentTerms: ReadonlySet<string>): number {
    let matched = 0;
    let idfSum = 0;
    let rarestIdf = 0;
    for (const term of queryTerms) {
      if (!documentTerms.has(term)) continue;
      matched++;
      const frequency = this.documentFrequency.get(term) ?? 0;
      const idf = Math.log((this.documents.length + 1) / (frequency + 1));
      idfSum += idf;
      rarestIdf = Math.max(rarestIdf, idf);
    }
    return idfSum + rarestIdf + matched / queryTerms.length;
  }
}

function sourceDocumentId(doc: BenchDoc): string {
  return doc.title.replace(/\s+chunk\s+\d+$/i, "");
}

function edge(
  from: SearchNode,
  to: SearchNode,
  operation: SearchEdge["operation"],
  observations: SearchEdgeObservations,
): SearchEdge {
  return { from, to, operation, observations };
}

function append<T>(map: Map<string, T[]>, key: string, value: T): void {
  const values = map.get(key) ?? [];
  values.push(value);
  map.set(key, values);
}

function nodeKey(node: SearchNode): string {
  return `${node.kind}:${node.id}`;
}

function deduplicateEdges(edges: readonly SearchEdge[]): SearchEdge[] {
  const best = new Map<string, SearchEdge>();
  for (const candidate of edges) {
    const key = `${candidate.operation}:${nodeKey(candidate.to)}`;
    const previous = best.get(key);
    if (!previous) best.set(key, candidate);
  }
  return Array.from(best.values());
}

function rankedQueryObservation(
  question: string,
  text: string,
  rank: number,
  candidateCount: number,
): NonNullable<SearchEdgeObservations["query"]> {
  return {
    lexical: {
      rank,
      candidateCount: Math.max(1, candidateCount),
      normalizedScore: lexicalRelevance(question, text),
    },
  };
}

function factText(fact: IndexedFact): string {
  return `${fact.edge.src} ${fact.edge.predicate} ${fact.edge.dst}`;
}

function benchmarkFactSupport(_fact: IndexedFact): SearchEdgeObservations["support"] {
  return {
    activeEvidenceCount: 1,
    curatedEvidenceCount: 0,
    derivedEvidenceCount: 1,
    distinctResourceCount: 1,
    conflictCount: 0,
    staleEvidenceCount: 0,
  };
}

function rankByQuestion<T>(
  values: readonly T[],
  question: string,
  text: (value: T) => string,
): T[] {
  return values
    .map((value, index) => ({ value, index, score: lexicalRelevance(question, text(value)) }))
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .map((item) => item.value);
}

function lexicalRelevance(question: string, text: string): number {
  const queryTerms = new Set(
    question
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((term) => term.length >= 3),
  );
  if (queryTerms.size === 0) return 0;
  const normalized = text.toLowerCase();
  let matches = 0;
  for (const term of queryTerms) {
    if (normalized.includes(term)) matches++;
  }
  return matches / queryTerms.size;
}

const LEXICAL_STOP_WORDS = new Set([
  "about",
  "according",
  "after",
  "also",
  "and",
  "are",
  "been",
  "before",
  "being",
  "but",
  "considered",
  "described",
  "did",
  "does",
  "during",
  "for",
  "from",
  "had",
  "has",
  "have",
  "how",
  "into",
  "known",
  "not",
  "of",
  "that",
  "the",
  "their",
  "them",
  "there",
  "these",
  "they",
  "this",
  "those",
  "was",
  "were",
  "what",
  "when",
  "where",
  "which",
  "while",
  "who",
  "whom",
  "why",
  "with",
  "within",
  "would",
]);

function lexicalTerms(value: string): string[] {
  return Array.from(
    new Set(
      value
        .toLowerCase()
        .split(/[^a-z0-9]+/)
        .filter((term) => term.length >= 3 && !LEXICAL_STOP_WORDS.has(term))
        .map(stemTerm),
    ),
  );
}

function stemTerm(term: string): string {
  if (term.length >= 6 && term.endsWith("ies")) return `${term.slice(0, -3)}y`;
  if (term.length >= 6 && term.endsWith("ing")) return term.slice(0, -3);
  if (term.length >= 5 && term.endsWith("ed")) return term.slice(0, -2);
  if (term.length >= 5 && term.endsWith("es")) return term.slice(0, -2);
  if (term.length >= 4 && term.endsWith("s")) return term.slice(0, -1);
  return term;
}
