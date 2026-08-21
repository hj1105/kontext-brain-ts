import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { BidirectionalNLayerRetriever, type Principal } from "@kontext-brain/core";
import { BidirectionalBenchmarkSearchGraph } from "../bidirectional-benchmark-search-graph.js";
import type { BenchDoc } from "../corpus.js";
import type { KGSerialized, KGStore } from "../kg-builder.js";
import type { BenchmarkQuery, RetrievedEvidence } from "./contracts.js";
import { readJsonLines } from "./jsonl.js";
import { CorpusBm25Ranker, fuseRankings } from "./max-existing-stack.js";
import { contextPrecisionForQuery, evidenceRecallForQuery } from "./metrics.js";

interface RankedSeeds {
  readonly queryId: string;
  readonly chunkIds: readonly string[];
}

interface FusionVariant {
  readonly candidateLimit: number;
  readonly vector: number;
  readonly graph: number;
  readonly bm25: number;
  readonly contextRerank: number;
  readonly reciprocalRankConstant: number;
}

interface VariantAccumulator {
  readonly variant: FusionVariant;
  readonly recalls: number[];
  readonly precisions: number[];
}

const principal: Principal = {
  organizationId: "rag-eval",
  subjectId: "v3-max-stack-tuner",
  groupIds: [],
};
const queries = readJsonLines<BenchmarkQuery>(requiredArgument("--queries"));
const developmentQueries = queries.filter((query) => splitBucket(query.id) >= 20);
const docs = readJsonLines<{ readonly id: string; readonly body: string }>(
  requiredArgument("--chunks"),
).map<BenchDoc>((doc) => ({ ...doc, title: doc.id }));
const docsById = new Map(docs.map((doc) => [doc.id, doc]));
const serialized = JSON.parse(readFileSync(requiredArgument("--graph"), "utf8")) as KGSerialized;
const graph: KGStore = {
  entities: new Map(serialized.entities.map((entity) => [entity.id, entity])),
  edges: serialized.edges,
  chunkToEntities: new Map(serialized.chunkToEntities),
};
const rankedSeeds = new Map(
  readJsonLines<RankedSeeds>(requiredArgument("--seeds")).map((seed) => [
    seed.queryId,
    seed.chunkIds,
  ]),
);
const seeds = queries.map((query) => ({
  question: query.text,
  chunkIds: rankedSeeds.get(query.id) ?? [],
}));
const searchGraph = new BidirectionalBenchmarkSearchGraph(graph, docs, seeds, {
  seedChunks: 10,
  lexicalSeedChunks: 5,
  queryAware: true,
  resourceChunks: 10,
  entityChunks: 10,
  entityFacts: 10,
  chunkEntities: 10,
  chunkFacts: 10,
});
const retriever = new BidirectionalNLayerRetriever(searchGraph);
const bm25Ranker = new CorpusBm25Ranker(
  docs.map((doc) => ({ id: doc.id, text: `${doc.title} ${doc.body}` })),
);
const accumulators = variants().map<VariantAccumulator>((variant) => ({
  variant,
  recalls: [],
  precisions: [],
}));
const candidatePoolRecalls: number[] = [];
const startedAt = performance.now();

for (const query of developmentQueries) {
  const result = await retriever.retrieve({
    question: query.text,
    principal,
    budget: {
      topK: 100,
      maxHops: 8,
      maxKgHops: 3,
      maxVisited: 400,
      maxCandidates: 1_000,
      timeBudgetMs: 1_200,
      minScore: 0.02,
    },
  });
  const graphIds = result.evidence.map((hit) => hit.chunkId);
  const vectorIds = rankedSeeds.get(query.id) ?? [];
  const bm25Ids = bm25Ranker.rank(query.text, 100);
  const candidateIds = Array.from(new Set([...graphIds, ...vectorIds, ...bm25Ids]));
  const contextIds = searchGraph.rankContextChunkIds(query.text, candidateIds);
  const poolEvidence = toEvidence(candidateIds, docsById);
  const poolRecall = evidenceRecallForQuery(query, poolEvidence);
  if (poolRecall !== null) candidatePoolRecalls.push(poolRecall);

  for (const accumulator of accumulators) {
    const variant = accumulator.variant;
    const fused = fuseRankings(
      [
        {
          name: "vector",
          ids: vectorIds.slice(0, variant.candidateLimit),
          weight: variant.vector,
        },
        {
          name: "graph",
          ids: graphIds.slice(0, variant.candidateLimit),
          weight: variant.graph,
        },
        {
          name: "bm25",
          ids: bm25Ids.slice(0, variant.candidateLimit),
          weight: variant.bm25,
        },
        {
          name: "context-rerank",
          ids: contextIds.slice(0, variant.candidateLimit * 3),
          weight: variant.contextRerank,
        },
      ],
      10,
      variant.reciprocalRankConstant,
    );
    const evidence = toEvidence(
      fused.map((candidate) => candidate.id),
      docsById,
    );
    const recall = evidenceRecallForQuery(query, evidence);
    const precision = contextPrecisionForQuery(query, evidence);
    if (recall !== null) accumulator.recalls.push(recall);
    if (precision !== null) accumulator.precisions.push(precision);
  }
}

const results = accumulators
  .map((accumulator) => ({
    ...accumulator.variant,
    queries: developmentQueries.length,
    evidenceRecallAtK: mean(accumulator.recalls),
    contextPrecision: mean(accumulator.precisions),
  }))
  .sort(
    (left, right) =>
      right.evidenceRecallAtK - left.evidenceRecallAtK ||
      right.contextPrecision - left.contextPrecision,
  );
process.stdout.write(
  `${JSON.stringify(
    {
      split: "development",
      queries: developmentQueries.length,
      variants: results.length,
      candidatePoolRecall: mean(candidatePoolRecalls),
      elapsedMs: performance.now() - startedAt,
      top: results.slice(0, 30),
    },
    null,
    2,
  )}\n`,
);

function variants(): FusionVariant[] {
  const output: FusionVariant[] = [];
  for (const candidateLimit of [20, 50]) {
    for (const vector of [1, 2, 3]) {
      for (const graphWeight of [1, 2, 3]) {
        for (const bm25 of [0.5, 1]) {
          for (const contextRerank of [0.5, 1]) {
            for (const reciprocalRankConstant of [10, 30, 60]) {
              output.push({
                candidateLimit,
                vector,
                graph: graphWeight,
                bm25,
                contextRerank,
                reciprocalRankConstant,
              });
            }
          }
        }
      }
    }
  }
  return output;
}

function toEvidence(
  ids: readonly string[],
  docsById: ReadonlyMap<string, BenchDoc>,
): RetrievedEvidence[] {
  return ids.flatMap((id, index) => {
    const doc = docsById.get(id);
    return doc
      ? [
          {
            id: `chunk:${id}`,
            sourceId: id,
            text: doc.body,
            score: 1,
            rank: index + 1,
            metadata: { chunkId: id },
          },
        ]
      : [];
  });
}

function splitBucket(queryId: string): number {
  return createHash("sha256").update(queryId).digest().readUInt32BE(0) % 100;
}

function mean(values: readonly number[]): number {
  return values.length === 0
    ? 0
    : values.reduce((total, value) => total + value, 0) / values.length;
}

function requiredArgument(name: string): string {
  const index = process.argv.indexOf(name);
  const value = index >= 0 ? process.argv[index + 1] : undefined;
  if (!value) throw new Error(`Missing ${name}`);
  return value;
}
