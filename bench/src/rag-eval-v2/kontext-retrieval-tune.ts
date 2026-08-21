import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { BidirectionalNLayerRetriever, type Principal } from "@kontext-brain/core";
import {
  BidirectionalBenchmarkSearchGraph,
  type BenchmarkGraphFanout,
} from "../bidirectional-benchmark-search-graph.js";
import type { BenchDoc } from "../corpus.js";
import type { KGSerialized, KGStore } from "../kg-builder.js";
import type { BenchmarkQuery, RetrievedEvidence } from "./contracts.js";
import { readJsonLines } from "./jsonl.js";
import { contextPrecisionForQuery, evidenceRecallForQuery } from "./metrics.js";

interface RankedSeeds {
  readonly queryId: string;
  readonly chunkIds: readonly string[];
}

interface Variant {
  readonly name: string;
  readonly fanout: BenchmarkGraphFanout;
}

const principal: Principal = {
  organizationId: "rag-eval",
  subjectId: "retrieval-tuner",
  groupIds: [],
};
const scope = optionalArgument("--scope") ?? "development";
if (!new Set(["development", "holdout", "all"]).has(scope)) {
  throw new Error("--scope must be development, holdout, or all");
}
const queries = readJsonLines<BenchmarkQuery>(requiredArgument("--queries"));
const selectedQueries = queries.filter((query) => {
  const holdout = splitBucket(query.id) < 20;
  return scope === "all" || (scope === "holdout" ? holdout : !holdout);
});
const docs = readJsonLines<{ readonly id: string; readonly body: string }>(
  requiredArgument("--chunks"),
).map<BenchDoc>((doc) => ({ ...doc, title: doc.id }));
const serialized = JSON.parse(readFileSync(requiredArgument("--graph"), "utf8")) as KGSerialized;
const graph: KGStore = {
  entities: new Map(serialized.entities.map((entity) => [entity.id, entity])),
  edges: serialized.edges,
  chunkToEntities: new Map(serialized.chunkToEntities),
};
const queryById = new Map(queries.map((query) => [query.id, query]));
const seeds = readJsonLines<RankedSeeds>(requiredArgument("--seeds")).flatMap((seed) => {
  const query = queryById.get(seed.queryId);
  return query ? [{ question: query.text, chunkIds: seed.chunkIds }] : [];
});
const variants: readonly Variant[] = [
  { name: "baseline-seed5", fanout: { seedChunks: 5 } },
  { name: "seed10", fanout: { seedChunks: 10 } },
  { name: "seed10-query-aware", fanout: { seedChunks: 10, queryAware: true } },
  {
    name: "seed10-lean",
    fanout: leanFanout(10, 0, 10),
  },
  {
    name: "seed10-lean-lex5",
    fanout: leanFanout(10, 5, 10),
  },
  {
    name: "seed20-lean-lex5",
    fanout: leanFanout(20, 5, 10),
  },
  {
    name: "seed20-tight-lex10",
    fanout: leanFanout(20, 10, 5),
  },
  { name: "seed8-fanout8-lex3", fanout: leanFanout(8, 3, 8) },
  { name: "seed8-fanout8-lex5", fanout: leanFanout(8, 5, 8) },
  { name: "seed8-fanout8-lex7", fanout: leanFanout(8, 7, 8) },
  { name: "seed10-fanout5-lex5", fanout: leanFanout(10, 5, 5) },
  { name: "seed10-fanout8-lex3", fanout: leanFanout(10, 3, 8) },
  { name: "seed10-fanout8-lex5", fanout: leanFanout(10, 5, 8) },
  { name: "seed10-fanout8-lex7", fanout: leanFanout(10, 7, 8) },
  { name: "seed10-fanout12-lex5", fanout: leanFanout(10, 5, 12) },
  { name: "seed12-fanout8-lex5", fanout: leanFanout(12, 5, 8) },
  { name: "seed12-fanout10-lex5", fanout: leanFanout(12, 5, 10) },
  { name: "seed15-fanout8-lex5", fanout: leanFanout(15, 5, 8) },
];

for (const variant of variants) {
  const startedAt = performance.now();
  const searchGraph = new BidirectionalBenchmarkSearchGraph(graph, docs, seeds, variant.fanout);
  const retriever = new BidirectionalNLayerRetriever(searchGraph);
  const recalls: number[] = [];
  const precisions: number[] = [];
  const evidenceCounts: number[] = [];
  const stoppedBy = new Map<string, number>();
  for (const query of selectedQueries) {
    const result = await retriever.retrieve({
      question: query.text,
      principal,
      budget: {
        topK: 10,
        maxHops: 8,
        maxKgHops: 3,
        maxVisited: 200,
        maxCandidates: 500,
        timeBudgetMs: 1_200,
        minScore: 0.02,
      },
    });
    const evidence: RetrievedEvidence[] = result.evidence.map((hit, index) => ({
      id: hit.evidenceId,
      sourceId: hit.resourceId,
      text: hit.text,
      score: hit.score,
      rank: index + 1,
      metadata: { chunkId: hit.chunkId },
    }));
    const recall = evidenceRecallForQuery(query, evidence);
    const precision = contextPrecisionForQuery(query, evidence);
    if (recall !== null) recalls.push(recall);
    if (precision !== null) precisions.push(precision);
    evidenceCounts.push(evidence.length);
    stoppedBy.set(result.trace.stoppedBy, (stoppedBy.get(result.trace.stoppedBy) ?? 0) + 1);
  }
  process.stdout.write(
    `${JSON.stringify({
      variant: variant.name,
      scope,
      queries: selectedQueries.length,
      evidenceRecallAtK: mean(recalls),
      contextPrecision: mean(precisions),
      averageEvidence: mean(evidenceCounts),
      stoppedBy: Object.fromEntries(stoppedBy),
      elapsedMs: performance.now() - startedAt,
      fanout: variant.fanout,
    })}\n`,
  );
}

function leanFanout(
  seedChunks: number,
  lexicalSeedChunks: number,
  fanout: number,
): BenchmarkGraphFanout {
  return {
    seedChunks,
    lexicalSeedChunks,
    queryAware: true,
    resourceChunks: fanout,
    entityChunks: fanout,
    entityFacts: fanout,
    chunkEntities: fanout,
    chunkFacts: fanout,
  };
}

function splitBucket(queryId: string): number {
  return createHash("sha256").update(queryId).digest().readUInt32BE(0) % 100;
}

function mean(values: readonly number[]): number | null {
  return values.length === 0
    ? null
    : values.reduce((total, value) => total + value, 0) / values.length;
}

function requiredArgument(name: string): string {
  const value = optionalArgument(name);
  if (!value) throw new Error(`Missing ${name}`);
  return value;
}

function optionalArgument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}
