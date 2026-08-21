import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import {
  AdaptiveKnowledgeEnricher,
  type ExtractedEntity,
  type ExtractedFact,
  type LLMAdapter,
  type ResourceSnapshot,
} from "@kontext-brain/core";
import type { KGEntity, KGSerialized } from "../kg-builder.js";
import { CodexJsonClient } from "./codex-json.js";
import { readJsonLines, writeJsonAtomic } from "./jsonl.js";
import { DEFAULT_RAG_EVAL_MANIFEST } from "./manifest.js";

interface ChunkInput {
  readonly id: string;
  readonly body: string;
}

interface BuildOptions {
  readonly chunksPath: string;
  readonly baseGraphPath: string;
  readonly outputGraphPath: string;
  readonly cacheDirectory: string;
  readonly concurrency: number;
}

interface CachedCompletion {
  readonly schemaVersion: 1;
  readonly value: string;
  readonly inputTokens: number | null;
  readonly outputTokens: number | null;
  readonly latencyMs: number;
}

interface ResourceProjection {
  readonly resourceId: string;
  readonly entities: readonly ExtractedEntity[];
  readonly facts: readonly ExtractedFact[];
  readonly capabilities: readonly string[];
  readonly processedWindows: number;
  readonly hypothesisCount: number;
  readonly validationFailureCount: number;
}

class CheckpointingCodexLlmAdapter implements LLMAdapter {
  private readonly client = new CodexJsonClient();
  private completed = 0;
  private cacheHits = 0;

  constructor(private readonly cacheDirectory: string) {
    mkdirSync(cacheDirectory, { recursive: true });
  }

  async complete(systemPrompt: string, context: string, query: string): Promise<string> {
    const digest = createHash("sha256")
      .update("adaptive-eece-v1")
      .update("\0")
      .update(systemPrompt)
      .update("\0")
      .update(context)
      .update("\0")
      .update(query)
      .digest("hex");
    const path = resolve(this.cacheDirectory, `${digest}.json`);
    if (existsSync(path)) {
      const cached = JSON.parse(readFileSync(path, "utf8")) as CachedCompletion;
      if (cached.schemaVersion === 1 && typeof cached.value === "string") {
        this.cacheHits += 1;
        return cached.value;
      }
    }

    const result = await this.client.completeText(
      {
        model: DEFAULT_RAG_EVAL_MANIFEST.models.answer.model,
        reasoningEffort: DEFAULT_RAG_EVAL_MANIFEST.models.answer.reasoningEffort ?? "medium",
      },
      systemPrompt,
      context,
      query,
    );
    const cached: CachedCompletion = {
      schemaVersion: 1,
      value: result.value,
      inputTokens: result.inputTokens,
      outputTokens: result.outputTokens,
      latencyMs: result.latencyMs,
    };
    writeJsonAtomic(path, cached);
    appendFileSync(
      resolve(this.cacheDirectory, "usage.jsonl"),
      `${JSON.stringify({ digest, ...cached })}\n`,
      "utf8",
    );
    this.completed += 1;
    process.stderr.write(
      `[adaptive-kg] llm completed=${this.completed} cacheHits=${this.cacheHits}\n`,
    );
    return result.value;
  }
}

async function main(): Promise<void> {
  const options = parseOptions(process.argv.slice(2));
  const startedAt = Date.now();
  const chunks = readJsonLines<ChunkInput>(options.chunksPath);
  const grouped = groupChunks(chunks);
  const llm = new CheckpointingCodexLlmAdapter(resolve(options.cacheDirectory, "llm"));
  const enricher = new AdaptiveKnowledgeEnricher(llm, {
    concurrency: options.concurrency,
    maxExtractionAttempts: 5,
    validationFailurePolicy: "empty-window",
  });
  const projections = await mapWithConcurrency(grouped, 2, async (resource, index) => {
    const [resourceId, resourceChunks] = resource;
    const checkpointPath = resolve(
      options.cacheDirectory,
      "resources",
      `${createHash("sha256").update(resourceId).digest("hex")}.json`,
    );
    if (existsSync(checkpointPath)) {
      const projection = JSON.parse(readFileSync(checkpointPath, "utf8")) as ResourceProjection;
      process.stderr.write(
        `[adaptive-kg] resource cached=${index + 1}/${grouped.length} id=${resourceId}\n`,
      );
      return {
        ...projection,
        validationFailureCount: projection.validationFailureCount ?? 0,
      };
    }
    const result = await enricher.enrich(resourceSnapshot(resourceId, resourceChunks));
    const projection: ResourceProjection = {
      resourceId,
      entities: result.snapshot.entities ?? [],
      facts: result.snapshot.facts ?? [],
      capabilities: result.capabilities,
      processedWindows: result.processedWindows,
      hypothesisCount: result.hypothesisCount,
      validationFailureCount: result.validationFailureCount,
    };
    writeJsonAtomic(checkpointPath, projection);
    process.stderr.write(
      `[adaptive-kg] resource complete=${index + 1}/${grouped.length} id=${resourceId} windows=${projection.processedWindows} entities=${projection.entities.length} facts=${projection.facts.length} hypotheses=${projection.hypothesisCount} validationFailures=${projection.validationFailureCount}\n`,
    );
    return projection;
  });

  const base = JSON.parse(readFileSync(options.baseGraphPath, "utf8")) as KGSerialized;
  const augmented = augmentGraph(base, projections);
  mkdirSync(dirname(options.outputGraphPath), { recursive: true });
  writeFileSync(options.outputGraphPath, `${JSON.stringify(augmented)}\n`, "utf8");
  const usage = summarizeUsage(resolve(options.cacheDirectory, "llm"));
  const report = {
    schemaVersion: 1,
    sourceCommit: "996b8bb",
    goldAccess: false,
    inputs: {
      chunksPath: options.chunksPath,
      baseGraphPath: options.baseGraphPath,
      chunks: chunks.length,
      resources: grouped.length,
    },
    policy: {
      chunksPerWindow: 6,
      overlapChunks: 1,
      maxWindowCharacters: 12_000,
      maxExtractionAttempts: 5,
      validationFailurePolicy: "empty-window",
      concurrency: options.concurrency,
      resourceConcurrency: 2,
      model: DEFAULT_RAG_EVAL_MANIFEST.models.answer.model,
      reasoningEffort: DEFAULT_RAG_EVAL_MANIFEST.models.answer.reasoningEffort,
    },
    adaptive: {
      entities: projections.reduce((sum, projection) => sum + projection.entities.length, 0),
      facts: projections.reduce((sum, projection) => sum + projection.facts.length, 0),
      hypotheses: projections.reduce((sum, projection) => sum + projection.hypothesisCount, 0),
      validationFailures: projections.reduce(
        (sum, projection) => sum + projection.validationFailureCount,
        0,
      ),
      processedWindows: projections.reduce(
        (sum, projection) => sum + projection.processedWindows,
        0,
      ),
      capabilities: Array.from(
        new Set(projections.flatMap((projection) => projection.capabilities)),
      ).sort(),
    },
    graph: {
      baseEntities: base.entities.length,
      baseFacts: base.edges.length,
      augmentedEntities: augmented.entities.length,
      augmentedFacts: augmented.edges.length,
    },
    localLlmUsage: usage,
    elapsedMs: Date.now() - startedAt,
    outputGraphPath: options.outputGraphPath,
  };
  writeJsonAtomic(resolve(options.cacheDirectory, "build-report.json"), report);
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
}

function groupChunks(chunks: readonly ChunkInput[]): Array<readonly [string, ChunkInput[]]> {
  const grouped = new Map<string, ChunkInput[]>();
  for (const chunk of chunks) {
    const match = /^(.*)-(\d+)$/.exec(chunk.id);
    const resourceId = match?.[1] ?? chunk.id;
    const values = grouped.get(resourceId) ?? [];
    values.push(chunk);
    grouped.set(resourceId, values);
  }
  return Array.from(grouped.entries()).map(([resourceId, values]) => [
    resourceId,
    values.sort((left, right) => chunkPosition(left.id) - chunkPosition(right.id)),
  ]);
}

async function mapWithConcurrency<T, R>(
  values: readonly T[],
  concurrency: number,
  operation: (value: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(values.length);
  let nextIndex = 0;
  const workers = Array.from({ length: Math.min(concurrency, values.length) }, async () => {
    while (true) {
      const index = nextIndex;
      nextIndex += 1;
      if (index >= values.length) return;
      const value = values[index];
      if (value === undefined) return;
      results[index] = await operation(value, index);
    }
  });
  await Promise.all(workers);
  return results;
}

function resourceSnapshot(resourceId: string, chunks: readonly ChunkInput[]): ResourceSnapshot {
  const contentHash = createHash("sha256");
  for (const chunk of chunks) contentHash.update(chunk.id).update("\0").update(chunk.body);
  return {
    organizationId: "rag-eval",
    source: { connectorId: "source-corpus", externalId: resourceId, type: "text/plain" },
    title: resourceId,
    contentHash: contentHash.digest("hex"),
    body: chunks.map((chunk) => chunk.body).join("\n"),
    acl: { organizationWide: true },
    chunks: chunks.map((chunk, index) => ({
      id: chunk.id,
      contentHash: createHash("sha256").update(chunk.body).digest("hex"),
      text: chunk.body,
      position: chunkPosition(chunk.id, index),
    })),
  };
}

function chunkPosition(id: string, fallback = 0): number {
  const match = /-(\d+)$/.exec(id);
  return match ? Number(match[1]) : fallback;
}

function augmentGraph(
  base: KGSerialized,
  projections: readonly ResourceProjection[],
): KGSerialized {
  const entities = new Map(base.entities.map((entity) => [entity.id, entity]));
  const chunkToEntities = new Map(
    base.chunkToEntities.map(([chunkId, entityIds]) => [chunkId, new Set(entityIds)]),
  );
  const edges = [...base.edges];

  for (const projection of projections) {
    const scopedId = (entityId: string) => `${projection.resourceId}:${entityId}`;
    const adaptiveById = new Map(projection.entities.map((entity) => [entity.entityId, entity]));
    for (const entity of projection.entities) {
      const id = scopedId(entity.entityId);
      entities.set(id, {
        id,
        surface: entity.name,
        chunkIds: [...entity.mentionChunkIds],
        freq: entity.mentionChunkIds.length,
      });
      for (const chunkId of entity.mentionChunkIds) appendEntity(chunkToEntities, chunkId, id);
    }
    for (const fact of projection.facts) {
      const subjectId = scopedId(fact.subject.entityId);
      if (!adaptiveById.has(fact.subject.entityId)) {
        throw new Error(`Adaptive Fact ${fact.factKey} has no subject Entity`);
      }
      const objectId =
        fact.object.kind === "entity"
          ? scopedEntityObject(fact, projection.resourceId, adaptiveById)
          : literalEntity(fact, entities);
      for (const chunkId of fact.evidenceChunkIds) {
        edges.push({ src: subjectId, predicate: fact.predicate, dst: objectId, chunkId });
        appendEntity(chunkToEntities, chunkId, subjectId);
        appendEntity(chunkToEntities, chunkId, objectId);
      }
    }
  }

  return {
    entities: Array.from(entities.values()),
    edges,
    chunkToEntities: Array.from(chunkToEntities.entries()).map(([chunkId, entityIds]) => [
      chunkId,
      Array.from(entityIds),
    ]),
  };
}

function scopedEntityObject(
  fact: ExtractedFact,
  resourceId: string,
  entities: ReadonlyMap<string, ExtractedEntity>,
): string {
  if (fact.object.kind !== "entity") throw new Error("Expected Entity Fact object");
  if (!entities.has(fact.object.entity.entityId)) {
    throw new Error(`Adaptive Fact ${fact.factKey} has no object Entity`);
  }
  return `${resourceId}:${fact.object.entity.entityId}`;
}

function literalEntity(fact: ExtractedFact, entities: Map<string, KGEntity>): string {
  if (fact.object.kind !== "literal") throw new Error("Expected literal Fact object");
  const id = `adaptive-literal:${createHash("sha256")
    .update(fact.factKey)
    .update("\0")
    .update(String(fact.object.value))
    .digest("hex")
    .slice(0, 24)}`;
  entities.set(id, {
    id,
    surface: String(fact.object.value),
    chunkIds: [...fact.evidenceChunkIds],
    freq: fact.evidenceChunkIds.length,
  });
  return id;
}

function appendEntity(map: Map<string, Set<string>>, chunkId: string, entityId: string): void {
  const values = map.get(chunkId) ?? new Set<string>();
  values.add(entityId);
  map.set(chunkId, values);
}

function summarizeUsage(cacheDirectory: string) {
  const path = resolve(cacheDirectory, "usage.jsonl");
  const records = existsSync(path) ? readJsonLines<CachedCompletion>(path) : [];
  return {
    calls: records.length,
    inputTokens: sumNullable(records.map((record) => record.inputTokens)),
    outputTokens: sumNullable(records.map((record) => record.outputTokens)),
    latencyMs: records.reduce((sum, record) => sum + record.latencyMs, 0),
    providerApiCostUsd: 0,
    execution: "local-codex-cli",
  };
}

function sumNullable(values: readonly (number | null)[]): number | null {
  return values.some((value) => value === null)
    ? null
    : values.reduce<number>((sum, value) => sum + (value ?? 0), 0);
}

function parseOptions(args: readonly string[]): BuildOptions {
  const values = new Map<string, string>();
  for (let index = 0; index < args.length; index += 2) {
    const name = args[index];
    const value = args[index + 1];
    if (!name?.startsWith("--") || !value) throw new Error(`Invalid argument ${name ?? ""}`);
    values.set(name, value);
  }
  return {
    chunksPath: requiredPath(values, "--chunks"),
    baseGraphPath: requiredPath(values, "--base-graph"),
    outputGraphPath: requiredPath(values, "--output-graph"),
    cacheDirectory: requiredPath(values, "--cache-dir"),
    concurrency: positiveInteger(values.get("--concurrency") ?? "20", "--concurrency"),
  };
}

function requiredPath(values: ReadonlyMap<string, string>, name: string): string {
  const value = values.get(name);
  if (!value) throw new Error(`${name} is required`);
  return resolve(value);
}

function positiveInteger(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${name} must be positive`);
  return parsed;
}

main().catch((error) => {
  process.stderr.write(`${(error as Error).stack ?? (error as Error).message}\n`);
  process.exitCode = 1;
});
