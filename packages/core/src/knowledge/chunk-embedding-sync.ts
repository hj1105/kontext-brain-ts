import type { OntologyBuildProgressSink } from "../ingest/ontology-build-progress.js";
import type { ResourceContentStore } from "./ports.js";
import type { SqliteKnowledgeGraphRepository } from "./sqlite-knowledge-graph.js";
import type { TextEmbedder } from "./text-embedder.js";

/**
 * Gives every active chunk a vector in the embedder's space, skipping the ones
 * that already have one. Runs after a build synced documents and code, and
 * again on its own when a person switches embedding provider: the graph stays
 * authoritative, and vectors are a derived index that can always be rebuilt.
 */

export interface ChunkEmbeddingOutcome {
  readonly model: string;
  readonly chunksEmbedded: number;
  readonly chunksTotal: number;
}

export async function embedMissingChunks(
  repository: SqliteKnowledgeGraphRepository,
  contentStore: ResourceContentStore,
  embedder: TextEmbedder,
  organizationId: string,
  options: { batchSize?: number; onProgress?: OntologyBuildProgressSink } = {},
): Promise<ChunkEmbeddingOutcome> {
  const batchSize = options.batchSize ?? 16;
  const existing = await repository.listChunkVectorHashes(organizationId, embedder.model);
  const pending: { chunkId: string; contentHash: string; text: string }[] = [];
  let total = 0;
  for (const resource of await repository.listResources(organizationId)) {
    if (resource.status !== "active") continue;
    const chunks = await repository.listChunks(organizationId, resource.resourceId);
    const active = chunks.filter((chunk) => chunk.status === "active");
    total += active.length;
    // Why the hash: a chunk keeps its id when its text changes; the old vector must not survive that.
    const missing = active.filter((chunk) => existing.get(chunk.chunkId) !== chunk.contentHash);
    if (missing.length === 0) continue;
    const content = await contentStore.get(resource.contentObjectKey);
    for (const chunk of missing) {
      const text = content?.chunks[chunk.sourceChunkId];
      if (text === undefined || text.trim() === "") continue;
      // Why the title: a chunk out of context ("see above") embeds better with its page name.
      pending.push({
        chunkId: chunk.chunkId,
        contentHash: chunk.contentHash,
        text: `${resource.title}\n${text}`,
      });
    }
  }
  await repository.deleteOrphanChunkVectors(organizationId);
  let embedded = 0;
  options.onProgress?.({ phase: "embed", done: 0, total: pending.length });
  for (let start = 0; start < pending.length; start += batchSize) {
    const batch = pending.slice(start, start + batchSize);
    const vectors = await embedder.embed(
      batch.map((entry) => entry.text),
      "passage",
    );
    await repository.saveChunkVectors(
      organizationId,
      embedder.model,
      batch.map((entry, index) => ({
        chunkId: entry.chunkId,
        contentHash: entry.contentHash,
        vector: vectors[index] as Float32Array,
      })),
    );
    embedded += batch.length;
    options.onProgress?.({ phase: "embed", done: embedded, total: pending.length });
  }
  return { model: embedder.model, chunksEmbedded: embedded, chunksTotal: total };
}
