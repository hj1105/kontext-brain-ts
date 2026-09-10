import path from "node:path";
import {
  FileResourceContentStore,
  type KnowledgeSearchResult,
  LocalKnowledgeSearch,
  SqliteKnowledgeGraphRepository,
} from "@kontext-brain/core";
import {
  createTextEmbedder,
  loadLocalKnowledgePrincipal,
  readEmbeddingSettings,
} from "@kontext-brain/loader";
import { z } from "zod";

/**
 * Lets a worker or the host ask the local knowledge graph a question. The
 * answer is Evidence the sidecar already holds — documents and code the
 * ontology build wrote — so an agent cites a chunk id instead of re-reading
 * repositories through GitHub one file at a time.
 */
export const searchKnowledgeToolShape = {
  question: z.string().trim().min(1).max(2_000),
  limit: z.number().int().min(1).max(50).optional(),
  ontologyNodeIds: z.array(z.string().min(1)).max(32).optional(),
  connectorIds: z.array(z.string().min(1)).max(32).optional(),
};
const searchKnowledgeSchema = z.object(searchKnowledgeToolShape).strict();

export class LocalKnowledgeSearchOperations {
  private search: LocalKnowledgeSearch | undefined;

  constructor(private readonly dataDirectory: string) {}

  async searchKnowledge(input: unknown): Promise<KnowledgeSearchResult> {
    const request = searchKnowledgeSchema.parse(input);
    const principal = await loadLocalKnowledgePrincipal(this.dataDirectory);
    // Why one instance: it caches chunk text per resource content hash across questions,
    // and keeps the embedding model loaded.
    this.search ??= new LocalKnowledgeSearch(
      await SqliteKnowledgeGraphRepository.open(this.dataDirectory),
      new FileResourceContentStore(path.join(this.dataDirectory, "knowledge-content")),
      this.embedder(),
    );
    return this.search.search({ ...request, principal });
  }

  /** The space the last build embedded in; lexical when none was recorded or it cannot be built. */
  private embedder() {
    const settings = readEmbeddingSettings(this.dataDirectory);
    if (!settings) return null;
    try {
      return createTextEmbedder(settings, { dataDirectory: this.dataDirectory });
    } catch {
      return null;
    }
  }
}
