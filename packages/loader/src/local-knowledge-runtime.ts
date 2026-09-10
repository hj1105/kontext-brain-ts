import path from "node:path";
import type { CodeResourceSyncPort } from "@kontext-brain/code";
import {
  FileResourceContentStore,
  InMemoryOntologyProposalQueue,
  type KnowledgeGraphRepository,
  type OntologyProposalQueue,
  SqliteKnowledgeGraphRepository,
  SyncResourceUseCase,
} from "@kontext-brain/core";
import { GenericMCPResourceSnapshotAdapter, MCPKnowledgeSynchronizer } from "@kontext-brain/mcp";
import type { MCPConfigDto } from "./kontext-config.js";
import { loadLocalKnowledgePrincipal } from "./local-knowledge-principal.js";

/**
 * The knowledge runtime a local ontology build writes into: the same SQLite
 * graph and content store the Task sidecar reads from, under the same data
 * directory and organization. Without it a build kept only the node schema and
 * threw away what every Notion page, Markdown file and code module said, so the
 * ontology was a table of contents with no book behind it.
 */

export interface LocalKnowledgeRuntime {
  readonly organizationId: string;
  readonly dataDirectory: string;
  readonly repository: KnowledgeGraphRepository;
  readonly mcpKnowledgeSynchronizer: MCPKnowledgeSynchronizer;
  /** Symbol-level code projection writes through the same use case as documents. */
  readonly codeResourceSync: CodeResourceSyncPort;
  readonly ontologyProposalQueue: OntologyProposalQueue;
}

/** Where the sidecar keeps its state; the host passes it, a shell may set it. */
export function resolveKontextDataDirectory(
  override: string | undefined,
  env: NodeJS.ProcessEnv = process.env,
): string | undefined {
  const value = override?.trim() || env.KONTEXT_PLUGIN_DATA?.trim();
  return value ? path.resolve(value) : undefined;
}

export async function createLocalKnowledgeRuntime(
  dataDirectory: string,
  sources: readonly MCPConfigDto[],
): Promise<LocalKnowledgeRuntime> {
  const principal = await loadLocalKnowledgePrincipal(dataDirectory);
  const repository = await SqliteKnowledgeGraphRepository.open(dataDirectory);
  const contentStore = new FileResourceContentStore(path.join(dataDirectory, "knowledge-content"));
  const resourceSync = new SyncResourceUseCase(repository, contentStore);
  // Why organization-wide: the local graph belongs to one person on one machine;
  // the Task sidecar's own registrations use the same principal and policy.
  const adapters = sources.map(
    (source) =>
      new GenericMCPResourceSnapshotAdapter(source.name, source.type ?? source.transport ?? "mcp", {
        organizationWide: true,
      }),
  );
  return {
    organizationId: principal.organizationId,
    dataDirectory,
    repository,
    mcpKnowledgeSynchronizer: new MCPKnowledgeSynchronizer(resourceSync, adapters),
    codeResourceSync: resourceSync,
    ontologyProposalQueue: new InMemoryOntologyProposalQueue(),
  };
}
