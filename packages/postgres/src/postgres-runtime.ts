import {
  ActivateOntologyUseCase,
  BidirectionalNLayerRetriever,
  type OntologyReindexer,
  type ResourceContentStore,
  type ResourceSnapshotEnricher,
  SyncResourceUseCase,
} from "@kontext-brain/core";
import { MCPKnowledgeSynchronizer, type MCPResourceSnapshotAdapter } from "@kontext-brain/mcp";
import type { Pool } from "pg";
import { PostgresExtractionJobQueue } from "./postgres-extraction-job-queue.js";
import {
  PostgresAuthorizedKnowledgeGraphReader,
  PostgresKnowledgeGraphRepository,
} from "./postgres-knowledge-graph.js";
import { PostgresKnowledgeSearchGraph } from "./postgres-knowledge-search-graph.js";
import type { StoredOntologyGraph } from "./postgres-knowledge-search-graph.js";
import { PostgresOntologyDeploymentRepository } from "./postgres-ontology-deployments.js";
import { PostgresOntologyProposalQueue } from "./postgres-ontology-proposal-queue.js";

export function createPostgresKnowledgeRuntime(
  pool: Pool,
  contentStore: ResourceContentStore,
  mcpAdapters: readonly MCPResourceSnapshotAdapter[],
  ontologyReindexer: OntologyReindexer<StoredOntologyGraph>,
  snapshotEnricher?: ResourceSnapshotEnricher,
) {
  const repository = new PostgresKnowledgeGraphRepository(pool);
  const resourceSync = new SyncResourceUseCase(repository, contentStore);
  const searchGraph = new PostgresKnowledgeSearchGraph(pool, contentStore);
  const ontologyProposalQueue = new PostgresOntologyProposalQueue(pool);
  const ontologyDeployments = new PostgresOntologyDeploymentRepository<StoredOntologyGraph>(pool);
  return {
    repository,
    authorizedReader: new PostgresAuthorizedKnowledgeGraphReader(pool),
    resourceSync,
    searchGraph,
    knowledgeRetriever: new BidirectionalNLayerRetriever(searchGraph),
    mcpKnowledgeSynchronizer: new MCPKnowledgeSynchronizer(
      resourceSync,
      mcpAdapters,
      snapshotEnricher,
    ),
    ontologyProposalQueue,
    ontologyDeployments,
    ontologyActivation: {
      async activate(input: {
        organizationId: string;
        yaml: string;
        graph: StoredOntologyGraph;
      }): Promise<void> {
        await new ActivateOntologyUseCase(
          ontologyDeployments,
          {
            async compile() {
              return input.graph;
            },
          },
          {
            async validate(candidate) {
              const nodes = Array.isArray(candidate.graph.nodes)
                ? candidate.graph.nodes
                : Object.values(candidate.graph.nodes);
              const ids = new Set(nodes.map((node) => node.id));
              for (const edge of candidate.graph.edges) {
                if (!ids.has(edge.from) || !ids.has(edge.to)) {
                  throw new Error(
                    `Ontology edge ${edge.from} -> ${edge.to} references an unknown node`,
                  );
                }
              }
            },
          },
          ontologyReindexer,
        ).execute({ organizationId: input.organizationId, yaml: input.yaml });
      },
    },
    extractionJobs: new PostgresExtractionJobQueue(pool),
  };
}
