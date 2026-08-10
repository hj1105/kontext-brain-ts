import {
  DEFAULT_PIPELINE,
  DepthType,
  type DocumentContent,
  type MetaDocument,
  type PipelineStep,
  type QueryPipelineTrace,
  type TraversalResult,
} from "../graph/layered-models.js";
import type { OntologyNode } from "../graph/ontology-node.js";
import type { ContentFetcherRegistry, MetaDocumentSelector } from "./content-fetcher.js";
import type { MetaIndexStore } from "./meta-index-store.js";
import { type Candidate, type CandidateKind, NLayerRunner, makeLayer } from "./n-layer.js";
import { type StepContext, StepExecutorRegistry } from "./step-executor.js";
import { DefaultTokenEstimator, type TokenEstimator } from "./token-estimator.js";
import type { VectorStore } from "./vector-store.js";

export interface LayeredContext {
  readonly text: string;
  readonly usedOntologyNodes: readonly OntologyNode[];
  readonly selectedMetaDocs: readonly MetaDocument[];
  readonly fetchedContents: readonly DocumentContent[];
  readonly tokensUsed: number;
  readonly traces: readonly QueryPipelineTrace[];
}

/**
 * Runs one N-layer retrieval pipeline for every traversed ontology node.
 *
 * Graph depth and retrieval stage are deliberately independent: a node
 * selected at traversal depth 0 still executes META → CONTENT. `depth` on a
 * PipelineStep is therefore ordering metadata, not a graph-depth dispatch
 * key. This makes flat and hierarchical ontologies behave consistently.
 */
export class LayeredContextCollector {
  private readonly orderedPipeline: readonly PipelineStep[];

  constructor(
    private readonly metaIndexStore: MetaIndexStore,
    private readonly metaSelector: MetaDocumentSelector,
    private readonly fetcherRegistry: ContentFetcherRegistry,
    private readonly maxTokens = 8000,
    pipeline: readonly PipelineStep[] = DEFAULT_PIPELINE,
    private readonly vectorStore: VectorStore | null = null,
    private readonly executorRegistry = new StepExecutorRegistry(),
    private readonly tokenEstimator: TokenEstimator = DefaultTokenEstimator,
  ) {
    this.orderedPipeline = pipeline
      .map((pipelineStep, index) => ({ pipelineStep, index }))
      .sort((a, b) => a.pipelineStep.depth - b.pipelineStep.depth || a.index - b.index)
      .map(({ pipelineStep }) => pipelineStep);
  }

  async collect(traversal: TraversalResult, query: string): Promise<LayeredContext> {
    const usedNodes: OntologyNode[] = [];
    const selectedDocuments = new Map<string, MetaDocument>();
    const fetchedContents = new Map<string, DocumentContent>();
    const parts: string[] = [];
    const traces: QueryPipelineTrace[] = [];
    let remaining = this.maxTokens;

    for (const traversed of traversal.nodes) {
      if (remaining <= 0) break;

      const node = traversed.node;
      const nodeDocuments = new Map<string, MetaDocument>();
      let exhausted = false;

      const layers = this.orderedPipeline.map((pipelineStep) =>
        makeLayer(
          `${node.id}:${pipelineStep.depth}:${pipelineStep.type}`,
          "any",
          outputKindFor(pipelineStep.type),
          async () => {
            if (exhausted) return [];

            const executor = this.executorRegistry.resolve(pipelineStep.type);
            const context: StepContext = {
              node,
              query,
              accumulatedDocs: Array.from(nodeDocuments.values()),
              metaIndexStore: this.metaIndexStore,
              metaSelector: this.metaSelector,
              fetcherRegistry: this.fetcherRegistry,
              vectorStore: this.vectorStore,
            };
            const result = await executor.execute(context, pipelineStep);

            for (const document of result.selectedDocs) {
              const key = documentKey(document);
              nodeDocuments.set(key, document);
              selectedDocuments.set(key, document);
            }
            for (const content of result.fetchedContents) {
              fetchedContents.set(contentKey(content), content);
            }

            if (result.contextSection.trim().length > 0) {
              const tokens = this.tokenEstimator.estimate(result.contextSection);
              if (remaining - tokens >= 0) {
                parts.push(result.contextSection);
                remaining -= tokens;
              } else {
                const approximateCharacterBudget = Math.max(0, remaining * 4);
                parts.push(
                  `${result.contextSection.slice(0, approximateCharacterBudget)}\n... [truncated]`,
                );
                remaining = 0;
                exhausted = true;
              }
            }

            return candidatesFor(
              pipelineStep.type,
              node,
              traversed.cumulativeWeight,
              result.selectedDocs,
              result.fetchedContents,
            );
          },
        ),
      );

      if (layers.length === 0) continue;
      usedNodes.push(node);
      const run = await new NLayerRunner({ layers }).run(query);
      traces.push(...run.traces);
    }

    return {
      text: parts.join("\n\n---\n\n"),
      usedOntologyNodes: usedNodes,
      selectedMetaDocs: Array.from(selectedDocuments.values()),
      fetchedContents: Array.from(fetchedContents.values()),
      tokensUsed: this.maxTokens - remaining,
      traces,
    };
  }
}

function outputKindFor(type: DepthType): CandidateKind {
  switch (type) {
    case DepthType.ONTOLOGY:
      return "node";
    case DepthType.META:
    case DepthType.VECTOR:
      return "doc";
    case DepthType.CONTENT:
    case DepthType.SECTION:
    case DepthType.CHUNK:
      return "chunk";
  }
}

function candidatesFor(
  type: DepthType,
  node: OntologyNode,
  nodeScore: number,
  documents: readonly MetaDocument[],
  contents: readonly DocumentContent[],
): readonly Candidate[] {
  if (type === DepthType.CONTENT || type === DepthType.SECTION || type === DepthType.CHUNK) {
    return contents.map((content, index) => ({
      kind: "chunk",
      docId: content.metaDocumentId,
      chunkId: `${content.metaDocumentId}:${index}`,
      text: content.sectionContent ?? content.body,
      score: 1 / (index + 1),
    }));
  }

  if (type === DepthType.META || type === DepthType.VECTOR) {
    return documents.map((document) => ({
      kind: "doc",
      docId: document.id,
      meta: document,
      score: document.score,
    }));
  }

  return [
    {
      kind: "node",
      nodeId: node.id,
      score: nodeScore,
    },
  ];
}

function documentKey(document: MetaDocument): string {
  return `${document.source}:${document.ontologyNodeId}:${document.id}`;
}

function contentKey(content: DocumentContent): string {
  return `${content.metadata?.connector ?? ""}:${content.source}:${content.metaDocumentId}:${content.title}`;
}
