import {
  DEFAULT_PIPELINE,
  DepthType,
  type DocumentContent,
  type MetaDocument,
  type PipelineStep,
  type TraversalResult,
  step,
} from "../graph/layered-models.js";
import type { OntologyNode } from "../graph/ontology-node.js";
import type { ContentFetcherRegistry, MetaDocumentSelector } from "./content-fetcher.js";
import type { MetaIndexStore } from "./meta-index-store.js";
import {
  type StepContext,
  StepExecutorRegistry,
} from "./step-executor.js";
import {
  DefaultTokenEstimator,
  type TokenEstimator,
} from "./token-estimator.js";
import type { VectorStore } from "./vector-store.js";

export interface LayeredContext {
  readonly text: string;
  readonly usedOntologyNodes: readonly OntologyNode[];
  readonly selectedMetaDocs: readonly MetaDocument[];
  readonly fetchedContents: readonly DocumentContent[];
  readonly tokensUsed: number;
}

/**
 * Assembles context across N depths using step executors from the registry.
 * New types can be added by registering a new StepExecutor (OCP).
 */
export class LayeredContextCollector {
  private readonly stepsByDepth: ReadonlyMap<number, readonly PipelineStep[]>;

  constructor(
    private readonly metaIndexStore: MetaIndexStore,
    private readonly metaSelector: MetaDocumentSelector,
    private readonly fetcherRegistry: ContentFetcherRegistry,
    private readonly maxTokens = 8000,
    private readonly pipeline: readonly PipelineStep[] = DEFAULT_PIPELINE,
    private readonly vectorStore: VectorStore | null = null,
    private readonly executorRegistry = new StepExecutorRegistry(),
    private readonly tokenEstimator: TokenEstimator = DefaultTokenEstimator,
  ) {
    const grouped = new Map<number, PipelineStep[]>();
    for (const s of pipeline) {
      const list = grouped.get(s.depth) ?? [];
      list.push(s);
      grouped.set(s.depth, list);
    }
    this.stepsByDepth = grouped;
  }

  async collect(traversal: TraversalResult, query: string): Promise<LayeredContext> {
    const usedNodes: OntologyNode[] = [];
    const allSelected: MetaDocument[] = [];
    const allContents: DocumentContent[] = [];
    const parts: string[] = [];
    let remaining = this.maxTokens;

    // Track maximum depth explicitly configured in the pipeline; anything past
    // this gets inferred via inferStep().
    const maxPipelineDepth = Math.max(...this.pipeline.map((s) => s.depth), -1);

    for (const traversed of traversal.nodes) {
      if (remaining <= 0) break;
      const node = traversed.node;
      const stepsForDepth =
        this.stepsByDepth.get(traversed.depth) ??
        (traversed.depth <= maxPipelineDepth
          ? []
          : [this.inferStep(traversed.depth)]);

      if (stepsForDepth.length === 0) continue;

      usedNodes.push(node);

      // Run every step configured at this depth in order — this fixes the bug
      // where a leaf-node mapping at depth 0 only ran the ONTOLOGY step and
      // never retrieved any documents. Configure PERNODE_PIPELINE to chain
      // META + CONTENT at the same depth.
      for (const pipelineStep of stepsForDepth) {
        const executor = this.executorRegistry.resolve(pipelineStep.type);
        const ctx: StepContext = {
          node,
          query,
          accumulatedDocs: allSelected,
          metaIndexStore: this.metaIndexStore,
          metaSelector: this.metaSelector,
          fetcherRegistry: this.fetcherRegistry,
          vectorStore: this.vectorStore,
        };
        const result = await executor.execute(ctx, pipelineStep);

        allSelected.push(...result.selectedDocs);
        allContents.push(...result.fetchedContents);

        if (result.contextSection.trim().length > 0) {
          const tokens = this.tokenEstimator.estimate(result.contextSection);
          if (remaining - tokens >= 0) {
            parts.push(result.contextSection);
            remaining -= tokens;
          } else {
            const budget = Math.max(0, remaining * 4);
            parts.push(`${result.contextSection.slice(0, budget)}\n... [truncated]`);
            remaining = 0;
            break;
          }
        }
      }
    }

    return {
      text: parts.join("\n\n---\n\n"),
      usedOntologyNodes: usedNodes,
      selectedMetaDocs: allSelected,
      fetchedContents: allContents,
      tokensUsed: this.maxTokens - remaining,
    };
  }

  private inferStep(depth: number): PipelineStep {
    const lastOntologyDepth = this.pipeline
      .filter((s) => s.type === DepthType.ONTOLOGY)
      .map((s) => s.depth)
      .reduce((max, d) => Math.max(max, d), -1);
    if (depth <= lastOntologyDepth) return step({ depth, type: DepthType.ONTOLOGY });
    if (depth === lastOntologyDepth + 1) return step({ depth, type: DepthType.META });
    return step({ depth, type: DepthType.CONTENT });
  }
}
