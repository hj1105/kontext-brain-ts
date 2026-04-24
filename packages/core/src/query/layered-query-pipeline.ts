import {
  DEFAULT_PIPELINE,
  type LayeredQueryResult,
  type PipelineStep,
} from "../graph/layered-models.js";
import { GraphTraverser } from "../graph/graph-traverser.js";
import type { OntologyGraph } from "../graph/ontology-graph.js";
import type { ContentFetcherRegistry, MetaDocumentSelector } from "./content-fetcher.js";
import { ScoreBasedSelector } from "./content-fetcher.js";
import { LayeredContextCollector } from "./layered-context-collector.js";
import type { MetaIndexStore } from "./meta-index-store.js";
import { KeywordMappingStrategy, type NodeMappingStrategy } from "./node-mapping-strategy.js";
import type { PromptTemplates } from "./prompt-templates.js";
import { DefaultPromptTemplates } from "./prompt-templates.js";
import { LLMRole, type RouterLLMAdapter } from "./llm-adapter.js";
import type { TokenEstimator } from "./token-estimator.js";
import { DefaultTokenEstimator } from "./token-estimator.js";
import type { VectorStore } from "./vector-store.js";

export interface LayeredQueryPipelineOptions {
  readonly mappingStrategy?: NodeMappingStrategy;
  readonly metaSelector?: MetaDocumentSelector;
  readonly maxTokens?: number;
  readonly vectorStore?: VectorStore | null;
  readonly pipeline?: readonly PipelineStep[];
  readonly templates?: PromptTemplates;
  readonly tokenEstimator?: TokenEstimator;
}

export class LayeredQueryPipeline {
  private readonly traverser: GraphTraverser;
  private readonly collector: LayeredContextCollector;
  private readonly mappingStrategy: NodeMappingStrategy;
  private readonly templates: PromptTemplates;
  public readonly activePipeline: readonly PipelineStep[];

  constructor(
    private readonly graph: OntologyGraph,
    private readonly router: RouterLLMAdapter,
    metaIndexStore: MetaIndexStore,
    fetcherRegistry: ContentFetcherRegistry,
    opts: LayeredQueryPipelineOptions = {},
  ) {
    this.traverser = new GraphTraverser(graph);
    this.mappingStrategy = opts.mappingStrategy ?? new KeywordMappingStrategy();
    this.templates = opts.templates ?? DefaultPromptTemplates;
    this.activePipeline = opts.pipeline ?? DEFAULT_PIPELINE;
    this.collector = new LayeredContextCollector(
      metaIndexStore,
      opts.metaSelector ?? new ScoreBasedSelector(),
      fetcherRegistry,
      opts.maxTokens ?? graph.config.maxTokens,
      this.activePipeline,
      opts.vectorStore ?? null,
      undefined,
      opts.tokenEstimator ?? DefaultTokenEstimator,
    );
  }

  async execute(userQuery: string): Promise<LayeredQueryResult> {
    const startNodes = await this.mappingStrategy.findStartNodes(userQuery, this.graph.nodes);
    const traversalResult = this.traverser.traverse(startNodes);
    const context = await this.collector.collect(traversalResult, userQuery);

    const answer = await this.router.complete(
      LLMRole.REASONING,
      this.templates.layeredReasoning,
      context.text,
      userQuery,
    );

    return {
      answer,
      usedOntologyNodes: context.usedOntologyNodes,
      selectedMetaDocs: context.selectedMetaDocs,
      fetchedContents: context.fetchedContents,
      contextTokensUsed: context.tokensUsed,
      traversalPath: traversalResult.path,
      pipelineSteps: this.activePipeline,
    };
  }
}
