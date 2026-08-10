import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import {
  type BidirectionalNLayerRetriever,
  ContentFetcherRegistry,
  DefaultPromptTemplates,
  DefaultTokenEstimator,
  DepthType,
  InMemoryMetaIndexStore,
  IngestPipeline,
  KoreanTokenEstimator,
  LLMMetaDocumentSelector,
  type MetaDocumentSelector,
  NodeMappingRegistry,
  OntologyGraph,
  type OntologyProposalQueue,
  OntologyStoreRegistry,
  type PipelineStep,
  type PromptTemplates,
  RouterLLMAdapter,
  ScoreBasedSelector,
  type TokenEstimator,
  TraversalStrategy,
  VectorMappingStrategy,
  VectorMetaIndexStore,
  type RouterLLMAdapter as _Router,
  createOrchestrationSnapshot,
  deserializeMetaDocument,
  toEdges,
  toOntologyNodes,
} from "@kontext-brain/core";
import {
  type LLMProviderConfig,
  LLMProviderRegistry,
  LangChainLLMAdapter,
  LangChainVectorStore,
} from "@kontext-brain/llm";
import {
  type MCPConnector,
  MCPContentFetcherBridge,
  type MCPKnowledgeSynchronizer,
  type MCPLayerAdapter,
  MCPLayerAdapterFactory,
  SseMCPConnector,
  StdioMCPConnector,
} from "@kontext-brain/mcp";
import { parse as parseYaml } from "yaml";
import { KontextAgent } from "./kontext-agent.js";
import {
  type KontextConfig,
  KontextConfigSchema,
  type LLMProviderConfigDto,
  type MCPConfigDto,
} from "./kontext-config.js";
import type { OntologyNodeConfig } from "./kontext-config.js";
import {
  OntologyEmbedder,
  OntologyGraphBuilder,
  validateOntologyConfiguration,
} from "./ontology-graph-builder.js";

export function computeOntologyContentHash(ontology: readonly OntologyNodeConfig[]): string {
  return createHash("sha256").update(JSON.stringify(ontology)).digest("hex");
}

export interface KnowledgeOntologyActivationPort {
  activate(input: {
    readonly organizationId: string;
    readonly yaml: string;
    readonly graph: {
      readonly nodes: ReadonlyArray<{
        readonly id: string;
        readonly description: string;
        readonly parentId?: string | null;
      }>;
      readonly edges: ReadonlyArray<{
        readonly from: string;
        readonly to: string;
        readonly weight: number;
        readonly type?: string;
      }>;
    };
  }): Promise<void>;
}

function resolvePromptTemplates(_language: string): PromptTemplates {
  return DefaultPromptTemplates;
}

function resolveTokenEstimator(language: string): TokenEstimator {
  return language === "ko" ? KoreanTokenEstimator : DefaultTokenEstimator;
}

function toLLMConfig(dto: LLMProviderConfigDto): LLMProviderConfig {
  return {
    provider: dto.provider,
    model: dto.model,
    apiKey: dto.apiKey,
    baseUrl: dto.baseUrl,
  };
}

function toPipelineStep(dto: NonNullable<KontextConfig["pipeline"]>[number]): PipelineStep {
  const typeStr = dto.type.toUpperCase();
  const type = (DepthType as Record<string, DepthType | undefined>)[typeStr] ?? DepthType.CONTENT;
  return {
    depth: dto.depth,
    type,
    maxSelect: dto.maxSelect,
    sectionKey: dto.sectionKey ?? null,
    fetchFull: dto.fetchFull,
    threshold: dto.threshold,
  };
}

function createConnector(dto: MCPConfigDto): MCPConnector {
  const transport = dto.transport ?? (dto.command ? "stdio" : "sse");
  if (transport === "stdio") {
    if (!dto.command) throw new Error(`MCP '${dto.name}': stdio transport requires 'command'`);
    return new StdioMCPConnector(dto.name, dto.command, dto.args ?? []);
  }
  if (!dto.url) throw new Error(`MCP '${dto.name}': sse transport requires 'url'`);
  return new SseMCPConnector(dto.name, dto.url);
}

function createLayerAdapter(dto: MCPConfigDto, connector: MCPConnector): MCPLayerAdapter {
  switch ((dto.type ?? "").toLowerCase()) {
    case "notion":
      return MCPLayerAdapterFactory.notion(connector);
    case "jira":
      return MCPLayerAdapterFactory.jira(connector);
    case "github_pr":
    case "github-pr":
      return MCPLayerAdapterFactory.githubPr(connector);
    case "slack":
      return MCPLayerAdapterFactory.slack(connector);
    default:
      return MCPLayerAdapterFactory.notion(connector);
  }
}

export interface KontextLoaderOptions {
  llmRegistry?: LLMProviderRegistry;
  storeRegistry?: OntologyStoreRegistry;
  mappingRegistry?: NodeMappingRegistry;
  knowledgeRuntime?: {
    readonly organizationId: string;
    readonly knowledgeRetriever: BidirectionalNLayerRetriever;
    readonly mcpKnowledgeSynchronizer: MCPKnowledgeSynchronizer;
    readonly ontologyProposalQueue: OntologyProposalQueue;
    readonly ontologyActivation?: KnowledgeOntologyActivationPort;
  };
}

/**
 * Assembles a KontextAgent from a YAML config file / string / object.
 */
export class KontextLoader {
  private readonly llmRegistry: LLMProviderRegistry;
  private readonly storeRegistry: OntologyStoreRegistry;
  private readonly mappingRegistry: NodeMappingRegistry;
  private readonly knowledgeRuntime?: KontextLoaderOptions["knowledgeRuntime"];

  constructor(options: KontextLoaderOptions = {}) {
    this.llmRegistry = options.llmRegistry ?? new LLMProviderRegistry();
    this.storeRegistry = options.storeRegistry ?? new OntologyStoreRegistry();
    this.mappingRegistry = options.mappingRegistry ?? new NodeMappingRegistry();
    this.knowledgeRuntime = options.knowledgeRuntime;
  }

  static async fromFile(path: string, options: KontextLoaderOptions = {}): Promise<KontextAgent> {
    return new KontextLoader(options).fromFile(path);
  }

  static async fromYaml(yaml: string, options: KontextLoaderOptions = {}): Promise<KontextAgent> {
    return new KontextLoader(options).fromYaml(yaml);
  }

  async fromFile(path: string): Promise<KontextAgent> {
    const text = readFileSync(path, "utf-8");
    return this.fromYaml(text);
  }

  async fromYaml(yaml: string): Promise<KontextAgent> {
    const raw = parseYaml(yaml);
    const config = KontextConfigSchema.parse(raw);
    return this.from(config);
  }

  async from(config: KontextConfig): Promise<KontextAgent> {
    const templates = resolvePromptTemplates(config.language);
    const tokenEstimator = resolveTokenEstimator(config.language);

    // LLM
    const traversalModel = this.llmRegistry.createChat(toLLMConfig(config.llm.traversal));
    const reasoningModel = this.llmRegistry.createChat(toLLMConfig(config.llm.reasoning));
    const traversalAdapter = new LangChainLLMAdapter(traversalModel, templates);
    const reasoningAdapter = new LangChainLLMAdapter(reasoningModel, templates);
    const router = new RouterLLMAdapter(traversalAdapter, reasoningAdapter);

    // Vector + embedding
    let vectorStore: LangChainVectorStore | null = null;
    try {
      const embeddingModel = this.llmRegistry.createEmbedding(toLLMConfig(config.llm.traversal));
      vectorStore = new LangChainVectorStore(embeddingModel);
    } catch {
      // Embedding not available for this provider — fall back to no vector store
      vectorStore = null;
    }

    // Store
    const stateId = "default";
    const ontologyStore = this.storeRegistry.create(config.storage);
    let persistedState = await ontologyStore.load(stateId);

    // Meta index + fetchers
    const metaIndexStore = vectorStore
      ? new VectorMetaIndexStore(vectorStore)
      : new InMemoryMetaIndexStore();
    const fetcherRegistry = new ContentFetcherRegistry();

    for (const [nodeId, documents] of Object.entries(persistedState.metaDocuments ?? {})) {
      await metaIndexStore.index(nodeId, documents.map(deserializeMetaDocument));
    }

    // MCP connectors + layer adapters
    const mcpConnectors: MCPConnector[] = config.mcp.map(createConnector);
    const mcpLayerAdapters: MCPLayerAdapter[] = config.mcp.map((dto, i) => {
      const connector = mcpConnectors[i];
      if (!connector) {
        throw new Error(`MCP connector at index ${i} was not created`);
      }
      const adapter = createLayerAdapter(dto, connector);
      fetcherRegistry.register(new MCPContentFetcherBridge(adapter));
      return adapter;
    });

    // Mapping strategy
    const mappingStrategy = vectorStore
      ? new VectorMappingStrategy(vectorStore)
      : this.mappingRegistry.resolve("keyword");

    // Meta selector
    const metaSelector: MetaDocumentSelector =
      config.llm.traversal.provider !== "none"
        ? new LLMMetaDocumentSelector(traversalAdapter, templates)
        : new ScoreBasedSelector();

    // Small YAML-derived ontology schema cache. Instance KG data is owned by
    // the production knowledge runtime and is never loaded here.
    const embedder = vectorStore
      ? new OntologyEmbedder(vectorStore)
      : new OntologyEmbedder({
          async embed() {
            return new Float32Array(0);
          },
          async upsert() {},
          async similaritySearch() {
            return [];
          },
          async similaritySearchWithPrefix() {
            return [];
          },
        });
    const hasPersistedGraph = Object.keys(persistedState.nodes).length > 0;
    const configuredOntologyHash =
      config.ontology.length > 0 ? computeOntologyContentHash(config.ontology) : undefined;
    const yamlChanged =
      configuredOntologyHash !== undefined &&
      persistedState.ontologyContentHash !== configuredOntologyHash;
    let ontologySchemaGraph: OntologyGraph;
    let candidatePersistedState: typeof persistedState | undefined;
    if (hasPersistedGraph && !yamlChanged) {
      ontologySchemaGraph = new OntologyGraph(
        toOntologyNodes(persistedState),
        toEdges(persistedState),
        persistedState.graphConfig ?? {
          maxDepth: config.graph.maxDepth,
          maxTokens: config.graph.maxTokens,
          strategy:
            config.graph.strategy.toUpperCase() === "BFS"
              ? TraversalStrategy.BFS
              : config.graph.strategy.toUpperCase() === "DFS"
                ? TraversalStrategy.DFS
                : TraversalStrategy.WEIGHTED_DFS,
        },
      );
      await embedder.embed(ontologySchemaGraph.nodes.values());
    } else {
      validateOntologyConfiguration(config.ontology);
      ontologySchemaGraph = await new OntologyGraphBuilder(embedder).build(
        config.ontology,
        config.graph,
      );
      if (configuredOntologyHash) {
        const candidateMeta = new Map(
          Object.entries(persistedState.metaDocuments ?? {})
            .filter(([nodeId]) => ontologySchemaGraph.nodes.has(nodeId))
            .map(([nodeId, documents]) => [nodeId, documents.map(deserializeMetaDocument)]),
        );
        const candidateResources = (persistedState.resources ?? []).map((resource) => ({
          ...resource,
          nodeIds: resource.nodeIds.filter((nodeId) => ontologySchemaGraph.nodes.has(nodeId)),
        }));
        candidatePersistedState = createOrchestrationSnapshot(
          stateId,
          ontologySchemaGraph,
          candidateMeta,
          candidateResources,
          configuredOntologyHash,
        );
      }
    }

    if (configuredOntologyHash && this.knowledgeRuntime?.ontologyActivation) {
      await this.knowledgeRuntime.ontologyActivation.activate({
        organizationId: this.knowledgeRuntime.organizationId,
        yaml: JSON.stringify(config.ontology),
        graph: {
          nodes: Array.from(ontologySchemaGraph.nodes.values()).map((node) => ({
            id: node.id,
            description: node.description,
            parentId: node.parentId,
          })),
          edges: ontologySchemaGraph.edges.map((edge) => ({
            from: edge.from,
            to: edge.to,
            weight: edge.weight,
            type: edge.type,
          })),
        },
      });
    }
    if (candidatePersistedState) {
      await ontologyStore.save(stateId, candidatePersistedState);
      persistedState = candidatePersistedState;
    }

    // Ingest pipeline
    const ingestPipeline = new IngestPipeline(
      traversalAdapter,
      ontologyStore,
      vectorStore ?? {
        async embed() {
          return new Float32Array(0);
        },
        async upsert() {},
        async similaritySearch() {
          return [];
        },
        async similaritySearchWithPrefix() {
          return [];
        },
      },
      templates,
    );

    // Pipeline config
    const pipeline = config.pipeline?.map(toPipelineStep);

    const agent = new KontextAgent({
      ontologySchemaGraph,
      router,
      mcpConnectors,
      mcpLayerAdapters,
      metaIndexStore,
      fetcherRegistry,
      vectorStore,
      mappingStrategy,
      metaSelector,
      ingestPipeline,
      pipeline,
      templates,
      tokenEstimator,
      legacySnapshotStore: ontologyStore,
      stateId,
      mcpResourceCacheEntries: persistedState.resources ?? [],
      ontologyContentHash: configuredOntologyHash ?? persistedState.ontologyContentHash,
      organizationId: this.knowledgeRuntime?.organizationId ?? stateId,
      knowledgeRetriever: this.knowledgeRuntime?.knowledgeRetriever,
      mcpKnowledgeSynchronizer: this.knowledgeRuntime?.mcpKnowledgeSynchronizer,
      ontologyProposalQueue: this.knowledgeRuntime?.ontologyProposalQueue,
    });
    await agent.initialize();
    return agent;
  }
}
