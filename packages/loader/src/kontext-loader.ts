import { readFileSync } from "node:fs";
import {
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
import { OntologyEmbedder, OntologyGraphBuilder } from "./ontology-graph-builder.js";

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
}

/**
 * Assembles a KontextAgent from a YAML config file / string / object.
 */
export class KontextLoader {
  private readonly llmRegistry: LLMProviderRegistry;
  private readonly storeRegistry: OntologyStoreRegistry;
  private readonly mappingRegistry: NodeMappingRegistry;

  constructor(options: KontextLoaderOptions = {}) {
    this.llmRegistry = options.llmRegistry ?? new LLMProviderRegistry();
    this.storeRegistry = options.storeRegistry ?? new OntologyStoreRegistry();
    this.mappingRegistry = options.mappingRegistry ?? new NodeMappingRegistry();
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
    const persistedState = await ontologyStore.load(stateId);

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

    // Graph
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
    let graph: OntologyGraph;
    if (hasPersistedGraph) {
      graph = new OntologyGraph(
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
      await embedder.embed(graph.nodes.values());
    } else {
      graph = await new OntologyGraphBuilder(embedder).build(config.ontology, config.graph);
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
      graph,
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
      ontologyStore,
      stateId,
      resourceRecords: persistedState.resources ?? [],
    });
    await agent.initialize();
    return agent;
  }
}
