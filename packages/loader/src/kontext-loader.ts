import { readFileSync } from "node:fs";
import {
  ContentFetcherRegistry,
  DefaultPromptTemplates,
  DefaultTokenEstimator,
  DepthType,
  IngestPipeline,
  KoreanTokenEstimator,
  LLMMetaDocumentSelector,
  type MetaDocumentSelector,
  NodeMappingRegistry,
  type PipelineStep,
  type PromptTemplates,
  type RouterLLMAdapter as _Router,
  RouterLLMAdapter,
  ScoreBasedSelector,
  type TokenEstimator,
  VectorMappingStrategy,
  VectorMetaIndexStore,
} from "@kontext-brain/core";
import {
  LangChainLLMAdapter,
  LangChainVectorStore,
  LLMProviderRegistry,
  type LLMProviderConfig,
} from "@kontext-brain/llm";
import {
  MCPContentFetcherBridge,
  MCPLayerAdapterFactory,
  SseMCPConnector,
  StdioMCPConnector,
  type MCPConnector,
  type MCPLayerAdapter,
} from "@kontext-brain/mcp";
import { parse as parseYaml } from "yaml";
import {
  InMemoryOntologyStore,
  OntologyStoreRegistry,
} from "@kontext-brain/core";
import { KontextAgent } from "./kontext-agent.js";
import {
  KontextConfigSchema,
  type KontextConfig,
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
  const type =
    typeStr in DepthType
      ? (DepthType as Record<string, DepthType>)[typeStr]!
      : DepthType.CONTENT;
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
    const _tokenEstimator = resolveTokenEstimator(config.language);

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
    const _ontologyStore = this.storeRegistry.create(config.storage);

    // Meta index + fetchers
    const metaIndexStore = vectorStore
      ? new VectorMetaIndexStore(vectorStore)
      : new (await import("@kontext-brain/core")).InMemoryMetaIndexStore();
    const fetcherRegistry = new ContentFetcherRegistry();

    // MCP connectors + layer adapters
    const mcpConnectors: MCPConnector[] = config.mcp.map(createConnector);
    const mcpLayerAdapters: MCPLayerAdapter[] = config.mcp.map((dto, i) => {
      const connector = mcpConnectors[i]!;
      const adapter = createLayerAdapter(dto, connector);
      fetcherRegistry.register(new MCPContentFetcherBridge(adapter));
      return adapter;
    });

    // Mapping strategy
    const mappingStrategy = vectorStore
      ? new VectorMappingStrategy(vectorStore)
      : (await import("@kontext-brain/core")).KeywordMappingStrategy.prototype.constructor
        ? new (await import("@kontext-brain/core")).KeywordMappingStrategy()
        : new (await import("@kontext-brain/core")).KeywordMappingStrategy();

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
    const graph = await new OntologyGraphBuilder(embedder).build(
      config.ontology,
      config.graph,
    );

    // Ingest pipeline
    const ingestPipeline = new IngestPipeline(
      traversalAdapter,
      new InMemoryOntologyStore(),
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

    return new KontextAgent({
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
    });
  }
}
