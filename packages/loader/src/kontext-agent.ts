import {
  type ContentFetcherRegistry,
  type DataSource,
  DocumentClassifier,
  DEFAULT_PIPELINE,
  DepthType,
  type Edge,
  type IngestPipeline,
  type LayeredQueryResult,
  LayeredQueryPipeline,
  type MCPResourceInfo,
  type MetaDocument,
  type MetaDocumentSelector,
  type MetaIndexStore,
  type NodeMappingStrategy,
  OntologyAutoBuilder,
  OntologyGraph,
  type OntologyNode,
  type PipelineStep,
  type PromptTemplates,
  DefaultPromptTemplates,
  type RouterLLMAdapter,
  type VectorStore,
  createMetaDocument,
} from "@kontext-brain/core";
import {
  MCPDocumentSource,
  type MCPConnector,
  type MCPLayerAdapter,
} from "@kontext-brain/mcp";

export interface AutoSetupResult {
  readonly nodesCreated: number;
  readonly nodesReused: number;
  readonly documentsClassified: number;
  readonly documentsUnmapped: number;
  readonly ontologyYaml: string;
}

export interface KontextAgentDeps {
  graph: OntologyGraph;
  router: RouterLLMAdapter;
  mcpConnectors: readonly MCPConnector[];
  mcpLayerAdapters: readonly MCPLayerAdapter[];
  metaIndexStore: MetaIndexStore;
  fetcherRegistry: ContentFetcherRegistry;
  vectorStore: VectorStore | null;
  mappingStrategy: NodeMappingStrategy;
  metaSelector: MetaDocumentSelector;
  ingestPipeline: IngestPipeline;
  pipeline?: readonly PipelineStep[];
  templates?: PromptTemplates;
}

/**
 * KontextAgent — N-depth variable pipeline ontology AI agent.
 *
 * Use `autoSetup()` after connecting MCP sources to auto-build or expand the
 * ontology via LLM classification.
 */
export class KontextAgent {
  private graph: OntologyGraph;
  private readonly router: RouterLLMAdapter;
  private readonly mcpConnectors: readonly MCPConnector[];
  private readonly mcpLayerAdapters: readonly MCPLayerAdapter[];
  private readonly metaIndexStore: MetaIndexStore;
  private readonly fetcherRegistry: ContentFetcherRegistry;
  private readonly vectorStore: VectorStore | null;
  private readonly mappingStrategy: NodeMappingStrategy;
  private readonly metaSelector: MetaDocumentSelector;
  private readonly ingestPipeline: IngestPipeline;
  private readonly pipeline: readonly PipelineStep[];
  private readonly templates: PromptTemplates;
  private queryPipeline: LayeredQueryPipeline;

  constructor(deps: KontextAgentDeps) {
    this.graph = deps.graph;
    this.router = deps.router;
    this.mcpConnectors = deps.mcpConnectors;
    this.mcpLayerAdapters = deps.mcpLayerAdapters;
    this.metaIndexStore = deps.metaIndexStore;
    this.fetcherRegistry = deps.fetcherRegistry;
    this.vectorStore = deps.vectorStore;
    this.mappingStrategy = deps.mappingStrategy;
    this.metaSelector = deps.metaSelector;
    this.ingestPipeline = deps.ingestPipeline;
    this.pipeline = deps.pipeline ?? DEFAULT_PIPELINE;
    this.templates = deps.templates ?? DefaultPromptTemplates;
    this.queryPipeline = this.buildQueryPipeline();
  }

  get ontologyGraph(): OntologyGraph {
    return this.graph;
  }

  get activePipeline(): readonly PipelineStep[] {
    return this.pipeline;
  }

  private buildQueryPipeline(): LayeredQueryPipeline {
    return new LayeredQueryPipeline(
      this.graph,
      this.router,
      this.metaIndexStore,
      this.fetcherRegistry,
      {
        mappingStrategy: this.mappingStrategy,
        metaSelector: this.metaSelector,
        vectorStore: this.vectorStore,
        pipeline: this.pipeline,
        templates: this.templates,
      },
    );
  }

  async query(question: string): Promise<LayeredQueryResult> {
    return this.queryPipeline.execute(question);
  }

  async ingest(data: unknown, source = "manual"): Promise<void> {
    return this.ingestPipeline.ingest("default", data, source);
  }

  /**
   * Sync L2 meta index from connected MCP adapters.
   * If VECTOR step is active, also embeds L3 content with "content:nodeId:docId" keys.
   */
  async syncMCP(connectorName?: string): Promise<void> {
    const targets = connectorName
      ? this.mcpLayerAdapters.filter((a) => a.connectorName === connectorName)
      : this.mcpLayerAdapters;

    const hasVectorStep = this.pipeline.some((s) => s.type === DepthType.VECTOR);

    for (const adapter of targets) {
      for (const nodeId of this.graph.nodes.keys()) {
        const metaDocs = await adapter.listMeta(nodeId);
        if (metaDocs.length === 0) continue;
        await this.metaIndexStore.index(nodeId, metaDocs);

        if (hasVectorStep && this.vectorStore) {
          for (const doc of metaDocs) {
            if (!this.fetcherRegistry.supports(doc.source)) continue;
            try {
              const content = await this.fetcherRegistry.fetch(doc);
              const text = `${doc.title}\n${content.body}`.slice(0, 2000);
              const embedding = await this.vectorStore.embed(text);
              await this.vectorStore.upsert(`content:${nodeId}:${doc.id}`, embedding, {
                nodeId,
                docId: doc.id,
                title: doc.title,
              });
            } catch {
              // ignore
            }
          }
        }
      }
    }
  }

  describeGraph(): string {
    const lines: string[] = ["=== KontextAgent Ontology Graph ==="];
    for (const node of this.graph.nodes.values()) {
      lines.push(`- ${node.id} (weight=${node.weight})`);
      if (node.mcpSource) lines.push(`  MCP: ${node.mcpSource}`);
      if (node.webSearch) lines.push("  Web Search enabled");
      for (const edge of this.graph.edges.filter((e) => e.from === node.id)) {
        lines.push(`  -> ${edge.to} (${edge.weight})`);
      }
    }
    lines.push("");
    lines.push("=== Pipeline ===");
    for (const step of this.pipeline) {
      const extras: string[] = [];
      if (step.maxSelect !== 5) extras.push(`maxSelect=${step.maxSelect}`);
      if (step.threshold > 0) extras.push(`threshold=${step.threshold}`);
      if (step.sectionKey) extras.push(`sectionKey='${step.sectionKey}'`);
      lines.push(`  depth ${step.depth}: ${step.type}${extras.length ? ` ${extras.join(" ")}` : ""}`);
    }
    lines.push("");
    lines.push("=== MCP Adapters ===");
    for (const a of this.mcpLayerAdapters) {
      lines.push(`- ${a.connectorName} (${a.dataSource})`);
    }
    return lines.join("\n");
  }

  // ── autoSetup ───────────────────────────────────────────────

  async autoSetup(targetNodeCount = 10): Promise<AutoSetupResult> {
    const resourceInfos = await this.collectAllResources();
    if (resourceInfos.length === 0) {
      return {
        nodesCreated: 0,
        nodesReused: 0,
        documentsClassified: 0,
        documentsUnmapped: 0,
        ontologyYaml: "",
      };
    }

    const hadExistingNodes = this.graph.nodes.size > 0;
    let newNodes: readonly OntologyNode[];
    let newEdges: readonly Edge[] = [];
    let mappings: ReadonlyMap<string, readonly MCPResourceInfo[]>;
    let unmapped: readonly MCPResourceInfo[];

    if (!hadExistingNodes) {
      // Build from scratch using OntologyAutoBuilder
      const docSources = this.mcpConnectors.map((c) => new MCPDocumentSource(c));
      const builder = new OntologyAutoBuilder(
        this.router.traversalAdapter,
        targetNodeCount,
        20,
        this.templates,
      );
      const buildResult = await builder.build(docSources);
      newNodes = buildResult.nodes;
      newEdges = buildResult.edges;
      this.expandGraph(newNodes, newEdges);

      // Classify documents into the newly-created nodes
      const classifier = new DocumentClassifier(this.router.traversalAdapter, this.templates);
      const classification = await classifier.classify(resourceInfos, this.graph.nodes);
      mappings = classification.mappings;
      unmapped = classification.unmapped;
      if (classification.newNodes.length > 0) {
        this.expandGraph(classification.newNodes, []);
        newNodes = [...newNodes, ...classification.newNodes];
      }
    } else {
      const classifier = new DocumentClassifier(this.router.traversalAdapter, this.templates);
      const classification = await classifier.classify(resourceInfos, this.graph.nodes);
      newNodes = classification.newNodes;
      mappings = classification.mappings;
      unmapped = classification.unmapped;
      this.expandGraph(newNodes, []);
    }

    const classified = await this.indexClassifiedDocuments(mappings);
    const hasVectorStep = this.pipeline.some((s) => s.type === DepthType.VECTOR);
    if (hasVectorStep && this.vectorStore) {
      await this.embedClassifiedContent(mappings);
    }

    const { OntologyYamlWriter } = await import("./ontology-yaml-writer.js");
    const yaml = OntologyYamlWriter.write(
      Array.from(this.graph.nodes.values()),
      this.graph.edges,
    );

    return {
      nodesCreated: newNodes.length,
      nodesReused: hadExistingNodes ? this.graph.nodes.size - newNodes.length : 0,
      documentsClassified: classified,
      documentsUnmapped: unmapped.length,
      ontologyYaml: yaml,
    };
  }

  private async collectAllResources(): Promise<MCPResourceInfo[]> {
    const results: MCPResourceInfo[] = [];
    for (const connector of this.mcpConnectors) {
      const dataSource = this.resolveDataSource(connector);
      try {
        const resources = await connector.listResources();
        for (const r of resources) {
          results.push({
            id: r.id,
            title: r.name,
            description: r.description,
            source: dataSource,
            connectorName: connector.name,
          });
        }
      } catch {
        // ignore failing connectors
      }
    }
    return results;
  }

  private resolveDataSource(connector: MCPConnector): DataSource {
    const adapter = this.mcpLayerAdapters.find((a) => a.connectorName === connector.name);
    return adapter?.dataSource ?? ("CUSTOM" as DataSource);
  }

  private expandGraph(newNodes: readonly OntologyNode[], newEdges: readonly Edge[]): void {
    if (newNodes.length === 0 && newEdges.length === 0) return;
    const mergedNodes = new Map(this.graph.nodes);
    for (const node of newNodes) {
      if (!mergedNodes.has(node.id)) mergedNodes.set(node.id, node);
    }
    const mergedEdges = [...this.graph.edges, ...newEdges];
    this.graph = new OntologyGraph(mergedNodes, mergedEdges, this.graph.config);
    this.queryPipeline = this.buildQueryPipeline();

    // embed new nodes async (fire-and-forget is fine here, but we await to keep tests deterministic)
    if (this.vectorStore) {
      const vs = this.vectorStore;
      // eslint-disable-next-line @typescript-eslint/no-floating-promises
      Promise.all(
        newNodes.map(async (node) => {
          try {
            const emb = await vs.embed(node.description);
            await vs.upsert(node.id, emb);
          } catch {
            // ignore
          }
        }),
      );
    }
  }

  private async indexClassifiedDocuments(
    mappings: ReadonlyMap<string, readonly MCPResourceInfo[]>,
  ): Promise<number> {
    let count = 0;
    for (const [nodeId, resources] of mappings) {
      if (resources.length === 0) continue;
      const metaDocs: MetaDocument[] = resources.map((r) =>
        createMetaDocument({
          id: r.id,
          title: r.title,
          source: r.source,
          ontologyNodeId: nodeId,
          metadata: { connector: r.connectorName },
        }),
      );
      await this.metaIndexStore.index(nodeId, metaDocs);
      count += metaDocs.length;
    }
    return count;
  }

  private async embedClassifiedContent(
    mappings: ReadonlyMap<string, readonly MCPResourceInfo[]>,
  ): Promise<void> {
    const vs = this.vectorStore;
    if (!vs) return;
    for (const [nodeId, resources] of mappings) {
      for (const r of resources) {
        if (!this.fetcherRegistry.supports(r.source)) continue;
        try {
          const metaDoc = createMetaDocument({
            id: r.id,
            title: r.title,
            source: r.source,
            ontologyNodeId: nodeId,
          });
          const content = await this.fetcherRegistry.fetch(metaDoc);
          const text = `${r.title}\n${content.body}`.slice(0, 2000);
          const embedding = await vs.embed(text);
          await vs.upsert(`content:${nodeId}:${r.id}`, embedding, {
            nodeId,
            docId: r.id,
            title: r.title,
          });
        } catch {
          // ignore
        }
      }
    }
  }
}
