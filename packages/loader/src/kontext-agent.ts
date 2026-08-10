import {
  type AnswerGroundingValidator,
  type BidirectionalNLayerRetriever,
  CitationAnswerValidator,
  type ContentFetcherRegistry,
  DEFAULT_PIPELINE,
  type DataSource,
  DefaultPromptTemplates,
  DefaultTokenEstimator,
  DepthType,
  DocumentClassifier,
  type Edge,
  InMemoryOntologyProposalQueue,
  InMemoryOntologyStore,
  type IngestPipeline,
  LayeredQueryPipeline,
  type LayeredQueryResult,
  type LayeredRetrievalResult,
  type MCPResourceInfo,
  type MetaDocument,
  type MetaDocumentSelector,
  type MetaIndexStore,
  NoAccessibleEvidenceError,
  type NodeMappingStrategy,
  OntologyAutoBuilder,
  OntologyGraph,
  type OntologyNode,
  type OntologyProposal,
  type OntologyProposalQueue,
  type OntologyStore,
  type PipelineStep,
  type Principal,
  type PromptTemplates,
  type RouterLLMAdapter,
  type SerializableResourceRecord,
  type TokenEstimator,
  type VectorStore,
  createMetaDocument,
  createPersistedGraphState,
  resourceDocumentIdentity,
} from "@kontext-brain/core";
import {
  type MCPConnector,
  MCPDocumentSource,
  type MCPKnowledgeSynchronizer,
  type MCPLayerAdapter,
  type MCPResource,
} from "@kontext-brain/mcp";

export interface AutoSetupResult {
  readonly nodesCreated: number;
  readonly nodesReused: number;
  readonly documentsClassified: number;
  readonly documentsUnmapped: number;
  readonly ontologyYaml: string;
}

export interface SyncMCPResult {
  readonly connectorsSynced: number;
  readonly resourcesAdded: number;
  readonly resourcesUpdated: number;
  readonly resourcesRemoved: number;
  readonly resourcesClassified: number;
  readonly resourcesUnmapped: number;
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
  tokenEstimator?: TokenEstimator;
  ontologyStore?: OntologyStore;
  stateId?: string;
  resourceRecords?: readonly SerializableResourceRecord[];
  ontologyContentHash?: string;
  organizationId?: string;
  ontologyProposalQueue?: OntologyProposalQueue;
  knowledgeRetriever?: BidirectionalNLayerRetriever;
  mcpKnowledgeSynchronizer?: MCPKnowledgeSynchronizer;
  answerValidator?: AnswerGroundingValidator;
}

/**
 * High-level owner of the runtime graph, indexes, MCP assignments, and their
 * persisted snapshot. All mutations flow through this class so query state
 * cannot drift from storage state.
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
  private readonly tokenEstimator: TokenEstimator;
  private readonly ontologyStore: OntologyStore;
  private readonly stateId: string;
  private readonly organizationId: string;
  private readonly ontologyProposalQueue: OntologyProposalQueue;
  private readonly ontologyContentHash?: string;
  private readonly knowledgeRetriever?: BidirectionalNLayerRetriever;
  private readonly mcpKnowledgeSynchronizer?: MCPKnowledgeSynchronizer;
  private readonly answerValidator: AnswerGroundingValidator;
  private readonly resourceRecords = new Map<string, SerializableResourceRecord>();
  private mutationQueue: Promise<void> = Promise.resolve();
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
    this.tokenEstimator = deps.tokenEstimator ?? DefaultTokenEstimator;
    this.ontologyStore = deps.ontologyStore ?? new InMemoryOntologyStore();
    this.stateId = deps.stateId ?? "default";
    this.organizationId = deps.organizationId ?? this.stateId;
    this.ontologyProposalQueue = deps.ontologyProposalQueue ?? new InMemoryOntologyProposalQueue();
    this.ontologyContentHash = deps.ontologyContentHash;
    this.knowledgeRetriever = deps.knowledgeRetriever;
    this.mcpKnowledgeSynchronizer = deps.mcpKnowledgeSynchronizer;
    this.answerValidator = deps.answerValidator ?? new CitationAnswerValidator();
    for (const record of deps.resourceRecords ?? []) {
      this.resourceRecords.set(resourceKey(record.connectorName, record.resourceId), record);
    }
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
        tokenEstimator: this.tokenEstimator,
      },
    );
  }

  /** Retrieve evidence without invoking the final reasoning model. */
  async retrieve(question: string, principal?: Principal): Promise<LayeredRetrievalResult> {
    await this.mutationQueue;
    if (this.knowledgeRetriever) {
      if (!principal) {
        throw new Error("A Principal is required for evidence-backed retrieval");
      }
      if (principal.organizationId !== this.organizationId) {
        throw new Error(
          `Organization mismatch: expected "${this.organizationId}", received "${principal.organizationId}"`,
        );
      }
      const result = await this.knowledgeRetriever.retrieve({ question, principal });
      const context = result.evidence
        .map(
          (evidence) =>
            `[Evidence ${evidence.evidenceId}; Resource ${evidence.resourceId}; Chunk ${evidence.chunkId}]\n${evidence.text}`,
        )
        .join("\n\n---\n\n");
      return {
        context,
        usedOntologyNodes: [],
        selectedMetaDocs: [],
        fetchedContents: [],
        contextTokensUsed: this.tokenEstimator.estimate(context),
        traversalPath: [],
        pipelineSteps: [],
        pipelineTraces: [],
        retrievalMode: "bidirectional",
        evidence: result.evidence,
        searchTrace: result.trace,
      };
    }
    return this.queryPipeline.retrieve(question);
  }

  /** Retrieve evidence and invoke the final reasoning model. */
  async answer(question: string, principal?: Principal): Promise<LayeredQueryResult> {
    await this.mutationQueue;
    const retrieval = await this.retrieve(question, principal);
    if (retrieval.retrievalMode === "bidirectional" && (retrieval.evidence?.length ?? 0) === 0) {
      throw new NoAccessibleEvidenceError();
    }
    const result = await this.queryPipeline.answer(question, retrieval);
    if (retrieval.retrievalMode === "bidirectional") {
      await this.answerValidator.validate(result.answer, retrieval.evidence ?? []);
    }
    return result;
  }

  /** Backward-compatible alias for `answer()`. */
  async query(question: string, principal?: Principal): Promise<LayeredQueryResult> {
    return this.answer(question, principal);
  }

  /**
   * Rebuild runtime-only indexes from persisted resource assignments.
   * KontextLoader calls this once after constructing an agent.
   */
  async initialize(): Promise<void> {
    await this.rebuildMetaIndex();
    if (this.hasVectorStep()) {
      await this.embedResourceContent(Array.from(this.resourceRecords.values()));
    }
  }

  async ingest(data: unknown, source = "manual"): Promise<void> {
    await this.runMutation(async () => {
      const extracted = await this.ingestPipeline.extract(data, source);
      await this.expandGraph(extracted.newNodes, extracted.newEdges);
      await this.persistState();
    });
  }

  /**
   * Incrementally synchronize MCP resources.
   *
   * Existing assignments are retained for unchanged resources, changed/new
   * resources are classified, and deleted resources are removed from both
   * meta and vector indexes. Resources are never copied into every node.
   */
  async syncMCP(connectorName?: string): Promise<SyncMCPResult> {
    return this.runMutation(() => this.syncMCPUnlocked(connectorName));
  }

  private async syncMCPUnlocked(connectorName?: string): Promise<SyncMCPResult> {
    const targets = connectorName
      ? this.mcpConnectors.filter((connector) => connector.name === connectorName)
      : this.mcpConnectors;

    if (targets.length === 0) {
      return emptySyncResult;
    }
    if (this.graph.nodes.size === 0) {
      const setup = await this.autoSetupUnlocked(10, targets);
      return {
        connectorsSynced: targets.length,
        resourcesAdded: setup.documentsClassified + setup.documentsUnmapped,
        resourcesUpdated: 0,
        resourcesRemoved: 0,
        resourcesClassified: setup.documentsClassified,
        resourcesUnmapped: setup.documentsUnmapped,
      };
    }

    const toClassify: MCPResourceInfo[] = [];
    const previousByResource = new Map<string, SerializableResourceRecord>();
    const removed: SerializableResourceRecord[] = [];
    let connectorsSynced = 0;
    let resourcesAdded = 0;
    let resourcesUpdated = 0;

    for (const connector of targets) {
      let resources: MCPResource[];
      try {
        resources = await connector.listResources();
      } catch {
        // A failed connector must not make all of its prior resources look deleted.
        continue;
      }
      connectorsSynced++;
      const source = this.resolveDataSource(connector);
      const currentIds = new Set(resources.map((resource) => resource.id));
      const existing = Array.from(this.resourceRecords.values()).filter(
        (record) => record.connectorName === connector.name,
      );
      const seenAt = new Date().toISOString();

      for (const record of existing) {
        if (!currentIds.has(record.resourceId)) {
          removed.push(record);
          this.resourceRecords.delete(resourceKey(record.connectorName, record.resourceId));
        }
      }

      for (const resource of resources) {
        const key = resourceKey(connector.name, resource.id);
        const previous = this.resourceRecords.get(key);
        const nextSignature = resourceSignature(resource.name, resource.description, source);
        if (!previous) {
          resourcesAdded++;
        } else if (previous.signature !== nextSignature || previous.nodeIds.length === 0) {
          resourcesUpdated++;
          previousByResource.set(key, previous);
        } else {
          this.resourceRecords.set(key, { ...previous, lastSeenAt: seenAt });
          continue;
        }
        toClassify.push({
          id: resource.id,
          title: resource.name,
          description: resource.description,
          source,
          connectorName: connector.name,
        });
      }
    }

    for (const record of [...removed, ...previousByResource.values()]) {
      await this.deleteContentEmbeddings(record);
    }

    let resourcesClassified = 0;
    let resourcesUnmapped = 0;
    const changedRecords: SerializableResourceRecord[] = [];
    if (toClassify.length > 0) {
      const classifier = new DocumentClassifier(this.router.traversalAdapter, this.templates);
      const classification = await classifier.classify(toClassify, this.graph.nodes);
      await this.ontologyProposalQueue.enqueue(this.organizationId, classification.proposals);

      const assignments = reverseMappings(classification.mappings);
      const now = new Date().toISOString();
      for (const resource of toClassify) {
        const key = resourceKey(resource.connectorName, resource.id);
        const previous = previousByResource.get(key);
        const assignedNodeIds = assignments.get(key);
        const nodeIds = assignedNodeIds ?? (previous?.nodeIds.length ? previous.nodeIds : []);
        const signature =
          assignedNodeIds !== undefined || !previous
            ? resourceSignature(resource.title, resource.description, resource.source)
            : previous.signature;
        const record: SerializableResourceRecord = {
          connectorName: resource.connectorName,
          resourceId: resource.id,
          title: resource.title,
          description: resource.description,
          source: resource.source,
          nodeIds,
          signature,
          lastSeenAt: now,
        };
        this.resourceRecords.set(key, record);
        changedRecords.push(record);
        if (assignedNodeIds !== undefined) resourcesClassified++;
        else resourcesUnmapped++;
      }
    }

    await this.rebuildMetaIndex();
    if (this.hasVectorStep()) {
      await this.embedResourceContent(changedRecords);
    }
    await this.syncKnowledgeResources(changedRecords, removed);
    await this.persistState();

    return {
      connectorsSynced,
      resourcesAdded,
      resourcesUpdated,
      resourcesRemoved: removed.length,
      resourcesClassified,
      resourcesUnmapped,
    };
  }

  describeGraph(): string {
    const lines: string[] = ["=== KontextAgent Ontology Graph ==="];
    for (const node of this.graph.nodes.values()) {
      lines.push(`- ${node.id} (weight=${node.weight})`);
      if (node.mcpSource) lines.push(`  MCP: ${node.mcpSource}`);
      if (node.webSearch) lines.push("  Web Search enabled");
      for (const edge of this.graph.edges.filter((item) => item.from === node.id)) {
        lines.push(`  -> ${edge.to} (${edge.weight})`);
      }
    }
    lines.push("");
    lines.push("=== Pipeline ===");
    for (const pipelineStep of this.pipeline) {
      const extras: string[] = [];
      if (pipelineStep.maxSelect !== 5) {
        extras.push(`maxSelect=${pipelineStep.maxSelect}`);
      }
      if (pipelineStep.threshold > 0) {
        extras.push(`threshold=${pipelineStep.threshold}`);
      }
      if (pipelineStep.sectionKey) {
        extras.push(`sectionKey='${pipelineStep.sectionKey}'`);
      }
      lines.push(
        `  stage ${pipelineStep.depth}: ${pipelineStep.type}${
          extras.length ? ` ${extras.join(" ")}` : ""
        }`,
      );
    }
    lines.push("");
    lines.push("=== MCP Adapters ===");
    for (const adapter of this.mcpLayerAdapters) {
      const count = Array.from(this.resourceRecords.values()).filter(
        (record) => record.connectorName === adapter.connectorName,
      ).length;
      lines.push(`- ${adapter.connectorName} (${adapter.dataSource}, resources=${count})`);
    }
    return lines.join("\n");
  }

  async listOntologyProposals(): Promise<readonly OntologyProposal[]> {
    return this.ontologyProposalQueue.listOpen(this.organizationId);
  }

  // ── autoSetup ───────────────────────────────────────────────

  async autoSetup(targetNodeCount = 10): Promise<AutoSetupResult> {
    return this.runMutation(() => this.autoSetupUnlocked(targetNodeCount));
  }

  private async autoSetupUnlocked(
    targetNodeCount = 10,
    connectors: readonly MCPConnector[] = this.mcpConnectors,
  ): Promise<AutoSetupResult> {
    const resourceInfos = await this.collectAllResources(connectors);
    if (resourceInfos.length === 0) {
      return {
        nodesCreated: 0,
        nodesReused: this.graph.nodes.size,
        documentsClassified: 0,
        documentsUnmapped: 0,
        ontologyYaml: "",
      };
    }

    const initialNodeCount = this.graph.nodes.size;
    let newNodes: readonly OntologyNode[] = [];

    if (initialNodeCount === 0) {
      const documentSources = connectors.map((connector) => new MCPDocumentSource(connector));
      const builder = new OntologyAutoBuilder(
        this.router.traversalAdapter,
        targetNodeCount,
        20,
        this.templates,
      );
      const buildResult = await builder.build(documentSources);
      newNodes = buildResult.nodes;
      await this.expandGraph(buildResult.nodes, buildResult.edges);
    }

    const classifier = new DocumentClassifier(this.router.traversalAdapter, this.templates);
    const classification = await classifier.classify(resourceInfos, this.graph.nodes);
    await this.ontologyProposalQueue.enqueue(this.organizationId, classification.proposals);

    const assignments = reverseMappings(classification.mappings);
    const now = new Date().toISOString();
    for (const resource of resourceInfos) {
      const key = resourceKey(resource.connectorName, resource.id);
      this.resourceRecords.set(key, {
        connectorName: resource.connectorName,
        resourceId: resource.id,
        title: resource.title,
        description: resource.description,
        source: resource.source,
        nodeIds: assignments.get(key) ?? [],
        signature: resourceSignature(resource.title, resource.description, resource.source),
        lastSeenAt: now,
      });
    }

    await this.rebuildMetaIndex();
    const classifiedRecords = resourceInfos
      .map((resource) => this.resourceRecords.get(resourceKey(resource.connectorName, resource.id)))
      .filter(
        (record): record is SerializableResourceRecord =>
          record !== undefined && record.nodeIds.length > 0,
      );
    if (this.hasVectorStep()) {
      await this.embedResourceContent(classifiedRecords);
    }
    await this.syncKnowledgeResources(
      resourceInfos
        .map((resource) =>
          this.resourceRecords.get(resourceKey(resource.connectorName, resource.id)),
        )
        .filter((record): record is SerializableResourceRecord => record !== undefined),
      [],
    );
    await this.persistState();

    const { OntologyYamlWriter } = await import("./ontology-yaml-writer.js");
    const yaml = OntologyYamlWriter.write(Array.from(this.graph.nodes.values()), this.graph.edges);

    return {
      nodesCreated: new Set(newNodes.map((node) => node.id)).size,
      nodesReused: initialNodeCount,
      documentsClassified: classifiedRecords.length,
      documentsUnmapped: classification.unmapped.length,
      ontologyYaml: yaml,
    };
  }

  private async collectAllResources(
    connectors: readonly MCPConnector[] = this.mcpConnectors,
  ): Promise<MCPResourceInfo[]> {
    const results: MCPResourceInfo[] = [];
    for (const connector of connectors) {
      const source = this.resolveDataSource(connector);
      try {
        const resources = await connector.listResources();
        for (const resource of resources) {
          results.push({
            id: resource.id,
            title: resource.name,
            description: resource.description,
            source,
            connectorName: connector.name,
          });
        }
      } catch {
        // A single unavailable source should not block other sources.
      }
    }
    return results;
  }

  private resolveDataSource(connector: MCPConnector): DataSource {
    const adapter = this.mcpLayerAdapters.find(
      (candidate) => candidate.connectorName === connector.name,
    );
    return adapter?.dataSource ?? ("CUSTOM" as DataSource);
  }

  private async expandGraph(
    newNodes: readonly OntologyNode[],
    newEdges: readonly Edge[],
  ): Promise<void> {
    const mergedNodes = new Map(this.graph.nodes);
    const addedNodes: OntologyNode[] = [];
    for (const node of newNodes) {
      if (mergedNodes.has(node.id)) continue;
      mergedNodes.set(node.id, node);
      addedNodes.push(node);
    }

    const edgeKeys = new Set(
      this.graph.edges.map((edge) => `${edge.from}\u0000${edge.to}\u0000${edge.type ?? ""}`),
    );
    const mergedEdges = [...this.graph.edges];
    for (const edge of newEdges) {
      const key = `${edge.from}\u0000${edge.to}\u0000${edge.type ?? ""}`;
      if (edgeKeys.has(key)) continue;
      edgeKeys.add(key);
      mergedEdges.push(edge);
    }

    if (addedNodes.length === 0 && mergedEdges.length === this.graph.edges.length) {
      return;
    }
    this.graph = new OntologyGraph(mergedNodes, mergedEdges, this.graph.config);
    this.queryPipeline = this.buildQueryPipeline();

    if (this.vectorStore) {
      for (const node of addedNodes) {
        try {
          const embedding = await this.vectorStore.embed(node.description);
          await this.vectorStore.upsert(node.id, embedding);
        } catch {
          // Embeddings are an optimization; graph state remains authoritative.
        }
      }
    }
  }

  private async rebuildMetaIndex(): Promise<void> {
    const existing = await this.metaIndexStore.all();
    const byNode = new Map<string, MetaDocument[]>();

    // Preserve programmatically indexed documents that are not MCP-managed.
    for (const [nodeId, documents] of existing) {
      byNode.set(
        nodeId,
        documents.filter((document) => !document.metadata.connector),
      );
    }

    for (const record of this.resourceRecords.values()) {
      for (const nodeId of record.nodeIds) {
        if (!this.graph.nodes.has(nodeId)) continue;
        const documents = byNode.get(nodeId) ?? [];
        documents.push(
          createMetaDocument({
            id: record.resourceId,
            title: record.title,
            source: record.source,
            ontologyNodeId: nodeId,
            metadata: {
              connector: record.connectorName,
              signature: record.signature,
            },
          }),
        );
        byNode.set(nodeId, documents);
      }
    }

    const nodeIds = new Set([...existing.keys(), ...this.graph.nodes.keys()]);
    for (const nodeId of nodeIds) {
      await this.metaIndexStore.replace(nodeId, byNode.get(nodeId) ?? []);
    }
  }

  private async embedResourceContent(
    records: readonly SerializableResourceRecord[],
  ): Promise<void> {
    const vectorStore = this.vectorStore;
    if (!vectorStore) return;

    for (const record of records) {
      if (
        record.nodeIds.length === 0 ||
        !this.fetcherRegistry.supports(record.source, record.connectorName)
      ) {
        continue;
      }
      try {
        const primaryNodeId = record.nodeIds[0];
        if (!primaryNodeId) continue;
        const metaDocument = createMetaDocument({
          id: record.resourceId,
          title: record.title,
          source: record.source,
          ontologyNodeId: primaryNodeId,
          metadata: { connector: record.connectorName },
        });
        const content = await this.fetcherRegistry.fetch(metaDocument);
        const embedding = await vectorStore.embed(
          `${record.title}\n${content.body}`.slice(0, 4000),
        );
        for (const nodeId of record.nodeIds) {
          await vectorStore.upsert(
            `content:${nodeId}:${resourceDocumentIdentity(
              record.connectorName,
              record.resourceId,
            )}`,
            embedding,
            {
              nodeId,
              docId: record.resourceId,
              title: record.title,
              connector: record.connectorName,
            },
          );
        }
      } catch {
        // Keep metadata searchable when content fetch or embedding fails.
      }
    }
  }

  private async deleteContentEmbeddings(record: SerializableResourceRecord): Promise<void> {
    if (!this.vectorStore?.delete) return;
    for (const nodeId of record.nodeIds) {
      await this.vectorStore.delete(
        `content:${nodeId}:${resourceDocumentIdentity(record.connectorName, record.resourceId)}`,
      );
    }
  }

  private async syncKnowledgeResources(
    changed: readonly SerializableResourceRecord[],
    removed: readonly SerializableResourceRecord[],
  ): Promise<void> {
    const synchronizer = this.mcpKnowledgeSynchronizer;
    if (!synchronizer) return;
    for (const record of changed) {
      const connector = this.mcpConnectors.find(
        (candidate) => candidate.name === record.connectorName,
      );
      if (!connector) continue;
      await synchronizer.sync(
        this.organizationId,
        connector,
        {
          id: record.resourceId,
          name: record.title,
          description: record.description,
        },
        record.nodeIds,
      );
    }
    for (const record of removed) {
      await synchronizer.remove(this.organizationId, record.connectorName, record.resourceId);
    }
  }

  private hasVectorStep(): boolean {
    return this.pipeline.some((pipelineStep) => pipelineStep.type === DepthType.VECTOR);
  }

  private async persistState(): Promise<void> {
    const metaDocuments = await this.metaIndexStore.all();
    const snapshot = createPersistedGraphState(
      this.stateId,
      this.graph,
      metaDocuments,
      Array.from(this.resourceRecords.values()),
      this.ontologyContentHash,
    );
    await this.ontologyStore.save(this.stateId, snapshot);
  }

  private async runMutation<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.mutationQueue;
    let release: () => void = () => undefined;
    this.mutationQueue = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    try {
      return await operation();
    } finally {
      release();
    }
  }
}

const emptySyncResult: SyncMCPResult = {
  connectorsSynced: 0,
  resourcesAdded: 0,
  resourcesUpdated: 0,
  resourcesRemoved: 0,
  resourcesClassified: 0,
  resourcesUnmapped: 0,
};

function resourceKey(connectorName: string, resourceId: string): string {
  return `${connectorName}\u0000${resourceId}`;
}

function resourceSignature(title: string, description: string, source: DataSource): string {
  return `${source}\u0000${title}\u0000${description}`;
}

function reverseMappings(
  mappings: ReadonlyMap<string, readonly MCPResourceInfo[]>,
): Map<string, string[]> {
  const assignments = new Map<string, string[]>();
  for (const [nodeId, resources] of mappings) {
    for (const resource of resources) {
      const key = resourceKey(resource.connectorName, resource.id);
      const nodeIds = assignments.get(key) ?? [];
      if (!nodeIds.includes(nodeId)) nodeIds.push(nodeId);
      assignments.set(key, nodeIds);
    }
  }
  return assignments;
}
