import type { Edge, OntologyNode } from "../graph/ontology-node.js";
import { createNode } from "../graph/ontology-node.js";
import type { LLMAdapter } from "../query/llm-adapter.js";
import type { PromptTemplates } from "../query/prompt-templates.js";
import { DefaultPromptTemplates } from "../query/prompt-templates.js";
import type { VectorStore } from "../query/vector-store.js";
import type {
  OntologyStore,
  SerializableEdge,
  SerializableNode,
  UserOntologyGraph,
} from "../store/ontology-store.js";

export interface OntologyNodeContribution {
  readonly node: OntologyNode;
  readonly autoEdges: readonly Edge[];
}

export interface OntologyNodeSource {
  readonly sourceName: string;
  contributeNodes(): Promise<OntologyNodeContribution[]>;
}

export interface ExtractionResult {
  readonly newNodes: readonly OntologyNode[];
  readonly newEdges: readonly Edge[];
}

export class IngestPipeline {
  constructor(
    private readonly traversalAdapter: LLMAdapter,
    private readonly store: OntologyStore,
    private readonly vectorStore: VectorStore,
    private readonly templates: PromptTemplates = DefaultPromptTemplates,
  ) {}

  async ingest(userId: string, data: unknown, source = "manual"): Promise<void> {
    const text = String(data);
    const extracted = await this.extractEntities(text, source);
    const existing = await this.store.load(userId);
    const merged = merge(existing, extracted);

    for (const node of extracted.newNodes) {
      const embedding = await this.vectorStore.embed(node.description);
      await this.vectorStore.upsert(`${userId}:${node.id}`, embedding, {
        userId,
        nodeId: node.id,
      });
    }

    await this.store.save(userId, merged);
  }

  async ingestFromSource(userId: string, source: OntologyNodeSource): Promise<void> {
    const contributions = await source.contributeNodes();
    const existing = await this.store.load(userId);
    const extraction: ExtractionResult = {
      newNodes: contributions.map((c) => c.node),
      newEdges: contributions.flatMap((c) => c.autoEdges),
    };
    await this.store.save(userId, merge(existing, extraction));
  }

  private async extractEntities(text: string, source: string): Promise<ExtractionResult> {
    const response = await this.traversalAdapter.complete(
      `${this.templates.entityExtraction}\nText source: ${source}`,
      "",
      text.slice(0, 4000),
    );
    return parseExtraction(response);
  }
}

function parseExtraction(response: string): ExtractionResult {
  try {
    const clean = response.trim().replace(/^```json/, "").replace(/```$/, "").trim();
    const json = JSON.parse(clean) as {
      nodes?: Array<Record<string, unknown>>;
      edges?: Array<Record<string, unknown>>;
    };
    const nodes = (json.nodes ?? []).map((n) =>
      createNode({
        id: String(n.id),
        description: String(n.description ?? ""),
        weight: typeof n.weight === "number" ? n.weight : 0.5,
      }),
    );
    const edges: Edge[] = (json.edges ?? []).map((e) => ({
      from: String(e.from),
      to: String(e.to),
      weight: typeof e.weight === "number" ? e.weight : 0.5,
    }));
    return { newNodes: nodes, newEdges: edges };
  } catch {
    return { newNodes: [], newEdges: [] };
  }
}

function merge(existing: UserOntologyGraph, extracted: ExtractionResult): UserOntologyGraph {
  const mergedNodes: Record<string, SerializableNode> = { ...existing.nodes };
  const mergedEdges: SerializableEdge[] = [...existing.edges];

  for (const node of extracted.newNodes) {
    const prev = mergedNodes[node.id];
    if (prev) {
      mergedNodes[node.id] = { ...prev, weight: Math.min(1.0, prev.weight + 0.05) };
    } else {
      mergedNodes[node.id] = {
        id: node.id,
        description: node.description,
        weight: node.weight,
        mcpSource: node.mcpSource ?? null,
        webSearch: node.webSearch,
      };
    }
  }
  for (const edge of extracted.newEdges) {
    const exists = mergedEdges.some((e) => e.from === edge.from && e.to === edge.to);
    if (!exists) mergedEdges.push({ from: edge.from, to: edge.to, weight: edge.weight });
  }
  return { ...existing, nodes: mergedNodes, edges: mergedEdges };
}
