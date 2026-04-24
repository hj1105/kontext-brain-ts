import type { Edge, OntologyNode } from "../graph/ontology-node.js";
import { createNode } from "../graph/ontology-node.js";
import type { LLMAdapter } from "../query/llm-adapter.js";
import type { PromptTemplates } from "../query/prompt-templates.js";
import { DefaultPromptTemplates } from "../query/prompt-templates.js";

export interface SourceDocument {
  readonly id: string;
  readonly title: string;
  readonly metadata: Readonly<Record<string, string>>;
}

export interface DocumentSource {
  collect(): Promise<SourceDocument[]>;
}

export interface OntologyBuildResult {
  readonly nodes: readonly OntologyNode[];
  readonly edges: readonly Edge[];
  readonly docCount: number;
}

export const emptyOntologyBuildResult: OntologyBuildResult = {
  nodes: [],
  edges: [],
  docCount: 0,
};

/**
 * Auto-builds an ontology (nodes + edges) from document sources.
 *
 * Pipeline:
 *   1. Collect docs from all sources
 *   2. Parallel batch: extract topic categories per batch (Haiku)
 *   3. Cluster categories into N nodes with level/parentId (LLM)
 *   4. Infer edges between nodes (LLM)
 */
export class OntologyAutoBuilder {
  constructor(
    private readonly adapter: LLMAdapter,
    private readonly targetNodeCount = 10,
    private readonly batchSize = 20,
    private readonly templates: PromptTemplates = DefaultPromptTemplates,
  ) {}

  async build(sources: readonly DocumentSource[]): Promise<OntologyBuildResult> {
    const docs = (await Promise.all(sources.map((s) => s.collect()))).flat();
    if (docs.length === 0) return emptyOntologyBuildResult;

    const categories = await this.extractCategories(docs);
    const nodes = await this.clusterToNodes(docs, categories);
    const edges = await this.inferEdges(nodes);

    return { nodes, edges, docCount: docs.length };
  }

  private async extractCategories(docs: readonly SourceDocument[]): Promise<string[]> {
    const batches: SourceDocument[][] = [];
    for (let i = 0; i < docs.length; i += this.batchSize) {
      batches.push(docs.slice(i, i + this.batchSize));
    }
    const results = await Promise.all(batches.map((b) => this.extractBatchCategories(b)));
    return Array.from(new Set(results.flat()));
  }

  private async extractBatchCategories(batch: readonly SourceDocument[]): Promise<string[]> {
    const docList = batch
      .map((d) => {
        const metaStr = Object.entries(d.metadata)
          .slice(0, 2)
          .map(([, v]) => ` [${v}]`)
          .join("");
        return `- ${d.title}${metaStr}`;
      })
      .join("\n");

    const response = await this.adapter.complete(
      this.templates.categoryExtraction,
      docList,
      "Extract topic categories from these documents.",
    );

    try {
      const clean = response.trim().replace(/^```json/, "").replace(/```$/, "").trim();
      const parsed = JSON.parse(clean);
      if (Array.isArray(parsed)) return parsed.map((x) => String(x));
    } catch {
      // ignore
    }
    return [];
  }

  private async clusterToNodes(
    docs: readonly SourceDocument[],
    rawCategories: readonly string[],
  ): Promise<OntologyNode[]> {
    const docTitles = docs.slice(0, 100).map((d) => `- ${d.title}`).join("\n");
    const catList = rawCategories.join(", ");

    const response = await this.adapter.complete(
      this.templates.nodeDesign(this.targetNodeCount),
      `Documents:\n${docTitles}\n\nExtracted categories: ${catList}`,
      "Design ontology nodes.",
    );

    try {
      const clean = response.trim().replace(/^```json/, "").replace(/```$/, "").trim();
      const parsed: unknown = JSON.parse(clean);
      if (typeof parsed !== "object" || parsed === null || !("nodes" in parsed)) return [];
      const rawNodes = (parsed as { nodes: unknown }).nodes;
      if (!Array.isArray(rawNodes)) return [];
      return rawNodes.map((n: Record<string, unknown>) => {
        const parentId = n.parentId;
        return createNode({
          id: String(n.id),
          description: String(n.description ?? ""),
          weight: typeof n.weight === "number" ? n.weight : 0.8,
          level: typeof n.level === "number" ? n.level : 0,
          parentId:
            parentId && parentId !== "null" && parentId !== null ? String(parentId) : null,
        });
      });
    } catch {
      return [];
    }
  }

  private async inferEdges(nodes: readonly OntologyNode[]): Promise<Edge[]> {
    if (nodes.length < 2) return [];

    const nodeList = nodes.map((n) => `- ${n.id}: ${n.description}`).join("\n");
    const response = await this.adapter.complete(
      this.templates.edgeInference,
      nodeList,
      "Infer relationships between nodes.",
    );

    try {
      const clean = response.trim().replace(/^```json/, "").replace(/```$/, "").trim();
      const parsed: unknown = JSON.parse(clean);
      if (typeof parsed !== "object" || parsed === null || !("edges" in parsed)) return [];
      const rawEdges = (parsed as { edges: unknown }).edges;
      if (!Array.isArray(rawEdges)) return [];
      const nodeIds = new Set(nodes.map((n) => n.id));
      const edges: Edge[] = [];
      for (const e of rawEdges) {
        if (typeof e !== "object" || e === null) continue;
        const o = e as Record<string, unknown>;
        const from = typeof o.from === "string" ? o.from : null;
        const to = typeof o.to === "string" ? o.to : null;
        if (!from || !to || !nodeIds.has(from) || !nodeIds.has(to)) continue;
        edges.push({
          from,
          to,
          weight: typeof o.weight === "number" ? o.weight : 0.6,
        });
      }
      return edges;
    } catch {
      return [];
    }
  }
}

// ── DocumentSource implementations ───────────────────────────

export class InMemoryDocumentSource implements DocumentSource {
  constructor(private readonly documents: readonly SourceDocument[]) {}
  async collect(): Promise<SourceDocument[]> {
    return [...this.documents];
  }
}
