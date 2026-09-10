import type { Edge, OntologyNode } from "../graph/ontology-node.js";
import { createNode } from "../graph/ontology-node.js";
import type { LLMAdapter } from "../query/llm-adapter.js";
import type { PromptTemplates } from "../query/prompt-templates.js";
import { DefaultPromptTemplates } from "../query/prompt-templates.js";
import { mapWithConcurrency, stratifiedSample } from "./bounded-concurrency.js";

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

export const MIN_AUTO_NODE_COUNT = 3;

/**
 * A safety rail on prompt size, not a statement about how many concepts a
 * Codebase may have. The previous fixed ceiling of 20 was the latter, so a
 * Codebase with more governance boundaries than that had two of them merged
 * into one Ontology Node — and merged boundaries are what makes a symbol
 * receive a neighbouring area's approved decision as if it were its own.
 */
export const ABSOLUTE_MAX_AUTO_NODE_COUNT = 200;

function assertValidTargetNodeCount(value: number): number {
  if (
    !Number.isInteger(value) ||
    value < MIN_AUTO_NODE_COUNT ||
    value > ABSOLUTE_MAX_AUTO_NODE_COUNT
  ) {
    throw new RangeError(
      `targetNodeCount must be an integer between ${MIN_AUTO_NODE_COUNT} and ${ABSOLUTE_MAX_AUTO_NODE_COUNT}`,
    );
  }
  return value;
}

function inferTargetNodeCount(documentCount: number, categoryCount: number): number {
  const maxUsefulNodeCount = Math.min(ABSOLUTE_MAX_AUTO_NODE_COUNT, documentCount);
  const minUsefulNodeCount = Math.min(MIN_AUTO_NODE_COUNT, maxUsefulNodeCount);
  // Grow sublinearly with corpus size, and let extracted topics raise the
  // target one-for-one. Halving the topic count merges distinct areas, and for
  // a governance ontology that is the expensive direction to be wrong in: a
  // merged node hands a symbol its neighbour's approved decision with the same
  // authority as its own, while one node too many only costs a little recall.
  const corpusSizeEstimate = Math.ceil(Math.sqrt(documentCount));
  const topicDiversityEstimate = categoryCount;

  return Math.min(
    maxUsefulNodeCount,
    Math.max(minUsefulNodeCount, corpusSizeEstimate, topicDiversityEstimate),
  );
}

/**
 * Auto-builds an ontology (nodes + edges) from document sources.
 *
 * Pipeline:
 *   1. Collect docs from all sources
 *   2. Parallel batch: extract topic categories per batch (Haiku)
 *   3. Cluster categories into N nodes with level/parentId (LLM)
 *   4. Infer edges between nodes (LLM)
 */
/** Documents read for topic discovery; enough to see every source's themes. */
const MAX_CATEGORY_SAMPLE = 300;
/** Model calls in flight during discovery. */
const MAX_CONCURRENT_CALLS = 3;
/** Categories handed to node design; beyond this the prompt repeats itself. */
const MAX_CATEGORIES = 150;

export class OntologyAutoBuilder {
  constructor(
    private readonly adapter: LLMAdapter,
    private readonly targetNodeCount?: number,
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
    // Why sample: topics of a corpus show in a few hundred documents drawn from
    // every source; reading all of an organization's files here would cost
    // thousands of model calls before a single node exists. Classification
    // later covers every document, in batches.
    const sample = stratifiedSample(docs, MAX_CATEGORY_SAMPLE, (doc) => doc.metadata.source ?? "");
    const batches: SourceDocument[][] = [];
    for (let i = 0; i < sample.length; i += this.batchSize) {
      batches.push(sample.slice(i, i + this.batchSize));
    }
    const results = await mapWithConcurrency(batches, MAX_CONCURRENT_CALLS, (b) =>
      this.extractBatchCategories(b),
    );
    const frequency = new Map<string, number>();
    for (const category of results.flat()) {
      frequency.set(category, (frequency.get(category) ?? 0) + 1);
    }
    // Most common first, so the prompt cap below keeps the topics the corpus repeats.
    return Array.from(frequency.entries())
      .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
      .map(([category]) => category);
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
      const clean = response
        .trim()
        .replace(/^```json/, "")
        .replace(/```$/, "")
        .trim();
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
    const targetNodeCount =
      this.targetNodeCount === undefined
        ? inferTargetNodeCount(docs.length, rawCategories.length)
        : assertValidTargetNodeCount(this.targetNodeCount);
    const docTitles = docs
      .slice(0, 100)
      .map((d) => `- ${d.title}`)
      .join("\n");
    // Why: the node count is inferred from every topic found, but the prompt lists
    // only the most frequent — past this many the list repeats itself.
    const catList = rawCategories.slice(0, MAX_CATEGORIES).join(", ");

    const response = await this.adapter.complete(
      this.templates.nodeDesign(targetNodeCount),
      `Documents:\n${docTitles}\n\nExtracted categories: ${catList}`,
      "Design ontology nodes.",
    );

    try {
      const clean = response
        .trim()
        .replace(/^```json/, "")
        .replace(/```$/, "")
        .trim();
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
          parentId: parentId && parentId !== "null" && parentId !== null ? String(parentId) : null,
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
      const clean = response
        .trim()
        .replace(/^```json/, "")
        .replace(/```$/, "")
        .trim();
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
