import type { DataSource } from "../graph/layered-models.js";
import type { OntologyNode } from "../graph/ontology-node.js";
import type { OntologyProposalDraft } from "../knowledge/ontology-proposals.js";
import type { LLMAdapter } from "../query/llm-adapter.js";
import type { PromptTemplates } from "../query/prompt-templates.js";
import { DefaultPromptTemplates } from "../query/prompt-templates.js";
import { mapWithConcurrency } from "./bounded-concurrency.js";

export interface MCPResourceInfo {
  readonly id: string;
  readonly title: string;
  readonly description: string;
  readonly source: DataSource;
  readonly connectorName: string;
}

export interface ClassificationResult {
  readonly mappings: ReadonlyMap<string, readonly MCPResourceInfo[]>;
  readonly newNodes: readonly OntologyNode[];
  readonly unmapped: readonly MCPResourceInfo[];
  readonly proposals: readonly OntologyProposalDraft[];
}

export const emptyClassification: ClassificationResult = {
  mappings: new Map(),
  newNodes: [],
  unmapped: [],
  proposals: [],
};

/**
 * Classifies documents into existing ontology nodes via LLM.
 * Documents that do not fit remain unmapped and produce reviewable proposals.
 */
export interface DocumentClassifierOptions {
  /** Documents per model call. One call over a whole organization's code never returns in time. */
  readonly maxDocumentsPerBatch?: number;
  /** Calls in flight at once; the model answers each batch independently. */
  readonly concurrency?: number;
  /** Description text per document in the prompt; a barrel's export list says little past this. */
  readonly maxDescriptionChars?: number;
  /** Called after each classification batch with batches done and total. */
  readonly onProgress?: (done: number, total: number) => void;
}

const DEFAULT_BATCH = 100;
const DEFAULT_CONCURRENCY = 4;
const DEFAULT_DESCRIPTION_CHARS = 200;

export class DocumentClassifier {
  private readonly batchSize: number;
  private readonly concurrency: number;
  private readonly descriptionChars: number;
  private readonly onProgress: ((done: number, total: number) => void) | undefined;

  constructor(
    private readonly adapter: LLMAdapter,
    private readonly templates: PromptTemplates = DefaultPromptTemplates,
    options: DocumentClassifierOptions = {},
  ) {
    this.batchSize = Math.max(1, options.maxDocumentsPerBatch ?? DEFAULT_BATCH);
    this.concurrency = Math.max(1, options.concurrency ?? DEFAULT_CONCURRENCY);
    this.descriptionChars = Math.max(0, options.maxDescriptionChars ?? DEFAULT_DESCRIPTION_CHARS);
    this.onProgress = options.onProgress;
  }

  async classify(
    documents: readonly MCPResourceInfo[],
    existingNodes: ReadonlyMap<string, OntologyNode>,
  ): Promise<ClassificationResult> {
    if (documents.length === 0) return emptyClassification;

    const nodeList = Array.from(existingNodes.entries())
      .map(([id, n]) => `${id}: ${n.description}`)
      .join("\n");
    const validNodeIds = new Set(existingNodes.keys());

    // Why batches: the prompt grows with every document and a single call over an
    // organization's code exceeded the model's time budget; each batch is answered
    // on its own and the results merge by document identity.
    const batches: MCPResourceInfo[][] = [];
    for (let start = 0; start < documents.length; start += this.batchSize) {
      batches.push(documents.slice(start, start + this.batchSize));
    }
    let completed = 0;
    this.onProgress?.(0, batches.length);
    const batchResults = await mapWithConcurrency(batches, this.concurrency, async (batch) => {
      const response = await this.adapter.complete(
        this.templates.documentClassification,
        `Ontology nodes:\n${nodeList}\n\nDocuments:\n${this.describe(batch)}`,
        "Classify each document into the best matching node.",
      );
      completed += 1;
      this.onProgress?.(completed, batches.length);
      return parseClassification(response, batch.length, validNodeIds);
    });

    const docMappings = new Map<string, MCPResourceInfo[]>();
    const unmappedDocs: MCPResourceInfo[] = [];
    batchResults.forEach(({ indexMappings, unmappedIndices }, batchIndex) => {
      const batch = batches[batchIndex] ?? [];
      for (const [nodeId, indices] of indexMappings) {
        const list = docMappings.get(nodeId) ?? [];
        list.push(...indices.map((i) => batch[i]).filter(isDefined));
        docMappings.set(nodeId, list);
      }
      unmappedDocs.push(...unmappedIndices.map((i) => batch[i]).filter(isDefined));
    });

    let proposals: OntologyProposalDraft[] = [];
    if (unmappedDocs.length > 0) {
      const proposalBatches: MCPResourceInfo[][] = [];
      for (let start = 0; start < unmappedDocs.length; start += this.batchSize) {
        proposalBatches.push(unmappedDocs.slice(start, start + this.batchSize));
      }
      proposals = (
        await mapWithConcurrency(proposalBatches, this.concurrency, (batch) =>
          this.proposeNewNodes(batch),
        )
      ).flat();
    }

    const mappedDocs = new Set<MCPResourceInfo>();
    for (const list of docMappings.values()) for (const d of list) mappedDocs.add(d);
    const unmapped = unmappedDocs.filter((d) => !mappedDocs.has(d));

    return { mappings: docMappings, newNodes: [], unmapped, proposals };
  }

  private describe(documents: readonly MCPResourceInfo[]): string {
    return documents
      .map((d, i) => {
        const description = d.description.trim().slice(0, this.descriptionChars);
        const desc = description ? ` — ${description}` : "";
        return `[${i}] ${d.title}${desc}`;
      })
      .join("\n");
  }

  private async proposeNewNodes(
    unmappedDocs: readonly MCPResourceInfo[],
  ): Promise<OntologyProposalDraft[]> {
    const docList = unmappedDocs
      .map((d, i) => {
        const desc = d.description.trim() ? ` — ${d.description}` : "";
        return `[${i}] ${d.title}${desc}`;
      })
      .join("\n");

    const response = await this.adapter.complete(
      this.templates.nodeExpansion,
      docList,
      "Create new ontology nodes for these uncategorized documents.",
    );

    try {
      const clean = response
        .trim()
        .replace(/^```json/, "")
        .replace(/```$/, "")
        .trim();
      const parsed = JSON.parse(clean) as {
        nodes?: Array<{ id: string; description?: string; weight?: number }>;
        mappings?: Record<string, number[]>;
      };
      const mappings = parsed.mappings ?? {};
      return (parsed.nodes ?? []).map((node) => ({
        suggestedNodeId: node.id,
        description: node.description ?? "",
        resourceIds: (mappings[node.id] ?? []).flatMap((index) => {
          const resource = unmappedDocs[index];
          return resource
            ? [`${encodeURIComponent(resource.connectorName)}:${encodeURIComponent(resource.id)}`]
            : [];
        }),
      }));
    } catch {
      return [];
    }
  }
}

function isDefined<T>(value: T | undefined): value is T {
  return value !== undefined;
}

function parseClassification(
  response: string,
  docCount: number,
  validNodeIds: ReadonlySet<string>,
): { indexMappings: Map<string, number[]>; unmappedIndices: number[] } {
  try {
    const clean = response
      .trim()
      .replace(/^```json/, "")
      .replace(/```$/, "")
      .trim();
    const parsed = JSON.parse(clean) as {
      mappings?: Record<string, number[]>;
      unmapped?: number[];
    };

    if (!parsed.mappings) {
      return {
        indexMappings: new Map(),
        unmappedIndices: Array.from({ length: docCount }, (_, i) => i),
      };
    }

    const mapped = new Set<number>();
    const result = new Map<string, number[]>();
    for (const [nodeId, indices] of Object.entries(parsed.mappings)) {
      if (!validNodeIds.has(nodeId)) continue;
      const filtered = indices.filter((i) => Number.isInteger(i) && i >= 0 && i < docCount);
      result.set(nodeId, filtered);
      for (const i of filtered) mapped.add(i);
    }

    const unmappedSet = new Set<number>(parsed.unmapped ?? []);
    for (let i = 0; i < docCount; i++) if (!mapped.has(i)) unmappedSet.add(i);

    return {
      indexMappings: result,
      unmappedIndices: Array.from(unmappedSet)
        .filter((i) => i < docCount)
        .sort((a, b) => a - b),
    };
  } catch {
    return {
      indexMappings: new Map(),
      unmappedIndices: Array.from({ length: docCount }, (_, i) => i),
    };
  }
}
