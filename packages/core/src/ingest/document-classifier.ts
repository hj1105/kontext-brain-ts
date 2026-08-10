import type { DataSource } from "../graph/layered-models.js";
import type { OntologyNode } from "../graph/ontology-node.js";
import type { OntologyProposalDraft } from "../knowledge/ontology-proposals.js";
import type { LLMAdapter } from "../query/llm-adapter.js";
import type { PromptTemplates } from "../query/prompt-templates.js";
import { DefaultPromptTemplates } from "../query/prompt-templates.js";

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
export class DocumentClassifier {
  constructor(
    private readonly adapter: LLMAdapter,
    private readonly templates: PromptTemplates = DefaultPromptTemplates,
  ) {}

  async classify(
    documents: readonly MCPResourceInfo[],
    existingNodes: ReadonlyMap<string, OntologyNode>,
  ): Promise<ClassificationResult> {
    if (documents.length === 0) return emptyClassification;

    const nodeList = Array.from(existingNodes.entries())
      .map(([id, n]) => `${id}: ${n.description}`)
      .join("\n");

    const docList = documents
      .map((d, i) => {
        const desc = d.description.trim() ? ` — ${d.description}` : "";
        return `[${i}] ${d.title}${desc}`;
      })
      .join("\n");

    const response = await this.adapter.complete(
      this.templates.documentClassification,
      `Ontology nodes:\n${nodeList}\n\nDocuments:\n${docList}`,
      "Classify each document into the best matching node.",
    );

    const { indexMappings, unmappedIndices } = parseClassification(
      response,
      documents.length,
      new Set(existingNodes.keys()),
    );

    const docMappings = new Map<string, MCPResourceInfo[]>();
    for (const [nodeId, indices] of indexMappings) {
      docMappings.set(nodeId, indices.map((i) => documents[i]).filter(isDefined));
    }

    const unmappedDocs = unmappedIndices.map((i) => documents[i]).filter(isDefined);
    let proposals: OntologyProposalDraft[] = [];
    if (unmappedDocs.length > 0) {
      proposals = await this.proposeNewNodes(unmappedDocs);
    }

    const mappedDocs = new Set<MCPResourceInfo>();
    for (const list of docMappings.values()) for (const d of list) mappedDocs.add(d);
    const unmapped = unmappedDocs.filter((d) => !mappedDocs.has(d));

    return { mappings: docMappings, newNodes: [], unmapped, proposals };
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
