import type { OntologyNode } from "../graph/ontology-node.js";
import type { LLMAdapter } from "./llm-adapter.js";
import type { PromptTemplates } from "./prompt-templates.js";
import { DefaultPromptTemplates } from "./prompt-templates.js";
import type { VectorStore } from "./vector-store.js";

/** Query -> start node mapping strategy port. */
export interface NodeMappingStrategy {
  readonly strategyName: string;
  findStartNodes(query: string, nodes: ReadonlyMap<string, OntologyNode>): Promise<string[]>;
}

/** Keyword-based matching (no LLM). */
export class KeywordMappingStrategy implements NodeMappingStrategy {
  readonly strategyName = "keyword";

  async findStartNodes(query: string, nodes: ReadonlyMap<string, OntologyNode>): Promise<string[]> {
    const q = query.toLowerCase();
    const matches: Array<{ id: string; weight: number }> = [];
    for (const [id, node] of nodes) {
      const idMatch = q.includes(id.toLowerCase());
      const descMatch = node.description
        .split(/\s+/)
        .some((w) => w.length > 0 && q.includes(w.toLowerCase()));
      if (idMatch || descMatch) matches.push({ id, weight: node.weight });
    }
    matches.sort((a, b) => b.weight - a.weight);
    if (matches.length > 0) return matches.map((m) => m.id);

    // Fallback: highest weighted node
    let top: { id: string; weight: number } | null = null;
    for (const [id, node] of nodes) {
      if (!top || node.weight > top.weight) top = { id, weight: node.weight };
    }
    return top ? [top.id] : [];
  }
}

/** Vector similarity-based matching. */
export class VectorMappingStrategy implements NodeMappingStrategy {
  readonly strategyName = "vector";

  constructor(private readonly vectorStore: VectorStore) {}

  async findStartNodes(query: string, nodes: ReadonlyMap<string, OntologyNode>): Promise<string[]> {
    const matches = await this.vectorStore.similaritySearch(query, 3);
    const valid = matches.filter((id) => nodes.has(id));
    if (valid.length > 0) return valid;
    return new KeywordMappingStrategy().findStartNodes(query, nodes);
  }
}

/** LLM-based matching with compressed prompt. */
export class LLMMappingStrategy implements NodeMappingStrategy {
  readonly strategyName = "llm";

  constructor(
    private readonly adapter: LLMAdapter,
    private readonly templates: PromptTemplates = DefaultPromptTemplates,
  ) {}

  async findStartNodes(query: string, nodes: ReadonlyMap<string, OntologyNode>): Promise<string[]> {
    const nodeDescs = Array.from(nodes.entries())
      .map(([id, n]) => `${id}:${n.description}`)
      .join("\n");

    const response = await this.adapter.complete(this.templates.nodeClassifier, nodeDescs, query);
    const parsed = parseNodeIds(response, new Set(nodes.keys()));
    if (parsed.length === 0) {
      return new KeywordMappingStrategy().findStartNodes(query, nodes);
    }
    return parsed;
  }
}

function parseNodeIds(response: string, validIds: Set<string>): string[] {
  const clean = response.trim().replace(/^```json/, "").replace(/```$/, "").trim();
  try {
    const parsed = JSON.parse(clean);
    if (Array.isArray(parsed)) {
      const out = parsed
        .map((v) => String(v))
        .filter((s) => validIds.has(s));
      if (out.length > 0) return out;
    }
  } catch {
    // fall through
  }
  return clean
    .split(",")
    .map((s) => s.trim().replace(/^"|"$/g, ""))
    .filter((s) => validIds.has(s));
}

export class NodeMappingRegistry {
  private readonly strategies = new Map<string, NodeMappingStrategy>();

  constructor() {
    this.register(new KeywordMappingStrategy());
  }

  register(strategy: NodeMappingStrategy): void {
    this.strategies.set(strategy.strategyName, strategy);
  }

  resolve(name: string): NodeMappingStrategy {
    const s = this.strategies.get(name);
    if (!s) {
      throw new Error(
        `Unsupported mapping strategy: '${name}'. Registered: ${Array.from(this.strategies.keys()).join(",")}`,
      );
    }
    return s;
  }
}
