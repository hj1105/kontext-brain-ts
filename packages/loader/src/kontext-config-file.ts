import { readFileSync, writeFileSync } from "node:fs";
import type { Edge, OntologyNode } from "@kontext-brain/core";
import { type Document, isMap, parseDocument, stringify } from "yaml";
import {
  type EmbeddingConfigDto,
  EmbeddingConfigSchema,
  type MCPConfigDto,
} from "./kontext-config.js";

/**
 * Read/modify/write access to `kontext.yaml`. Setup writes generated content
 * back into the file the user already owns, so edits must preserve every key
 * this module does not explicitly manage.
 */

export interface KontextConfigDocument {
  readonly path: string;
  readonly data: Record<string, unknown>;
  /** Parsed source, kept so a write preserves the user's comments and layout. */
  readonly source?: Document;
}

export function readConfigDocument(path: string): KontextConfigDocument {
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch {
    return { path, data: {} };
  }
  const source = parseDocument(text);
  const parsed = source.toJS() as unknown;
  if (parsed === null || parsed === undefined) return { path, data: {}, source };
  if (typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${path}: expected a YAML mapping at the top level`);
  }
  return { path, data: { ...(parsed as Record<string, unknown>) }, source };
}

export function writeConfigDocument(document: KontextConfigDocument): void {
  // Why: descriptions come from a model and routinely contain ':' and quotes, so
  // the document is serialized by the YAML writer rather than string-concatenated.
  // Editing the parsed source in place keeps the comments a hand-maintained
  // kontext.yaml carries; a plain object round-trip deletes every one of them.
  const source = document.source;
  if (source && isMap(source.contents)) {
    const managed = new Set(Object.keys(document.data));
    const current = source.toJS() as Record<string, unknown> | null;
    for (const key of managed) {
      // Why: replacing an untouched section discards the comments inside it, so only
      // a section the caller actually changed is rewritten.
      const before = JSON.stringify(current?.[key] ?? null);
      const after = JSON.stringify(document.data[key] ?? null);
      if (before !== after) source.set(key, document.data[key]);
    }
    for (const item of [...source.contents.items]) {
      const key = String((item.key as { value?: unknown })?.value ?? "");
      if (key !== "" && !managed.has(key)) source.delete(key);
    }
    writeFileSync(document.path, source.toString({ lineWidth: 0 }), "utf8");
    return;
  }
  writeFileSync(document.path, stringify(document.data, { lineWidth: 0 }), "utf8");
}

/**
 * The model used when kontext.yaml names none: the user's logged-in Codex CLI,
 * which a ChatGPT subscription already covers. A host that only adds sources
 * writes a file without `llm`, and setup must still have something to run.
 */
export const DEFAULT_LLM_CONFIG = {
  traversal: { provider: "codex", model: "codex" },
  reasoning: { provider: "codex", model: "codex" },
} as const;

export function withDefaultLlm(document: KontextConfigDocument): KontextConfigDocument {
  if (document.data.llm !== undefined && document.data.llm !== null) return document;
  return { ...document, data: { ...document.data, llm: structuredClone(DEFAULT_LLM_CONFIG) } };
}

/** The `embedding:` section as written, or undefined when the file leaves it to the default. */
export function readEmbeddingConfig(
  document: KontextConfigDocument,
): EmbeddingConfigDto | undefined {
  const raw = document.data.embedding;
  if (typeof raw !== "object" || raw === null) return undefined;
  const parsed = EmbeddingConfigSchema.safeParse(raw);
  return parsed.success ? parsed.data : undefined;
}

export function withEmbedding(
  document: KontextConfigDocument,
  embedding: EmbeddingConfigDto,
): KontextConfigDocument {
  return { ...document, data: { ...document.data, embedding: { ...embedding } } };
}

export function readMCPEntries(document: KontextConfigDocument): MCPConfigDto[] {
  const raw = document.data.mcp;
  if (!Array.isArray(raw)) return [];
  return raw.filter(
    (entry): entry is MCPConfigDto =>
      typeof entry === "object" &&
      entry !== null &&
      typeof (entry as { name?: unknown }).name === "string",
  );
}

export function withMCPEntries(
  document: KontextConfigDocument,
  entries: readonly MCPConfigDto[],
): KontextConfigDocument {
  return { ...document, data: { ...document.data, mcp: [...entries] } };
}

export interface OntologyNodeYaml {
  id: string;
  description: string;
  weight: number;
  parentId?: string;
  level?: number;
  relates?: Array<{ to: string; weight: number; type?: string }>;
  mcpSource?: string;
  webSearch?: boolean;
  nodeType?: string;
  keywords?: string[];
}

export function toOntologyYamlNodes(
  nodes: readonly OntologyNode[],
  edges: readonly Edge[],
): OntologyNodeYaml[] {
  const edgesByFrom = new Map<string, Edge[]>();
  for (const edge of edges) {
    const list = edgesByFrom.get(edge.from) ?? [];
    list.push(edge);
    edgesByFrom.set(edge.from, list);
  }
  return nodes.map((node) => {
    const related = (edgesByFrom.get(node.id) ?? [])
      .slice()
      .sort((left, right) => right.weight - left.weight)
      .map((edge) => ({
        to: edge.to,
        weight: edge.weight,
        ...(edge.type ? { type: edge.type } : {}),
      }));
    return {
      id: node.id,
      description: node.description,
      weight: node.weight,
      ...(node.parentId ? { parentId: node.parentId } : {}),
      ...(node.level > 0 ? { level: node.level } : {}),
      ...(related.length > 0 ? { relates: related } : {}),
      // Why: a rebuild replaces the whole ontology array, so anything the user
      // authored by hand that the graph still carries has to be written back or
      // it is silently reset to its default on the next load.
      ...(node.mcpSource ? { mcpSource: node.mcpSource } : {}),
      ...(node.webSearch ? { webSearch: true } : {}),
      ...(node.nodeType && node.nodeType !== "DOMAIN" ? { nodeType: String(node.nodeType) } : {}),
      ...(node.keywords && node.keywords.length > 0 ? { keywords: [...node.keywords] } : {}),
    };
  });
}

export function withOntology(
  document: KontextConfigDocument,
  nodes: readonly OntologyNodeYaml[],
): KontextConfigDocument {
  return { ...document, data: { ...document.data, ontology: [...nodes] } };
}

export function readOntologyNodeIds(document: KontextConfigDocument): string[] {
  const raw = document.data.ontology;
  if (!Array.isArray(raw)) return [];
  return raw
    .map((entry) =>
      typeof entry === "object" && entry !== null ? (entry as { id?: unknown }).id : undefined,
    )
    .filter((id): id is string => typeof id === "string");
}
