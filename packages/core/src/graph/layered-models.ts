import type { Edge, OntologyNode } from "./ontology-node.js";

// ── Pipeline Steps ────────────────────────────────────────────

export enum DepthType {
  ONTOLOGY = "ONTOLOGY",
  META = "META",
  VECTOR = "VECTOR",
  CONTENT = "CONTENT",
  SECTION = "SECTION",
  CHUNK = "CHUNK",
}

export interface PipelineStep {
  readonly depth: number;
  readonly type: DepthType;
  readonly maxSelect: number;
  readonly sectionKey?: string | null;
  readonly fetchFull: boolean;
  readonly threshold: number;
}

export function step(
  init: Partial<PipelineStep> & { depth: number; type: DepthType },
): PipelineStep {
  return {
    depth: init.depth,
    type: init.type,
    maxSelect: init.maxSelect ?? 5,
    sectionKey: init.sectionKey ?? null,
    fetchFull: init.fetchFull ?? false,
    threshold: init.threshold ?? 0.0,
  };
}

export const DEFAULT_PIPELINE: readonly PipelineStep[] = [
  step({ depth: 0, type: DepthType.ONTOLOGY, maxSelect: 5 }),
  step({ depth: 1, type: DepthType.META, maxSelect: 10 }),
  step({ depth: 2, type: DepthType.CONTENT, maxSelect: 5 }),
];

export const VECTOR_PIPELINE: readonly PipelineStep[] = [
  step({ depth: 0, type: DepthType.ONTOLOGY, maxSelect: 5 }),
  step({ depth: 1, type: DepthType.VECTOR, maxSelect: 20, threshold: 0.7 }),
  step({ depth: 2, type: DepthType.META, maxSelect: 5 }),
  step({ depth: 3, type: DepthType.CONTENT, maxSelect: 5 }),
];

export const N_LAYER_PIPELINE: readonly PipelineStep[] = [
  step({ depth: 0, type: DepthType.ONTOLOGY, maxSelect: 3 }),
  step({ depth: 1, type: DepthType.ONTOLOGY, maxSelect: 5 }),
  step({ depth: 2, type: DepthType.CHUNK, maxSelect: 3 }),
];

/**
 * Per-node pipeline: at every traversed node, run META then CONTENT back-to-back.
 * Works correctly even when the L1 mapping resolves to a leaf node (unlike
 * DEFAULT_PIPELINE, which only dispatches one step per depth).
 */
export const PERNODE_PIPELINE: readonly PipelineStep[] = [
  step({ depth: 0, type: DepthType.META, maxSelect: 3 }),
  step({ depth: 0, type: DepthType.CONTENT, maxSelect: 3 }),
];

// ── Data Sources ──────────────────────────────────────────────

export enum DataSource {
  NOTION = "NOTION",
  JIRA = "JIRA",
  GITHUB_PR = "GITHUB_PR",
  GITHUB_ISSUE = "GITHUB_ISSUE",
  SLACK = "SLACK",
  GDRIVE = "GDRIVE",
  EMAIL = "EMAIL",
  CUSTOM = "CUSTOM",
}

export interface MetaDocument {
  readonly id: string;
  readonly title: string;
  readonly source: DataSource;
  readonly ontologyNodeId: string;
  readonly url?: string | null;
  readonly score: number;
  readonly metadata: Readonly<Record<string, string>>;
  readonly fetchedAt: Date;
}

export function createMetaDocument(
  init: Partial<MetaDocument> & {
    id: string;
    title: string;
    source: DataSource;
    ontologyNodeId: string;
  },
): MetaDocument {
  return {
    id: init.id,
    title: init.title,
    source: init.source,
    ontologyNodeId: init.ontologyNodeId,
    url: init.url ?? null,
    score: init.score ?? 1.0,
    metadata: init.metadata ?? {},
    fetchedAt: init.fetchedAt ?? new Date(),
  };
}

export interface DocumentContent {
  readonly metaDocumentId: string;
  readonly title: string;
  readonly body: string;
  readonly source: DataSource;
  readonly sectionContent?: string | null;
  readonly fetchedAt: Date;
}

// ── Traversal Result ──────────────────────────────────────────

export interface TraversedNode {
  readonly node: OntologyNode;
  readonly depth: number;
  readonly cumulativeWeight: number;
}

export interface TraversalResult {
  readonly nodes: readonly TraversedNode[];
  readonly path: readonly Edge[];
}

// ── Query Result ──────────────────────────────────────────────

export interface LayeredQueryResult {
  readonly answer: string;
  readonly usedOntologyNodes: readonly OntologyNode[];
  readonly selectedMetaDocs: readonly MetaDocument[];
  readonly fetchedContents: readonly DocumentContent[];
  readonly contextTokensUsed: number;
  readonly traversalPath: readonly Edge[];
  readonly pipelineSteps: readonly PipelineStep[];
}
