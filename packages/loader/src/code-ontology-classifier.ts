import {
  type ClassificationResult,
  DefaultPromptTemplates,
  DocumentClassifier,
  type LLMAdapter,
  type MCPResourceInfo,
  type OntologyNode,
  type PromptTemplates,
} from "@kontext-brain/core";

/**
 * Assigns code files to Ontology Nodes with the same classifier documents use.
 *
 * CodeKnowledgeSynchronizer already accepts ontologyNodeIds, but nothing decided
 * what they should be — every caller, including the integration test, passed a
 * literal. Hand-assigning them defeats the purpose: the point of the link is
 * that the system knows which approved decisions govern a Code Symbol, and a
 * caller that already knows the answer does not need the link.
 *
 * Running code through DocumentClassifier keeps one classification path for the
 * whole Codebase, so a file and the specification that governs it land on the
 * same node for the same reason.
 */
export interface CodeFileForClassification {
  readonly relativePath: string;
  /** Behaviour-bearing symbol names, which carry most of the file's meaning. */
  readonly symbolNames: readonly string[];
}

export interface CodeOntologyAssignment {
  readonly relativePath: string;
  readonly ontologyNodeIds: readonly string[];
}

export interface CodeOntologyClassification {
  readonly assignments: readonly CodeOntologyAssignment[];
  readonly unassigned: readonly string[];
}

export class CodeOntologyClassifier {
  private readonly classifier: DocumentClassifier;

  constructor(adapter: LLMAdapter, templates: PromptTemplates = DefaultPromptTemplates) {
    this.classifier = new DocumentClassifier(adapter, templates);
  }

  async classify(
    files: readonly CodeFileForClassification[],
    nodes: ReadonlyMap<string, OntologyNode>,
  ): Promise<CodeOntologyClassification> {
    if (files.length === 0 || nodes.size === 0) {
      return { assignments: [], unassigned: files.map((file) => file.relativePath) };
    }
    const result = await this.classifier.classify(files.map(asResource), nodes);
    return toAssignments(files, result);
  }
}

/**
 * A code file is described by its path and its exported behaviour, which is what
 * a classifier can actually reason about. Including the whole body would drown
 * the signal in boilerplate that is identical across subsystems.
 */
function asResource(file: CodeFileForClassification): MCPResourceInfo {
  return {
    id: file.relativePath,
    title: file.relativePath,
    description: file.symbolNames.join(", "),
    source: "CUSTOM" as MCPResourceInfo["source"],
    connectorName: "code",
  };
}

function toAssignments(
  files: readonly CodeFileForClassification[],
  result: ClassificationResult,
): CodeOntologyClassification {
  const byPath = new Map<string, string[]>();
  for (const [nodeId, resources] of result.mappings) {
    for (const resource of resources) {
      const existing = byPath.get(resource.id);
      if (existing) existing.push(nodeId);
      else byPath.set(resource.id, [nodeId]);
    }
  }
  const assignments: CodeOntologyAssignment[] = [];
  const unassigned: string[] = [];
  for (const file of files) {
    const nodeIds = byPath.get(file.relativePath);
    if (nodeIds === undefined || nodeIds.length === 0) {
      unassigned.push(file.relativePath);
      continue;
    }
    assignments.push({
      relativePath: file.relativePath,
      ontologyNodeIds: [...new Set(nodeIds)].sort(),
    });
  }
  return { assignments, unassigned };
}
