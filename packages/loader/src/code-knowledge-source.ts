import type { CodeResourceSyncPort } from "@kontext-brain/code";
import type { MCPConnector } from "@kontext-brain/mcp";

/**
 * A source that can project its code into the knowledge graph at symbol level:
 * one Resource per file, one Chunk per Code Symbol, symbols as Entities and
 * their calls/imports/returns as Facts. The module documents the classifier
 * placed on ontology nodes decide which nodes those symbols inherit, which is
 * the design's Code Symbol → Ontology Node linkage.
 */
export interface CodeKnowledgeSyncInput {
  readonly organizationId: string;
  readonly resourceSync: CodeResourceSyncPort;
  /** Ontology nodes the classifier assigned to a module document, by module id. */
  readonly nodeIdsFor: (moduleId: string) => readonly string[];
}

export interface CodeKnowledgeSyncReport {
  readonly filesSynced: number;
  readonly filesFailed: number;
}

export interface CodeKnowledgeSource {
  syncCodeKnowledge(input: CodeKnowledgeSyncInput): Promise<CodeKnowledgeSyncReport>;
}

export function isCodeKnowledgeSource(
  connector: MCPConnector,
): connector is MCPConnector & CodeKnowledgeSource {
  return typeof (connector as Partial<CodeKnowledgeSource>).syncCodeKnowledge === "function";
}
