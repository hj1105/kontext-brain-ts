import { createHash } from "node:crypto";
import type {
  CodeFileAnalysis,
  CodeResourceSnapshot,
  CodeResourceSyncPort,
  CodeResourceSyncResult,
  CodeSnapshotNormalizationInput,
  CodeSyncInput,
  LanguageCodeProvider,
} from "./domain.js";

export class CodeResourceSnapshotAdapter {
  normalize(input: CodeSnapshotNormalizationInput): CodeResourceSnapshot {
    const { analysis } = input;
    const symbols = new Map(analysis.symbols.map((symbol) => [symbol.symbolId, symbol]));
    return {
      organizationId: input.organizationId,
      source: {
        connectorId: "code",
        externalId: `${analysis.codebaseId}:${analysis.relativePath}`,
        type: `${analysis.language}-module`,
      },
      title: analysis.relativePath,
      contentHash: analysis.contentHash,
      body: analysis.sourceText,
      acl: input.acl,
      ontologyNodeIds: input.ontologyNodeIds,
      chunks: analysis.symbols.map((symbol) => ({
        id: symbol.sourceChunkId,
        contentHash: symbol.contentHash,
        text: symbol.text,
        position: symbol.position,
        ontologyNodeIds: input.ontologyNodeIds,
        acl: input.acl,
      })),
      entities: analysis.symbols.map((symbol) => ({
        entityId: symbol.symbolId,
        scope: symbol.exported ? ("global" as const) : ("resource" as const),
        name: symbol.identity.qualifiedName,
        type: `code:${symbol.identity.kind}`,
        mentionChunkIds: [symbol.sourceChunkId],
        promotionEvidence: symbol.exported ? ("deterministic" as const) : undefined,
      })),
      facts: analysis.relationships.map((relationship) => {
        const subject = symbols.get(relationship.subjectSymbolId);
        const evidence = symbols.get(relationship.evidenceSymbolId);
        if (!subject || !evidence) {
          throw new Error(
            `Relationship ${relationship.relationshipId} references an unknown symbol`,
          );
        }
        return {
          factKey: relationship.relationshipId,
          subject: {
            entityId: subject.symbolId,
            scope: subject.exported ? ("global" as const) : ("resource" as const),
          },
          predicate: relationship.predicate,
          object:
            relationship.object.kind === "symbol"
              ? {
                  kind: "entity" as const,
                  entity: {
                    entityId: relationship.object.symbolId,
                    scope: relationship.object.entityScope,
                  },
                }
              : { kind: "literal" as const, value: relationship.object.value },
          evidenceChunkIds: [evidence.sourceChunkId],
          singleValue: relationship.predicate === "returns",
        };
      }),
    };
  }
}

export class CodeKnowledgeSynchronizer {
  constructor(
    private readonly resourceSync: CodeResourceSyncPort,
    private readonly provider: LanguageCodeProvider,
    private readonly adapter: CodeResourceSnapshotAdapter,
  ) {}

  async sync(input: CodeSyncInput): Promise<CodeResourceSyncResult> {
    const analysis = this.provider.analyze(input);
    const snapshot = this.adapter.normalize({
      organizationId: input.organizationId,
      analysis,
      acl: input.acl,
      ontologyNodeIds: input.ontologyNodeIds,
    });
    return this.resourceSync.execute(snapshot);
  }

  remove(organizationId: string, codebaseId: string, relativePath: string): Promise<boolean> {
    return this.resourceSync.remove(organizationId, {
      connectorId: "code",
      externalId: `${codebaseId}:${relativePath}`,
      type: `${this.provider.language}-module`,
    });
  }
}

export function codeSnapshotDigest(snapshot: CodeResourceSnapshot): string {
  return createHash("sha256").update(JSON.stringify(snapshot)).digest("hex");
}
