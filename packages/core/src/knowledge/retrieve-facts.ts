import type { FactRecord, Principal } from "./domain.js";
import type { AuthorizedKnowledgeGraphReader, ResourceContentStore } from "./ports.js";

export interface AccessibleEvidence {
  readonly evidenceId: string;
  readonly resourceId: string;
  readonly chunkId: string;
  readonly text: string;
}

export interface AccessibleFact {
  readonly fact: FactRecord;
  readonly evidence: readonly AccessibleEvidence[];
}

export interface RetrieveFactsInput {
  readonly principal: Principal;
  readonly factKeys?: readonly string[];
}

export class RetrieveFactsUseCase {
  constructor(
    private readonly reader: AuthorizedKnowledgeGraphReader,
    private readonly contentStore: ResourceContentStore,
  ) {}

  async execute(input: RetrieveFactsInput): Promise<readonly AccessibleFact[]> {
    const rows = await this.reader.listAccessibleFactEvidence(input.principal, input.factKeys);
    const visible = new Map<string, AccessibleFact>();
    for (const row of rows) {
      // Object content is resolved only after the authorized reader returns metadata.
      const content = await this.contentStore.get(row.chunk.contentObjectKey);
      const text = content?.chunks[row.chunk.sourceChunkId];
      if (text === undefined) continue;
      const item = visible.get(row.fact.factKey) ?? { fact: row.fact, evidence: [] };
      (item.evidence as AccessibleEvidence[]).push({
        evidenceId: row.evidence.evidenceId,
        resourceId: row.evidence.resourceId,
        chunkId: row.evidence.chunkId,
        text,
      });
      visible.set(row.fact.factKey, item);
    }
    return Array.from(visible.values());
  }
}
