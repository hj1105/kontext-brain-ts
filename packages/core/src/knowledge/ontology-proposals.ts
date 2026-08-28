export interface OntologyProposalDraft {
  readonly suggestedNodeId: string;
  readonly description: string;
  readonly resourceIds: readonly string[];
}

export interface OntologyProposal extends OntologyProposalDraft {
  readonly organizationId: string;
  readonly proposalKey: string;
  readonly occurrences: number;
  readonly status: "open" | "published" | "accepted" | "rejected";
  readonly updatedAt: string;
}

export interface OntologyProposalQueue {
  enqueue(organizationId: string, proposals: readonly OntologyProposalDraft[]): Promise<void>;
  listOpen(organizationId: string): Promise<readonly OntologyProposal[]>;
  listPending(organizationId: string): Promise<readonly OntologyProposal[]>;
  markPublished(organizationId: string, proposalKeys: readonly string[]): Promise<void>;
}

export interface OntologyYamlUpdater {
  update(yaml: string, proposals: readonly OntologyProposal[]): Promise<string>;
}

export interface OntologyProposalPublisher {
  upsert(input: {
    readonly organizationId: string;
    readonly yaml: string;
    readonly proposals: readonly OntologyProposal[];
  }): Promise<{ readonly url: string }>;
}

export class PublishOntologyProposalsUseCase {
  constructor(
    private readonly queue: OntologyProposalQueue,
    private readonly yamlUpdater: OntologyYamlUpdater,
    private readonly publisher: OntologyProposalPublisher,
  ) {}

  async execute(
    organizationId: string,
    activeYaml: string,
  ): Promise<{ readonly changed: boolean; readonly yaml: string; readonly url?: string }> {
    const open = await this.queue.listOpen(organizationId);
    if (open.length === 0) return { changed: false, yaml: activeYaml };
    const proposals = await this.queue.listPending(organizationId);
    const yaml = await this.yamlUpdater.update(activeYaml, proposals);
    const published = await this.publisher.upsert({ organizationId, yaml, proposals });
    await this.queue.markPublished(
      organizationId,
      proposals.map((proposal) => proposal.proposalKey),
    );
    return { changed: true, yaml, url: published.url };
  }
}

export class InMemoryOntologyProposalQueue implements OntologyProposalQueue {
  private readonly proposals = new Map<string, OntologyProposal>();

  async enqueue(
    organizationId: string,
    proposals: readonly OntologyProposalDraft[],
  ): Promise<void> {
    for (const draft of proposals) {
      const proposalKey = normalizeProposalKey(draft.suggestedNodeId);
      if (!proposalKey) continue;
      const key = `${organizationId}\u0000${proposalKey}`;
      const previous = this.proposals.get(key);
      this.proposals.set(key, {
        organizationId,
        proposalKey,
        suggestedNodeId: draft.suggestedNodeId,
        description: draft.description,
        resourceIds: Array.from(new Set([...(previous?.resourceIds ?? []), ...draft.resourceIds])),
        occurrences: (previous?.occurrences ?? 0) + 1,
        // Preserve an existing lifecycle status (published/accepted/rejected); only
        // brand-new proposals start as "open". Re-observing an unmapped node must not
        // resurrect a proposal that is already published or decided.
        status: previous?.status ?? "open",
        updatedAt: new Date().toISOString(),
      });
    }
  }

  async listOpen(organizationId: string): Promise<readonly OntologyProposal[]> {
    return Array.from(this.proposals.values()).filter(
      (proposal) => proposal.organizationId === organizationId && proposal.status === "open",
    );
  }

  async listPending(organizationId: string): Promise<readonly OntologyProposal[]> {
    return Array.from(this.proposals.values()).filter(
      (proposal) =>
        proposal.organizationId === organizationId &&
        (proposal.status === "open" || proposal.status === "published"),
    );
  }

  async markPublished(organizationId: string, proposalKeys: readonly string[]): Promise<void> {
    for (const proposalKey of proposalKeys) {
      const key = `${organizationId}\u0000${proposalKey}`;
      const proposal = this.proposals.get(key);
      if (proposal?.status === "open") {
        this.proposals.set(key, {
          ...proposal,
          status: "published",
          updatedAt: new Date().toISOString(),
        });
      }
    }
  }
}

/** Canonical proposal-key normalization shared by every OntologyProposalQueue backend. */
export function normalizeProposalKey(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9가-힣_-]+/g, "-");
}
