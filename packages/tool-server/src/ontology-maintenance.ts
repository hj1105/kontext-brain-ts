import type { OntologyProposalPublisher } from "@kontext-brain/core";
import type { CanonicalOntologySnapshot, CanonicalOntologySource } from "@kontext-brain/github";
import type { OntologyActivationResult } from "@kontext-brain/loader";

export interface OntologyMaintenanceTarget {
  readonly activeOntologyContentHash?: string;
  readonly activeOntologyNodeCount: number;
  activateOntologyYaml(yaml: string): Promise<OntologyActivationResult>;
  publishOntologyProposals(
    activeYaml: string,
    publisher: OntologyProposalPublisher,
  ): Promise<{ readonly changed: boolean; readonly yaml: string; readonly url?: string }>;
}

export interface OntologyMaintenanceStatus {
  readonly canonicalRevision?: string;
  readonly lastCheckedAt?: string;
  readonly lastActivatedAt?: string;
  readonly lastPublishedAt?: string;
  readonly lastProposalUrl?: string;
  readonly lastError?: string;
}

export class OntologyMaintenanceService {
  private canonical: CanonicalOntologySnapshot | null = null;
  private status: OntologyMaintenanceStatus = {};

  constructor(
    private readonly agent: OntologyMaintenanceTarget,
    private readonly source: CanonicalOntologySource,
    private readonly publisher: OntologyProposalPublisher,
    private readonly now: () => Date = () => new Date(),
  ) {}

  getStatus(): OntologyMaintenanceStatus {
    return this.status;
  }

  async refreshCanonical(): Promise<OntologyActivationResult> {
    try {
      const canonical = await this.source.read();
      const checkedAt = this.now().toISOString();
      if (canonical.revision === this.canonical?.revision) {
        this.status = { ...this.status, lastCheckedAt: checkedAt, lastError: undefined };
        return {
          changed: false,
          contentHash: this.agent.activeOntologyContentHash ?? "",
          nodeCount: this.agent.activeOntologyNodeCount,
        };
      }
      const activation = await this.agent.activateOntologyYaml(canonical.yaml);
      this.canonical = canonical;
      this.status = {
        ...this.status,
        canonicalRevision: canonical.revision,
        lastCheckedAt: checkedAt,
        lastActivatedAt: activation.changed ? checkedAt : this.status.lastActivatedAt,
        lastError: undefined,
      };
      return activation;
    } catch (error) {
      this.recordError(error);
      throw error;
    }
  }

  async publishProposals(): Promise<{
    readonly changed: boolean;
    readonly yaml: string;
    readonly url?: string;
  }> {
    try {
      if (!this.canonical) await this.refreshCanonical();
      if (!this.canonical) throw new Error("Canonical ontology is unavailable");
      const published = await this.agent.publishOntologyProposals(
        this.canonical.yaml,
        this.publisher,
      );
      if (published.changed) {
        this.status = {
          ...this.status,
          lastPublishedAt: this.now().toISOString(),
          lastProposalUrl: published.url,
          lastError: undefined,
        };
      }
      return published;
    } catch (error) {
      this.recordError(error);
      throw error;
    }
  }

  private recordError(error: unknown): void {
    this.status = {
      ...this.status,
      lastError: error instanceof Error ? error.message : String(error),
    };
  }
}
