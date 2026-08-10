import { createHash } from "node:crypto";

export interface OntologyDeployment<TGraph> {
  readonly organizationId: string;
  readonly contentHash: string;
  readonly yaml: string;
  readonly graph: TGraph;
  readonly gitCommit?: string;
  readonly status: "staged" | "active" | "retired" | "failed";
  readonly createdAt: string;
  readonly failure?: string;
}

export interface OntologyCompiler<TGraph> {
  compile(yaml: string, contentHash: string): Promise<TGraph>;
}

export interface OntologyCandidateValidator<TGraph> {
  validate(candidate: OntologyDeployment<TGraph>): Promise<void>;
}

export interface OntologyReindexer<TGraph> {
  prepare(candidate: OntologyDeployment<TGraph>): Promise<void>;
}

export interface OntologyDeploymentRepository<TGraph> {
  getActive(organizationId: string): Promise<OntologyDeployment<TGraph> | null>;
  stage(candidate: OntologyDeployment<TGraph>): Promise<void>;
  activate(organizationId: string, contentHash: string): Promise<OntologyDeployment<TGraph>>;
  markFailed(organizationId: string, contentHash: string, failure: string): Promise<void>;
}

export interface ActivateOntologyInput {
  readonly organizationId: string;
  readonly yaml: string;
  readonly gitCommit?: string;
}

export interface ActivateOntologyResult {
  readonly changed: boolean;
  readonly contentHash: string;
}

export class ActivateOntologyUseCase<TGraph> {
  constructor(
    private readonly repository: OntologyDeploymentRepository<TGraph>,
    private readonly compiler: OntologyCompiler<TGraph>,
    private readonly validator: OntologyCandidateValidator<TGraph>,
    private readonly reindexer: OntologyReindexer<TGraph>,
    private readonly now: () => Date = () => new Date(),
  ) {}

  async execute(input: ActivateOntologyInput): Promise<ActivateOntologyResult> {
    const contentHash = createHash("sha256").update(input.yaml).digest("hex");
    const active = await this.repository.getActive(input.organizationId);
    if (active?.contentHash === contentHash) return { changed: false, contentHash };

    let staged = false;
    try {
      const graph = await this.compiler.compile(input.yaml, contentHash);
      const candidate: OntologyDeployment<TGraph> = {
        organizationId: input.organizationId,
        contentHash,
        yaml: input.yaml,
        graph,
        gitCommit: input.gitCommit,
        status: "staged",
        createdAt: this.now().toISOString(),
      };
      await this.validator.validate(candidate);
      await this.repository.stage(candidate);
      staged = true;
      await this.reindexer.prepare(candidate);
      await this.repository.activate(input.organizationId, contentHash);
      return { changed: true, contentHash };
    } catch (error) {
      if (staged) {
        await this.repository.markFailed(
          input.organizationId,
          contentHash,
          error instanceof Error ? error.message : String(error),
        );
      }
      throw error;
    }
  }
}

export class InMemoryOntologyDeploymentRepository<TGraph>
  implements OntologyDeploymentRepository<TGraph>
{
  private readonly deployments = new Map<string, Map<string, OntologyDeployment<TGraph>>>();
  private readonly activeHashes = new Map<string, string>();

  async getActive(organizationId: string): Promise<OntologyDeployment<TGraph> | null> {
    const hash = this.activeHashes.get(organizationId);
    if (!hash) return null;
    return this.deployments.get(organizationId)?.get(hash) ?? null;
  }

  async stage(candidate: OntologyDeployment<TGraph>): Promise<void> {
    let organization = this.deployments.get(candidate.organizationId);
    if (!organization) {
      organization = new Map();
      this.deployments.set(candidate.organizationId, organization);
    }
    organization.set(candidate.contentHash, { ...candidate, status: "staged" });
  }

  async activate(organizationId: string, contentHash: string): Promise<OntologyDeployment<TGraph>> {
    const organization = this.deployments.get(organizationId);
    const candidate = organization?.get(contentHash);
    if (!candidate || candidate.status !== "staged") {
      throw new Error(`Ontology candidate "${contentHash}" is not staged`);
    }
    const active = { ...candidate, status: "active" as const };
    const previousHash = this.activeHashes.get(organizationId);
    const previous = previousHash ? organization?.get(previousHash) : undefined;
    if (previous && previousHash) {
      organization?.set(previousHash, { ...previous, status: "retired" });
    }
    organization?.set(contentHash, active);
    this.activeHashes.set(organizationId, contentHash);
    return active;
  }

  async markFailed(organizationId: string, contentHash: string, failure: string): Promise<void> {
    const organization = this.deployments.get(organizationId);
    const candidate = organization?.get(contentHash);
    if (!candidate) return;
    organization?.set(contentHash, { ...candidate, status: "failed", failure });
  }
}
