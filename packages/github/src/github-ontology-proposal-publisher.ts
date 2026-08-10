import type { OntologyProposalPublisher } from "@kontext-brain/core";

export interface GitHubOntologyProposalPublisherOptions {
  readonly owner: string;
  readonly repository: string;
  readonly token: string;
  readonly ontologyPath: string;
  readonly baseBranch?: string;
  readonly proposalBranch?: string;
  readonly apiUrl?: string;
  readonly title?: string;
}

interface GitHubResponse {
  readonly sha?: string;
  readonly object?: { readonly sha?: string };
  readonly html_url?: string;
  readonly number?: number;
}

export class GitHubOntologyProposalPublisher implements OntologyProposalPublisher {
  private readonly baseBranch: string;
  private readonly proposalBranch: string;
  private readonly apiUrl: string;
  private readonly title: string;

  constructor(
    private readonly options: GitHubOntologyProposalPublisherOptions,
    private readonly request: typeof fetch = fetch,
  ) {
    this.baseBranch = options.baseBranch ?? "main";
    this.proposalBranch = options.proposalBranch ?? "kontext/ontology-proposals";
    this.apiUrl = (options.apiUrl ?? "https://api.github.com").replace(/\/$/, "");
    this.title = options.title ?? "chore(ontology): apply Kontext proposals";
  }

  async upsert(
    input: Parameters<OntologyProposalPublisher["upsert"]>[0],
  ): Promise<{ url: string }> {
    await this.ensureProposalBranch();
    const current = await this.optionalJson<GitHubResponse>(
      `/repos/${this.options.owner}/${this.options.repository}/contents/${encodePath(this.options.ontologyPath)}?ref=${encodeURIComponent(this.proposalBranch)}`,
    );
    await this.json(
      `/repos/${this.options.owner}/${this.options.repository}/contents/${encodePath(this.options.ontologyPath)}`,
      {
        method: "PUT",
        body: JSON.stringify({
          message: "chore(ontology): update queued proposals",
          content: Buffer.from(input.yaml, "utf8").toString("base64"),
          branch: this.proposalBranch,
          sha: current?.sha,
        }),
      },
    );

    const head = `${this.options.owner}:${this.proposalBranch}`;
    const pulls = await this.json<GitHubResponse[]>(
      `/repos/${this.options.owner}/${this.options.repository}/pulls?state=open&head=${encodeURIComponent(head)}&base=${encodeURIComponent(this.baseBranch)}`,
    );
    const body = proposalBody(input.organizationId, input.proposals);
    const existing = pulls[0];
    const pull = existing?.number
      ? await this.json<GitHubResponse>(
          `/repos/${this.options.owner}/${this.options.repository}/pulls/${existing.number}`,
          { method: "PATCH", body: JSON.stringify({ title: this.title, body }) },
        )
      : await this.json<GitHubResponse>(
          `/repos/${this.options.owner}/${this.options.repository}/pulls`,
          {
            method: "POST",
            body: JSON.stringify({
              title: this.title,
              body,
              head: this.proposalBranch,
              base: this.baseBranch,
              draft: true,
            }),
          },
        );
    if (!pull.html_url) throw new Error("GitHub did not return a pull request URL");
    return { url: pull.html_url };
  }

  private async ensureProposalBranch(): Promise<void> {
    const branchPath = `/repos/${this.options.owner}/${this.options.repository}/git/ref/heads/${encodeURIComponent(this.proposalBranch)}`;
    if (await this.optionalJson<GitHubResponse>(branchPath)) return;
    const base = await this.json<GitHubResponse>(
      `/repos/${this.options.owner}/${this.options.repository}/git/ref/heads/${encodeURIComponent(this.baseBranch)}`,
    );
    const sha = base.object?.sha;
    if (!sha) throw new Error("GitHub base branch response did not include a SHA");
    await this.json(`/repos/${this.options.owner}/${this.options.repository}/git/refs`, {
      method: "POST",
      body: JSON.stringify({ ref: `refs/heads/${this.proposalBranch}`, sha }),
    });
  }

  private async optionalJson<T>(path: string): Promise<T | null> {
    const response = await this.request(`${this.apiUrl}${path}`, this.requestInit());
    if (response.status === 404) return null;
    if (!response.ok) throw new Error(`GitHub API ${response.status}: ${await response.text()}`);
    return (await response.json()) as T;
  }

  private async json<T = GitHubResponse>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await this.request(`${this.apiUrl}${path}`, this.requestInit(init));
    if (!response.ok) throw new Error(`GitHub API ${response.status}: ${await response.text()}`);
    return (await response.json()) as T;
  }

  private requestInit(init: RequestInit = {}): RequestInit {
    return {
      ...init,
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${this.options.token}`,
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        ...init.headers,
      },
    };
  }
}

function proposalBody(
  organizationId: string,
  proposals: Parameters<OntologyProposalPublisher["upsert"]>[0]["proposals"],
): string {
  const rows = proposals.map(
    (proposal) =>
      `- \`${proposal.suggestedNodeId}\`: ${proposal.description} (${proposal.occurrences} observations)`,
  );
  return [
    `Automated ontology proposal update for organization \`${organizationId}\`.`,
    "",
    ...rows,
    "",
    "The active ontology changes only after this PR is reviewed, merged, and deployed.",
  ].join("\n");
}

function encodePath(path: string): string {
  return path.split("/").map(encodeURIComponent).join("/");
}
