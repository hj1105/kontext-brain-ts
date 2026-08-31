export interface CanonicalOntologySnapshot {
  readonly revision: string;
  readonly yaml: string;
}

export interface CanonicalOntologySource {
  read(): Promise<CanonicalOntologySnapshot>;
}

export interface GitHubCanonicalOntologySourceOptions {
  readonly owner: string;
  readonly repository: string;
  readonly token: string;
  readonly ontologyPath: string;
  readonly baseBranch?: string;
  readonly apiUrl?: string;
}

interface GitHubContentResponse {
  readonly sha?: string;
  readonly content?: string;
  readonly encoding?: string;
}

export class GitHubCanonicalOntologySource implements CanonicalOntologySource {
  private readonly baseBranch: string;
  private readonly apiUrl: string;

  constructor(
    private readonly options: GitHubCanonicalOntologySourceOptions,
    private readonly request: typeof fetch = fetch,
  ) {
    this.baseBranch = options.baseBranch ?? "main";
    this.apiUrl = (options.apiUrl ?? "https://api.github.com").replace(/\/$/, "");
  }

  async read(): Promise<CanonicalOntologySnapshot> {
    const path = this.options.ontologyPath.split("/").map(encodeURIComponent).join("/");
    const response = await this.request(
      `${this.apiUrl}/repos/${encodeURIComponent(this.options.owner)}/${encodeURIComponent(this.options.repository)}/contents/${path}?ref=${encodeURIComponent(this.baseBranch)}`,
      {
        headers: {
          Accept: "application/vnd.github+json",
          Authorization: `Bearer ${this.options.token}`,
          "X-GitHub-Api-Version": "2022-11-28",
        },
      },
    );
    if (!response.ok) throw new Error(`GitHub API ${response.status}: ${await response.text()}`);
    const body = (await response.json()) as GitHubContentResponse;
    if (!body.sha || !body.content || body.encoding !== "base64") {
      throw new Error("GitHub ontology response did not include base64 content and a revision");
    }
    return {
      revision: body.sha,
      yaml: Buffer.from(body.content.replace(/\s/g, ""), "base64").toString("utf8"),
    };
  }
}
