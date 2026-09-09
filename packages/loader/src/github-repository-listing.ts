import { execFile } from "node:child_process";

/**
 * Lists the repositories of a GitHub organization or user so a host can offer
 * them as a checklist instead of asking for one clone URL at a time. The user's
 * own `gh` login does the asking, so private repositories appear exactly as far
 * as that login can see them and no token is copied anywhere. Without `gh`, a
 * token in GITHUB_TOKEN / GH_TOKEN (or an anonymous request for public
 * organizations) reaches the REST API directly.
 */

export interface GitHubRepositorySummary {
  readonly name: string;
  readonly fullName: string;
  readonly url: string;
  readonly cloneUrl: string;
  readonly defaultBranch: string;
  readonly private: boolean;
  readonly archived: boolean;
  readonly fork: boolean;
  readonly language: string | null;
  readonly description: string | null;
  readonly pushedAt: string | null;
}

export interface GitHubRepositoryListing {
  readonly owner: string;
  readonly kind: "organization" | "user";
  readonly repositories: readonly GitHubRepositorySummary[];
}

export class GitHubListingError extends Error {
  override readonly name = "GitHubListingError";
}

export interface GitHubApiResult {
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
}

/** Runs `gh api <endpoint> --paginate --jq <filter>`; exit 127 means gh is unavailable. */
export type GitHubApiRunner = (endpoint: string, jq: string) => Promise<GitHubApiResult>;

export interface ListGitHubRepositoriesOptions {
  readonly run?: GitHubApiRunner;
  readonly fetchImpl?: typeof fetch;
  readonly env?: NodeJS.ProcessEnv;
}

const OWNER_PATTERN = /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$/;
const REPOSITORY_FIELDS =
  "{name, full_name, html_url, clone_url, default_branch, private, archived, fork, language, description, pushed_at}";

/** Accepts `org`, `https://github.com/org`, `github.com/org/repo`, or `git@github.com:org/repo.git`. */
export function parseGitHubOwner(input: string): string {
  let value = input.trim();
  value = value.replace(/^git@github\.com:/i, "");
  value = value.replace(/^[a-z]+:\/\//i, "");
  value = value.replace(/^(www\.)?github\.com\//i, "");
  const owner = value.split(/[/?#]/)[0] ?? "";
  if (!OWNER_PATTERN.test(owner)) {
    throw new GitHubListingError(
      `"${input}" is not a GitHub organization or user; paste the organization URL or name.`,
    );
  }
  return owner;
}

export async function listGitHubRepositories(
  input: string,
  options: ListGitHubRepositoriesOptions = {},
): Promise<GitHubRepositoryListing> {
  const owner = parseGitHubOwner(input);
  const run = options.run ?? ghRunner(options.env ?? process.env);
  const organization = await run(
    `orgs/${owner}/repos?per_page=100&type=all`,
    `.[] | ${REPOSITORY_FIELDS}`,
  );
  if (organization.exitCode === 0) {
    return { owner, kind: "organization", repositories: parseLines(organization.stdout) };
  }
  if (organization.exitCode === 127) {
    return listThroughRest(owner, options);
  }
  if (!isNotFound(organization.stderr)) {
    throw new GitHubListingError(ghFailure(organization.stderr));
  }
  const user = await run(`users/${owner}/repos?per_page=100`, `.[] | ${REPOSITORY_FIELDS}`);
  if (user.exitCode !== 0) {
    throw new GitHubListingError(
      isNotFound(user.stderr)
        ? `GitHub has no organization or user named "${owner}".`
        : ghFailure(user.stderr),
    );
  }
  return { owner, kind: "user", repositories: parseLines(user.stdout) };
}

/**
 * Where gh lives when PATH does not say: a desktop app launched from the Dock
 * inherits a PATH without Homebrew, while the user's terminal has it.
 */
export function ghExecutableCandidates(
  env: NodeJS.ProcessEnv,
  platform: NodeJS.Platform = process.platform,
): readonly string[] {
  if (platform === "win32") {
    const programFiles = env.ProgramFiles ?? "C:\\Program Files";
    return ["gh", `${programFiles}\\GitHub CLI\\gh.exe`];
  }
  const home = env.HOME ?? "";
  return [
    "gh",
    "/opt/homebrew/bin/gh",
    "/usr/local/bin/gh",
    "/home/linuxbrew/.linuxbrew/bin/gh",
    ...(home ? [`${home}/.local/bin/gh`, `${home}/.nix-profile/bin/gh`] : []),
  ];
}

function ghRunner(env: NodeJS.ProcessEnv): GitHubApiRunner {
  const candidates = ghExecutableCandidates(env);
  const runWith = (executable: string, args: readonly string[]): Promise<GitHubApiResult> =>
    new Promise((resolve) => {
      execFile(
        executable,
        [...args],
        // Why: a large organization returns thousands of lines; the default buffer truncates them.
        { env, maxBuffer: 64 * 1024 * 1024, windowsHide: true },
        (error, stdout, stderr) => {
          if (error && "code" in error && error.code === "ENOENT") {
            resolve({ exitCode: 127, stdout: "", stderr: "gh is not installed" });
            return;
          }
          const exitCode =
            error && typeof (error as { code?: unknown }).code === "number"
              ? ((error as { code: number }).code as number)
              : error
                ? 1
                : 0;
          resolve({ exitCode, stdout: String(stdout), stderr: String(stderr) });
        },
      );
    });
  return async (endpoint, jq) => {
    const args = ["api", endpoint, "--paginate", "--jq", jq];
    let result: GitHubApiResult = { exitCode: 127, stdout: "", stderr: "gh is not installed" };
    for (const executable of candidates) {
      result = await runWith(executable, args);
      if (result.exitCode !== 127) return result;
    }
    return result;
  };
}

async function listThroughRest(
  owner: string,
  options: ListGitHubRepositoriesOptions,
): Promise<GitHubRepositoryListing> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const env = options.env ?? process.env;
  const token = env.GITHUB_TOKEN ?? env.GH_TOKEN;
  const headers: Record<string, string> = {
    Accept: "application/vnd.github+json",
    "User-Agent": "kontext-brain",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  const collect = async (kind: "organization" | "user") => {
    const base = kind === "organization" ? `orgs/${owner}/repos?type=all` : `users/${owner}/repos?`;
    const repositories: GitHubRepositorySummary[] = [];
    for (let page = 1; page <= 50; page += 1) {
      const response = await fetchImpl(
        `https://api.github.com/${base}&per_page=100&page=${page}`.replace("?&", "?"),
        { headers },
      );
      if (response.status === 404) return undefined;
      if (!response.ok) {
        throw new GitHubListingError(
          `GitHub answered ${response.status} for ${owner}; sign in with \`gh auth login\` or set GITHUB_TOKEN.`,
        );
      }
      const batch = (await response.json()) as unknown;
      if (!Array.isArray(batch))
        throw new GitHubListingError("GitHub returned an unexpected body.");
      for (const item of batch) repositories.push(summarize(item));
      if (batch.length < 100) break;
    }
    return repositories;
  };
  const organization = await collect("organization");
  if (organization) return { owner, kind: "organization", repositories: sortByPush(organization) };
  const user = await collect("user");
  if (user) return { owner, kind: "user", repositories: sortByPush(user) };
  throw new GitHubListingError(`GitHub has no organization or user named "${owner}".`);
}

function parseLines(stdout: string): readonly GitHubRepositorySummary[] {
  const repositories: GitHubRepositorySummary[] = [];
  for (const line of stdout.split(/\r?\n/)) {
    if (line.trim() === "") continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      throw new GitHubListingError("gh returned a line that is not JSON; update gh and retry.");
    }
    repositories.push(summarize(parsed));
  }
  return sortByPush(repositories);
}

function summarize(value: unknown): GitHubRepositorySummary {
  const record = (typeof value === "object" && value !== null ? value : {}) as Record<
    string,
    unknown
  >;
  const text = (key: string): string | null =>
    typeof record[key] === "string" ? (record[key] as string) : null;
  const name = text("name");
  const fullName = text("full_name");
  const cloneUrl = text("clone_url");
  if (!name || !fullName || !cloneUrl) {
    throw new GitHubListingError("GitHub returned a repository without a name or clone URL.");
  }
  return {
    name,
    fullName,
    url: text("html_url") ?? `https://github.com/${fullName}`,
    cloneUrl,
    defaultBranch: text("default_branch") ?? "main",
    private: record.private === true,
    archived: record.archived === true,
    fork: record.fork === true,
    language: text("language"),
    description: text("description"),
    pushedAt: text("pushed_at"),
  };
}

/** Most recently pushed first: the repositories a team is actually working in. */
function sortByPush(repositories: readonly GitHubRepositorySummary[]): GitHubRepositorySummary[] {
  return [...repositories].sort(
    (left, right) =>
      (right.pushedAt ?? "").localeCompare(left.pushedAt ?? "") ||
      left.name.localeCompare(right.name),
  );
}

function isNotFound(stderr: string): boolean {
  return /\b404\b|Not Found/i.test(stderr);
}

function ghFailure(stderr: string): string {
  const line =
    stderr
      .trim()
      .split(/\r?\n/)
      .find((part) => part.trim() !== "") ?? "";
  if (/auth login|not logged in|authentication/i.test(line)) {
    return "gh is not signed in; run `gh auth login` and retry.";
  }
  return line ? `gh failed: ${line}` : "gh failed without a message.";
}
