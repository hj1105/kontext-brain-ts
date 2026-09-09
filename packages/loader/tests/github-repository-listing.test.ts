import { describe, expect, it } from "vitest";
import {
  type GitHubApiRunner,
  GitHubListingError,
  ghExecutableCandidates,
  listGitHubRepositories,
  parseGitHubOwner,
} from "../src/index.js";

const line = (name: string, extra: Record<string, unknown> = {}) =>
  JSON.stringify({
    name,
    full_name: `modapl/${name}`,
    html_url: `https://github.com/modapl/${name}`,
    clone_url: `https://github.com/modapl/${name}.git`,
    default_branch: "main",
    private: false,
    archived: false,
    fork: false,
    language: "TypeScript",
    description: null,
    pushed_at: "2026-09-01T00:00:00Z",
    ...extra,
  });

describe("parseGitHubOwner", () => {
  it("accepts a name, an organization URL, a repository URL and an ssh remote", () => {
    for (const input of [
      "modapl",
      " https://github.com/modapl ",
      "https://github.com/modapl/",
      "github.com/modapl/kondex",
      "git@github.com:modapl/kondex.git",
      "https://www.github.com/modapl?tab=repositories",
    ]) {
      expect(parseGitHubOwner(input), input).toBe("modapl");
    }
  });

  it("refuses anything that is not an owner", () => {
    for (const input of ["", "https://gitlab.com/x", "-bad", "a b", "https://github.com/"]) {
      expect(() => parseGitHubOwner(input), input).toThrow(GitHubListingError);
    }
  });
});

describe("listGitHubRepositories", () => {
  it("lists an organization through gh, most recently pushed first", async () => {
    const calls: string[] = [];
    const run: GitHubApiRunner = async (endpoint) => {
      calls.push(endpoint);
      return {
        exitCode: 0,
        stderr: "",
        stdout: [
          line("older", { pushed_at: "2026-01-01T00:00:00Z", archived: true }),
          line("newer", { private: true, fork: true, language: null }),
          "",
        ].join("\n"),
      };
    };
    const listing = await listGitHubRepositories("https://github.com/modapl", { run });
    expect(calls).toEqual(["orgs/modapl/repos?per_page=100&type=all"]);
    expect(listing.kind).toBe("organization");
    expect(listing.repositories.map((repository) => repository.name)).toEqual(["newer", "older"]);
    expect(listing.repositories[0]).toMatchObject({
      fullName: "modapl/newer",
      cloneUrl: "https://github.com/modapl/newer.git",
      private: true,
      fork: true,
      language: null,
    });
    expect(listing.repositories[1]?.archived).toBe(true);
  });

  it("falls back to a user account when the organization does not exist", async () => {
    const run: GitHubApiRunner = async (endpoint) =>
      endpoint.startsWith("orgs/")
        ? { exitCode: 1, stdout: "", stderr: "gh: Not Found (HTTP 404)" }
        : { exitCode: 0, stdout: line("dotfiles"), stderr: "" };
    const listing = await listGitHubRepositories("hj1105", { run });
    expect(listing.kind).toBe("user");
    expect(listing.repositories.map((repository) => repository.name)).toEqual(["dotfiles"]);
  });

  it("reports a missing owner and a signed-out gh in words the user can act on", async () => {
    const missing: GitHubApiRunner = async () => ({
      exitCode: 1,
      stdout: "",
      stderr: "gh: Not Found (HTTP 404)",
    });
    await expect(listGitHubRepositories("nobody-here", { run: missing })).rejects.toThrow(
      /no organization or user named "nobody-here"/,
    );
    const signedOut: GitHubApiRunner = async () => ({
      exitCode: 4,
      stdout: "",
      stderr: "To get started with GitHub CLI, please run:  gh auth login",
    });
    await expect(listGitHubRepositories("modapl", { run: signedOut })).rejects.toThrow(
      /gh auth login/,
    );
  });

  it("uses the REST API with a token when gh is not installed", async () => {
    const requested: string[] = [];
    const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      requested.push(url);
      expect((init?.headers as Record<string, string>).Authorization).toBe("Bearer ghp_test");
      const page = Number(new URL(url).searchParams.get("page"));
      const body =
        page === 1
          ? Array.from({ length: 100 }, (_, index) => JSON.parse(line(`r${index}`)))
          : [JSON.parse(line("last"))];
      return new Response(JSON.stringify(body), { status: 200 });
    }) as typeof fetch;
    const listing = await listGitHubRepositories("modapl", {
      run: async () => ({ exitCode: 127, stdout: "", stderr: "gh is not installed" }),
      fetchImpl,
      env: { GITHUB_TOKEN: "ghp_test" },
    });
    expect(requested).toHaveLength(2);
    expect(listing.repositories).toHaveLength(101);
  });
});

describe("ghExecutableCandidates", () => {
  it("tries PATH first, then the places a Dock-launched app cannot see", () => {
    expect(ghExecutableCandidates({ HOME: "/Users/me" }, "darwin")).toEqual([
      "gh",
      "/opt/homebrew/bin/gh",
      "/usr/local/bin/gh",
      "/home/linuxbrew/.linuxbrew/bin/gh",
      "/Users/me/.local/bin/gh",
      "/Users/me/.nix-profile/bin/gh",
    ]);
    expect(ghExecutableCandidates({ ProgramFiles: "D:\\PF" }, "win32")).toEqual([
      "gh",
      "D:\\PF\\GitHub CLI\\gh.exe",
    ]);
  });
});
