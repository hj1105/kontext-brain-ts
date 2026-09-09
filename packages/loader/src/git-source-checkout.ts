import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

/**
 * Turns a repository URL into a directory the Markdown connector can read, so a
 * user can point the ontology at a repository without standing up an MCP server
 * or minting a token for it. Private repositories work exactly as far as the
 * user's own `git clone` does: the checkout runs with their git credentials.
 */

export interface GitSourceCheckoutOptions {
  /** Branch or tag to read; the remote's default branch when omitted. */
  readonly ref?: string;
  readonly cacheRoot?: string;
  readonly env?: NodeJS.ProcessEnv;
}

export class GitSourceError extends Error {
  override readonly name = "GitSourceError";
}

const CLONE_TIMEOUT_MS = 5 * 60 * 1000;

export function gitSourceCacheRoot(env: NodeJS.ProcessEnv = process.env): string {
  if (env.KONTEXT_GIT_SOURCE_CACHE) return env.KONTEXT_GIT_SOURCE_CACHE;
  const base =
    env.XDG_CACHE_HOME ||
    (process.platform === "win32" ? env.LOCALAPPDATA : undefined) ||
    join(homedir(), ".cache");
  return join(base, "kontext-brain", "git-sources");
}

export function gitSourceCheckoutDirectory(
  url: string,
  ref: string | undefined,
  cacheRoot: string,
) {
  const key = createHash("sha256")
    .update(`${url}\n${ref ?? ""}`)
    .digest("hex")
    .slice(0, 32);
  return join(cacheRoot, key);
}

function git(args: readonly string[], env: NodeJS.ProcessEnv): void {
  try {
    execFileSync("git", [...args], {
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
      timeout: CLONE_TIMEOUT_MS,
      // Why: an interactive credential prompt would hang the check with no output;
      // failing fast names the repository the user still has to sign in to.
      env: { ...env, GIT_TERMINAL_PROMPT: "0" },
    });
  } catch (error) {
    const stderr =
      error && typeof error === "object" && "stderr" in error
        ? String((error as { stderr: unknown }).stderr ?? "").trim()
        : "";
    const detail = stderr || (error instanceof Error ? error.message : String(error));
    throw new GitSourceError(`git ${args[0]} failed: ${detail}`);
  }
}

/**
 * Clones on first use and refreshes afterwards. A stale checkout would build the
 * ontology from documents the team has since changed, so every materialization
 * fetches the pinned ref (or the remote default) and resets to it.
 */
export function materializeGitSource(url: string, options: GitSourceCheckoutOptions = {}): string {
  const env = options.env ?? process.env;
  const cacheRoot = options.cacheRoot ?? gitSourceCacheRoot(env);
  const directory = gitSourceCheckoutDirectory(url, options.ref, cacheRoot);
  if (!existsSync(join(directory, ".git"))) {
    mkdirSync(cacheRoot, { recursive: true });
    git(
      [
        "clone",
        "--quiet",
        "--depth",
        "1",
        ...(options.ref ? ["--branch", options.ref] : []),
        "--",
        url,
        directory,
      ],
      env,
    );
    return directory;
  }
  git(["-C", directory, "fetch", "--quiet", "--depth", "1", "origin", options.ref ?? "HEAD"], env);
  git(["-C", directory, "reset", "--quiet", "--hard", "FETCH_HEAD"], env);
  return directory;
}
