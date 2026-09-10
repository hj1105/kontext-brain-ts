import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, renameSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

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

/**
 * `<cache>/<host>/<owner>/<repo>[@<ref>]`, so a person or an agent can browse
 * and grep every connected repository by name instead of asking GitHub for one
 * file at a time. A URL that does not name a host, owner and repository (a
 * file:// remote, an odd mirror) falls back to a digest directory.
 */
export function gitSourceCheckoutDirectory(
  url: string,
  ref: string | undefined,
  cacheRoot: string,
) {
  const readable = readableCheckoutPath(url, ref);
  return join(cacheRoot, readable ?? legacyCheckoutKey(url, ref));
}

function legacyCheckoutKey(url: string, ref: string | undefined): string {
  return createHash("sha256")
    .update(`${url}\n${ref ?? ""}`)
    .digest("hex")
    .slice(0, 32);
}

const SAFE_SEGMENT = /^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$/;

export function readableCheckoutPath(url: string, ref: string | undefined): string | undefined {
  let host: string | undefined;
  let repositoryPath: string | undefined;
  const ssh = /^(?:[\w.-]+@)?([\w.-]+):(?!\/)(.+)$/.exec(url.trim());
  if (ssh) {
    host = ssh[1];
    repositoryPath = ssh[2];
  } else {
    try {
      const parsed = new URL(url.trim());
      if (parsed.protocol !== "https:" && parsed.protocol !== "http:" && parsed.protocol !== "ssh:")
        return undefined;
      host = parsed.hostname;
      repositoryPath = parsed.pathname;
    } catch {
      return undefined;
    }
  }
  if (!host || !repositoryPath) return undefined;
  const segments = repositoryPath
    .replace(/\.git\/?$/, "")
    .split("/")
    .filter((segment) => segment !== "");
  if (segments.length < 2) return undefined;
  const repository = segments.pop() as string;
  const suffix = ref ? `@${ref.replace(/[^A-Za-z0-9._-]+/g, "-")}` : "";
  const parts = [host.toLowerCase(), ...segments, `${repository}${suffix}`];
  if (!parts.every((part) => SAFE_SEGMENT.test(part) && part !== "." && part !== "..")) {
    return undefined;
  }
  return join(...parts);
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

/** A checkout made under the old digest name moves to its readable name instead of cloning again. */
function adoptLegacyCheckout(
  url: string,
  ref: string | undefined,
  cacheRoot: string,
  directory: string,
): void {
  const legacy = join(cacheRoot, legacyCheckoutKey(url, ref));
  if (legacy === directory || !existsSync(join(legacy, ".git")) || existsSync(directory)) return;
  mkdirSync(dirname(directory), { recursive: true });
  renameSync(legacy, directory);
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
  adoptLegacyCheckout(url, options.ref, cacheRoot, directory);
  if (!existsSync(join(directory, ".git"))) {
    mkdirSync(dirname(directory), { recursive: true });
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
