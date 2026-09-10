import { existsSync } from "node:fs";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { gitSourceCheckoutDirectory, readableCheckoutPath } from "../src/index.js";

const roots: string[] = [];
afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("readableCheckoutPath", () => {
  it("names a checkout by host, owner and repository so it can be browsed", () => {
    expect(readableCheckoutPath("https://github.com/modapl/sphere-admin.git", undefined)).toBe(
      path.join("github.com", "modapl", "sphere-admin"),
    );
    expect(readableCheckoutPath("git@github.com:modapl/teslaFleet.git", "release")).toBe(
      path.join("github.com", "modapl", "teslaFleet@release"),
    );
    expect(readableCheckoutPath("https://gitlab.example.com/group/sub/repo", undefined)).toBe(
      path.join("gitlab.example.com", "group", "sub", "repo"),
    );
  });

  it("falls back to a digest for remotes without a host, owner and repository", () => {
    expect(readableCheckoutPath("file:///tmp/handbook.git", undefined)).toBeUndefined();
    expect(readableCheckoutPath("https://github.com/only-owner", undefined)).toBeUndefined();
    expect(readableCheckoutPath("https://github.com/a/..", undefined)).toBeUndefined();
    const directory = gitSourceCheckoutDirectory("file:///tmp/handbook.git", undefined, "/cache");
    expect(path.dirname(directory)).toBe("/cache");
    expect(path.basename(directory)).toMatch(/^[a-f0-9]{32}$/);
  });
});

describe("gitSourceCheckoutDirectory", () => {
  it("adopts a checkout made under the old digest name when materializing", async () => {
    const cache = await mkdtemp(path.join(tmpdir(), "kontext-git-cache-"));
    roots.push(cache);
    // Why: a 1.5 GB cache of digest-named clones must not be cloned again after the rename.
    const url = "https://github.com/modapl/sphere-admin.git";
    const { materializeGitSource } = await import("../src/git-source-checkout.js");
    const legacy = path.join(cache, "1f3a1b0a1b0a1b0a1b0a1b0a1b0a1b0a");
    const { createHash } = await import("node:crypto");
    const key = createHash("sha256").update(`${url}\n`).digest("hex").slice(0, 32);
    const legacyDirectory = path.join(cache, key);
    await mkdir(path.join(legacyDirectory, ".git"), { recursive: true });
    await writeFile(path.join(legacyDirectory, "README.md"), "# moved\n");
    void legacy;
    let failed = "";
    try {
      // The fetch against GitHub is not attempted here; the rename happens before it.
      materializeGitSource(url, { cacheRoot: cache, env: { PATH: "/nonexistent" } });
    } catch (error) {
      failed = String(error);
    }
    const readable = gitSourceCheckoutDirectory(url, undefined, cache);
    expect(readable).toBe(path.join(cache, "github.com", "modapl", "sphere-admin"));
    expect(existsSync(path.join(readable, "README.md"))).toBe(true);
    expect(existsSync(legacyDirectory)).toBe(false);
    expect(failed).toContain("git");
  });
});
