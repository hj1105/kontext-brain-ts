import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { parse } from "yaml";
import { readConfigDocument } from "../src/kontext-config-file.js";
import { runOntologyCli } from "../src/ontology-cli.js";
import {
  addSource,
  checkSources,
  setDocumentMapping,
  summarizeSources,
} from "../src/ontology-source-inventory.js";

const roots: string[] = [];

function makeConfig(mcp: string): string {
  const root = mkdtempSync(join(tmpdir(), "kontext-inventory-"));
  roots.push(root);
  mkdirSync(join(root, "docs"));
  writeFileSync(join(root, "docs", "a.md"), "# Alpha\n\nfirst\n");
  writeFileSync(join(root, "docs", "b.md"), "# Beta\n\nsecond\n");
  const config = join(root, "kontext.yaml");
  writeFileSync(
    config,
    [
      "llm:",
      "  traversal: {provider: none, model: none}",
      "  reasoning: {provider: none, model: none}",
      mcp,
      "",
    ].join("\n"),
  );
  return config;
}

function localMcp(root: string): string {
  return [
    "mcp:",
    "  - name: repo-docs",
    "    transport: local",
    `    path: ${root}`,
    "    include: [docs]",
  ].join("\n");
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
  vi.restoreAllMocks();
});

async function run(argv: readonly string[]): Promise<{ code: number; printed: string }> {
  const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
  const code = await runOntologyCli(argv);
  const printed = out.mock.calls.map((call) => String(call[0])).join("");
  out.mockRestore();
  return { code, printed };
}

describe("summarizeSources", () => {
  it("reports the address that actually applies to each transport", () => {
    const config = makeConfig(
      [
        "mcp:",
        "  - {name: local, transport: local, path: /repo}",
        "  - {name: remote, transport: sse, url: 'https://example.invalid'}",
        "  - {name: spawned, transport: stdio, command: server, args: [--mcp]}",
      ].join("\n"),
    );
    expect(summarizeSources(readConfigDocument(config))).toEqual([
      { name: "local", transport: "local", type: null, target: "/repo", code: false },
      {
        name: "remote",
        transport: "sse",
        type: null,
        target: "https://example.invalid",
        code: false,
      },
      { name: "spawned", transport: "stdio", type: null, target: "server --mcp", code: false },
    ]);
  });
});

describe("summarizeSources tolerance", () => {
  it("keeps listing when one entry carries values this build does not know", () => {
    // A row copied from another agent must not fail the whole listing and hide
    // every other source from the surface.
    const config = makeConfig(
      [
        "mcp:",
        "  - {name: odd, transport: carrier-pigeon, url: 'https://x.invalid'}",
        "  - {name: scalar-args, transport: stdio, command: server, args: 3}",
        "  - {name: docs, transport: local, path: /repo}",
      ].join("\n"),
    );
    const sources = summarizeSources(readConfigDocument(config));
    expect(sources.map((s) => s.name)).toEqual(["odd", "scalar-args", "docs"]);
    expect(sources[0]?.transport).toBe("sse");
    expect(sources[1]).toMatchObject({ transport: "stdio", target: "server" });
  });
});

describe("addSource", () => {
  const empty = { path: "kontext.yaml", data: {} };

  it("keeps only the fields the chosen transport uses", () => {
    const document = addSource(empty, {
      name: "notion",
      transport: "sse",
      url: "https://mcp.notion.invalid",
      type: "notion",
      command: "ignored-for-sse",
    });
    expect(summarizeSources(document)).toEqual([
      {
        name: "notion",
        transport: "sse",
        type: "notion",
        target: "https://mcp.notion.invalid",
        code: false,
      },
    ]);
    expect(JSON.stringify(document.data)).not.toContain("ignored-for-sse");
  });

  it("records that a repository's code is read too, only where code can be read", () => {
    const document = addSource(empty, {
      name: "handbook",
      transport: "git",
      url: "https://github.com/org/handbook.git",
      code: true,
    });
    expect(summarizeSources(document)).toEqual([
      {
        name: "handbook",
        transport: "git",
        type: null,
        target: "https://github.com/org/handbook.git",
        code: true,
      },
    ]);
    // Why: a local checkout reads code exactly like a clone does; the flag was
    // once kept only for git and a local source silently stayed documents-only.
    const local = addSource(empty, { name: "repo", transport: "local", path: "/repo", code: true });
    expect(readMCPEntries(local)[0]?.code).toBe(true);
    const server = addSource(empty, {
      name: "notion",
      transport: "sse",
      url: "https://mcp.notion.invalid",
      code: true,
    });
    expect(JSON.stringify(server.data)).not.toContain("code");
  });

  it("adds an HTTP server with headers and a tool document mapping", () => {
    const document = addSource(empty, {
      name: "notion",
      transport: "http",
      url: "https://mcp.notion.com/mcp",
      headers: { Authorization: "Bearer ${NOTION_TOKEN}" },
      documents: {
        list: {
          tool: "search",
          arguments: { query: "" },
          items: "results",
          id: "id",
          title: "title",
        },
        read: { tool: "fetch", idArgument: "id", content: "text" },
      },
    });
    const [entry] = readMCPEntries(document);
    expect(entry).toMatchObject({
      name: "notion",
      transport: "http",
      url: "https://mcp.notion.com/mcp",
      headers: { Authorization: "Bearer ${NOTION_TOKEN}" },
      documents: { list: { tool: "search" }, read: { tool: "fetch", idArgument: "id" } },
    });
    expect(summarizeSources(document)[0]).toMatchObject({
      transport: "http",
      target: "https://mcp.notion.com/mcp",
    });
    expect(() =>
      addSource(empty, {
        name: "docs",
        transport: "local",
        path: "/docs",
        documents: { list: { tool: "x", id: "id" }, read: { tool: "y", idArgument: "id" } },
      }),
    ).toThrow(/applies to an MCP server/);
  });

  it("sets a document mapping on an existing server and refuses one on files", () => {
    const withServer = addSource(empty, {
      name: "gh",
      transport: "stdio",
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-github"],
    });
    const mapped = setDocumentMapping(withServer, "gh", {
      list: { tool: "search_issues", items: "items", id: "number", title: "title" },
      read: { tool: "get_issue", idArgument: "issue_number", content: "body" },
    });
    expect(readMCPEntries(mapped)[0]?.documents?.list.tool).toBe("search_issues");
    expect(readMCPEntries(setDocumentMapping(mapped, "gh", null))[0]?.documents).toBeUndefined();
    const files = addSource(empty, { name: "docs", transport: "local", path: "/docs" });
    expect(() =>
      setDocumentMapping(files, "docs", {
        list: { tool: "x", id: "id" },
        read: { tool: "y", idArgument: "id" },
      }),
    ).toThrow(/applies to an MCP server/);
    expect(() => setDocumentMapping(files, "nope", null)).toThrow(/No source named/);
  });

  it("refuses a source that is missing its address", () => {
    expect(() => addSource(empty, { name: "a", transport: "sse" })).toThrow(/needs its URL/);
    expect(() => addSource(empty, { name: "a", transport: "stdio" })).toThrow(/needs the command/);
    expect(() => addSource(empty, { name: "a", transport: "local" })).toThrow(
      /needs the directory/,
    );
    expect(() => addSource(empty, { name: "  ", transport: "local", path: "/x" })).toThrow(
      /needs a name/,
    );
  });

  it("refuses an unrecognised layer instead of silently using the Notion adapter", () => {
    expect(() =>
      addSource(empty, { name: "a", transport: "sse", url: "https://x.invalid", type: "notionn" }),
    ).toThrow(/Unknown layer/);
    // An absent layer is legitimate and means the default adapter.
    expect(() =>
      addSource(empty, { name: "a", transport: "sse", url: "https://x.invalid" }),
    ).not.toThrow();
  });

  it("refuses to overwrite an existing name", () => {
    const first = addSource(empty, { name: "docs", transport: "local", path: "/repo" });
    expect(() =>
      addSource(first, { name: "docs", transport: "sse", url: "https://x.invalid" }),
    ).toThrow(/already exists/);
  });
});

describe("checkSources", () => {
  it("reports each source separately instead of stopping at the first failure", async () => {
    const config = makeConfig(
      [
        "mcp:",
        "  - {name: broken, transport: stdio, command: /nonexistent/server}",
        "  - {name: docs, transport: local, path: .}",
      ].join("\n"),
    );
    const results = await checkSources(readConfigDocument(config));
    expect(results.map((r) => r.name)).toEqual(["broken", "docs"]);
    expect(results[0]).toMatchObject({ ok: false, resourceCount: null });
    expect(results[0]?.error).toBeTruthy();
    expect(results[1]).toMatchObject({ ok: true });
  });
});

describe("checkSources connector lifetime", () => {
  it("closes every probe connector, including one that failed to answer", async () => {
    // A stdio source spawns its MCP server on first use; leaving it open leaks one
    // server process per source per check.
    const closed: string[] = [];
    const connectors = new Map<
      string,
      { name: string; listResources: () => Promise<[]>; close: () => Promise<void> }
    >();
    for (const [name, fails] of [
      ["docs", false],
      ["broken", true],
    ] as const) {
      connectors.set(name, {
        name,
        listResources: async () => {
          if (fails) throw new Error("refused");
          return [];
        },
        close: async () => {
          closed.push(name);
        },
      });
    }
    const inventory = await import("../src/ontology-source-inventory.js");
    const connectorModule = await import("../src/ontology-source-connectors.js");
    const spy = vi
      .spyOn(connectorModule, "createSourceConnector")
      .mockImplementation((entry) => connectors.get(entry.name) as never);
    try {
      const config = makeConfig(
        [
          "mcp:",
          "  - {name: docs, transport: local, path: /repo}",
          "  - {name: broken, transport: local, path: /repo}",
        ].join("\n"),
      );
      const results = await inventory.checkSources(readConfigDocument(config));
      expect(results.map((r) => r.ok)).toEqual([true, false]);
      expect(closed.sort()).toEqual(["broken", "docs"]);
    } finally {
      spy.mockRestore();
    }
  });
});

describe("checkSources deadline", () => {
  it("gives up on a silent source instead of waiting out the client's own timeout", async () => {
    const connectorModule = await import("../src/ontology-source-connectors.js");
    const spy = vi.spyOn(connectorModule, "createSourceConnector").mockImplementation(
      () =>
        ({
          name: "silent",
          listResources: () => new Promise(() => {}),
          close: async () => {},
        }) as never,
    );
    try {
      const config = makeConfig("mcp:\n  - {name: silent, transport: local, path: /repo}");
      const started = Date.now();
      const results = await checkSources(readConfigDocument(config), 60);
      expect(Date.now() - started).toBeLessThan(2000);
      expect(results[0]).toMatchObject({ ok: false });
      expect(results[0]?.error).toMatch(/No answer within/);
    } finally {
      spy.mockRestore();
    }
  });
});

describe("checkSources with an unusable entry", () => {
  it("reports the bad entry and still checks the rest", async () => {
    // A half-written entry throws before a connector exists; the failure must stay
    // scoped to that source instead of discarding every other source's result.
    const config = makeConfig(
      [
        "mcp:",
        "  - {name: half-written, transport: sse}",
        "  - {name: docs, transport: local, path: .}",
      ].join("\n"),
    );
    const results = await checkSources(readConfigDocument(config));
    expect(results.map((r) => r.name)).toEqual(["half-written", "docs"]);
    expect(results[0]).toMatchObject({ ok: false });
    expect(results[0]?.error).toMatch(/requires 'url'/);
    expect(results[1]).toMatchObject({ ok: true });
  });
});

describe("kontext-ontology CLI surface", () => {
  it("lists configured sources and says so when there are none", async () => {
    const empty = makeConfig("mcp: []");
    expect((await run(["list", "--config", empty])).printed).toContain("No sources configured");

    const config = makeConfig(localMcp("/repo"));
    const { code, printed } = await run(["list", "--config", config]);
    expect(code).toBe(0);
    expect(printed).toContain("repo-docs");
    expect(printed).toContain("local");
  });

  it("emits JSON a surface can render per source", async () => {
    const config = makeConfig(localMcp("/repo"));
    const { printed } = await run(["list", "--config", config, "--json"]);
    expect(JSON.parse(printed)).toEqual({
      command: "list",
      ok: true,
      sources: [
        { name: "repo-docs", transport: "local", type: null, target: "/repo", code: false },
      ],
      embedding: {
        provider: "builtin",
        model: "Xenova/multilingual-e5-small",
        baseUrl: null,
        apiKeyEnv: null,
      },
    });
  });

  it("records the search embedding choice in kontext.yaml and reports it on list", async () => {
    const config = makeConfig(localMcp("/repo"));
    const refused = await run(["embedding", "--config", config, "--provider", "bogus", "--json"]);
    expect(refused.code).toBe(1);
    expect(refused.printed).toMatch(/--provider must be builtin, ollama, openai or none/);
    const chosen = await run([
      "embedding",
      "--config",
      config,
      "--provider",
      "ollama",
      "--model",
      "bge-m3",
      "--write",
      "--json",
    ]);
    expect(chosen.code).toBe(0);
    expect(JSON.parse(chosen.printed)).toEqual({
      command: "embedding",
      ok: true,
      written: true,
      embedding: {
        provider: "ollama",
        model: "bge-m3",
        baseUrl: "http://127.0.0.1:11434",
        apiKeyEnv: null,
      },
    });
    expect(parse(readFileSync(config, "utf8")).embedding).toEqual({
      provider: "ollama",
      model: "bge-m3",
    });
    const listed = await run(["list", "--config", config, "--json"]);
    expect(JSON.parse(listed.printed).embedding.model).toBe("bge-m3");
    const off = await run(["embedding", "--config", config, "--provider", "none", "--write"]);
    expect(off.printed).toContain("none (lexical search only)");
  });

  it("adds a source directly and writes only when asked", async () => {
    const config = makeConfig("mcp: []");
    const preview = await run([
      "add",
      "--config",
      config,
      "--name",
      "notion",
      "--transport",
      "sse",
      "--url",
      "https://mcp.notion.invalid",
      "--type",
      "notion",
    ]);
    expect(preview.code).toBe(0);
    expect(parse(readFileSync(config, "utf8")).mcp).toEqual([]);

    const saved = await run([
      "add",
      "--config",
      config,
      "--name",
      "notion",
      "--transport",
      "sse",
      "--url",
      "https://mcp.notion.invalid",
      "--type",
      "notion",
      "--write",
    ]);
    expect(saved.code).toBe(0);
    expect(parse(readFileSync(config, "utf8")).mcp).toEqual([
      { name: "notion", transport: "sse", url: "https://mcp.notion.invalid", type: "notion" },
    ]);
  });

  it("fails with a usable message when add is missing its required flags", async () => {
    const config = makeConfig("mcp: []");
    const { code, printed } = await run(["add", "--config", config, "--name", "x"]);
    expect(code).toBe(1);
    expect(printed).toContain("--transport");
  });

  it("reports a duplicate name as a failure rather than a silent replace", async () => {
    const config = makeConfig(localMcp("/repo"));
    const { code, printed } = await run([
      "add",
      "--config",
      config,
      "--name",
      "repo-docs",
      "--transport",
      "local",
      "--path",
      "/other",
      "--write",
    ]);
    expect(code).toBe(1);
    expect(printed).toContain("already exists");
    expect(parse(readFileSync(config, "utf8")).mcp[0].path).toBe("/repo");
  });

  it("rejects an unrecognised --from instead of scanning nothing", async () => {
    const config = makeConfig("mcp: []");
    const { code, printed } = await run(["import-mcp", "--config", config, "--from", "codexx"]);
    expect(code).toBe(1);
    expect(printed).toContain("Unknown --from");
  });

  it("exits non-zero when a checked source is unreachable", async () => {
    const config = makeConfig(
      "mcp:\n  - {name: broken, transport: stdio, command: /nonexistent/server}",
    );
    const { code, printed } = await run(["check", "--config", config, "--json"]);
    expect(code).toBe(1);
    expect(JSON.parse(printed)).toMatchObject({ command: "check", ok: false });
  });
});

// --- git transport -----------------------------------------------------------

import { execFileSync } from "node:child_process";
import { readMCPEntries } from "../src/kontext-config-file.js";

function gitIn(cwd: string, args: readonly string[]): void {
  execFileSync("git", [...args], {
    cwd,
    stdio: "ignore",
    env: {
      ...process.env,
      GIT_AUTHOR_NAME: "Fixture",
      GIT_AUTHOR_EMAIL: "fixture@example.invalid",
      GIT_COMMITTER_NAME: "Fixture",
      GIT_COMMITTER_EMAIL: "fixture@example.invalid",
    },
  });
}

/** A bare "remote" plus a seed clone that pushes Markdown to it. */
function makeRemoteRepository(): { url: string; seed: string } {
  const root = mkdtempSync(join(tmpdir(), "kontext-git-source-"));
  roots.push(root);
  const bare = join(root, "remote.git");
  const seed = join(root, "seed");
  mkdirSync(bare);
  gitIn(bare, ["init", "--quiet", "--bare", "--initial-branch=main"]);
  mkdirSync(join(seed, "docs"), { recursive: true });
  gitIn(seed, ["init", "--quiet", "--initial-branch=main"]);
  writeFileSync(join(seed, "docs", "decisions.md"), "# Decisions\n\nRetry twice.\n");
  gitIn(seed, ["add", "."]);
  gitIn(seed, ["commit", "--quiet", "-m", "seed"]);
  gitIn(seed, ["remote", "add", "origin", bare]);
  gitIn(seed, ["push", "--quiet", "origin", "main"]);
  return { url: `file://${bare}`, seed };
}

describe("git sources", () => {
  it("adds a repository by URL and records the ref and layer", () => {
    const config = makeConfig("mcp: []");
    const next = addSource(readConfigDocument(config), {
      name: "handbook",
      transport: "git",
      url: "https://example.invalid/team/handbook.git",
      ref: "main",
      include: ["docs"],
    });
    expect(readMCPEntries(next)).toEqual([
      {
        name: "handbook",
        transport: "git",
        url: "https://example.invalid/team/handbook.git",
        ref: "main",
        include: ["docs"],
      },
    ]);
    expect(summarizeSources(next)).toEqual([
      {
        name: "handbook",
        transport: "git",
        type: null,
        target: "https://example.invalid/team/handbook.git (main)",
        code: false,
      },
    ]);
  });

  it("refuses a git source without a repository URL", () => {
    const config = makeConfig("mcp: []");
    expect(() =>
      addSource(readConfigDocument(config), { name: "handbook", transport: "git" }),
    ).toThrow(/repository URL/);
  });

  it("clones the repository, reads its Markdown, and picks up later commits", async () => {
    const cache = mkdtempSync(join(tmpdir(), "kontext-git-cache-"));
    roots.push(cache);
    vi.stubEnv("KONTEXT_GIT_SOURCE_CACHE", cache);
    const remote = makeRemoteRepository();
    const config = makeConfig(
      ["mcp:", "  - name: handbook", "    transport: git", `    url: ${remote.url}`].join("\n"),
    );

    const first = await checkSources(readConfigDocument(config));
    expect(first).toEqual([
      { name: "handbook", ok: true, resourceCount: 1, toolCount: null, error: null },
    ]);

    writeFileSync(join(remote.seed, "docs", "terms.md"), "# Terms\n\nEstablished term.\n");
    gitIn(remote.seed, ["add", "."]);
    gitIn(remote.seed, ["commit", "--quiet", "-m", "terms"]);
    gitIn(remote.seed, ["push", "--quiet", "origin", "main"]);

    // Why: a stale checkout would build the ontology from documents the team has
    // since changed, so a second check must see the new file without any reset.
    const second = await checkSources(readConfigDocument(config));
    expect(second).toEqual([
      { name: "handbook", ok: true, resourceCount: 2, toolCount: null, error: null },
    ]);
  });

  it("reports an unreachable repository as that source's failure, not a crash", async () => {
    const cache = mkdtempSync(join(tmpdir(), "kontext-git-cache-"));
    roots.push(cache);
    vi.stubEnv("KONTEXT_GIT_SOURCE_CACHE", cache);
    const config = makeConfig(
      [
        "mcp:",
        "  - name: missing",
        "    transport: git",
        `    url: file://${join(cache, "does-not-exist.git")}`,
      ].join("\n"),
    );
    const results = await checkSources(readConfigDocument(config));
    expect(results).toHaveLength(1);
    expect(results[0]).toMatchObject({ name: "missing", ok: false, resourceCount: null });
    expect(results[0]?.error).toMatch(/git clone failed/);
  });

  it("adds an HTTP server with a header and maps its tools through the CLI", async () => {
    const config = makeConfig("mcp: []");
    const added = await run([
      "add",
      "--config",
      config,
      "--name",
      "pages",
      "--transport",
      "http",
      "--url",
      "https://example.invalid/mcp",
      "--header",
      "Authorization=Bearer ${QA_TOKEN}",
      "--write",
      "--json",
    ]);
    expect(added.code).toBe(0);
    const mapped = await run([
      "map",
      "--config",
      config,
      "--name",
      "pages",
      "--list-tool",
      "search_pages",
      "--items",
      "data.pages",
      "--id",
      "id",
      "--read-tool",
      "read_page",
      "--read-arg",
      "page_id",
      "--content",
      "text",
      "--write",
      "--json",
    ]);
    expect(mapped.code).toBe(0);
    const entry = readMCPEntries(readConfigDocument(config))[0];
    expect(entry).toMatchObject({
      transport: "http",
      url: "https://example.invalid/mcp",
      headers: { Authorization: "Bearer ${QA_TOKEN}" },
      documents: {
        list: { tool: "search_pages", items: "data.pages", id: "id" },
        read: { tool: "read_page", idArgument: "page_id", content: "text" },
      },
    });
  });

  it("adds a git source and a stdio environment through the CLI", async () => {
    const config = makeConfig("mcp: []");
    const added = await run([
      "add",
      "--config",
      config,
      "--name",
      "handbook",
      "--transport",
      "git",
      "--url",
      "https://example.invalid/team/handbook.git",
      "--ref",
      "release",
      "--write",
      "--json",
    ]);
    expect(added.code).toBe(0);
    const withEnv = await run([
      "add",
      "--config",
      config,
      "--name",
      "github",
      "--transport",
      "stdio",
      "--command",
      "npx",
      "--arg",
      "-y",
      "--arg",
      "@modelcontextprotocol/server-github",
      "--env",
      "GITHUB_PERSONAL_ACCESS_TOKEN=ghp_example",
      "--env",
      "GITHUB_API_URL=https://ghe.example.invalid/api/v3",
      "--write",
      "--json",
    ]);
    expect(withEnv.code).toBe(0);
    const entries = parse(readFileSync(config, "utf8")).mcp;
    expect(entries).toEqual([
      {
        name: "handbook",
        transport: "git",
        url: "https://example.invalid/team/handbook.git",
        ref: "release",
      },
      {
        name: "github",
        transport: "stdio",
        command: "npx",
        args: ["-y", "@modelcontextprotocol/server-github"],
        env: {
          GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_example",
          GITHUB_API_URL: "https://ghe.example.invalid/api/v3",
        },
      },
    ]);
  });

  it("rejects a malformed --env instead of storing half a variable", async () => {
    const config = makeConfig("mcp: []");
    const result = await run([
      "add",
      "--config",
      config,
      "--name",
      "github",
      "--transport",
      "stdio",
      "--command",
      "npx",
      "--env",
      "NO_EQUALS_SIGN",
      "--json",
    ]);
    expect(result.code).not.toBe(0);
    expect(result.printed).toMatch(/--env needs KEY=VALUE/);
  });
});

describe("kontext-ontology github-repos", () => {
  const listing = {
    owner: "modapl",
    kind: "organization" as const,
    repositories: [
      {
        name: "handbook",
        fullName: "modapl/handbook",
        url: "https://github.com/modapl/handbook",
        cloneUrl: "https://github.com/modapl/handbook.git",
        defaultBranch: "main",
        private: true,
        archived: false,
        fork: false,
        language: "TypeScript",
        description: null,
        pushedAt: "2026-09-01T00:00:00Z",
      },
    ],
  };

  it("lists an owner's repositories as JSON without touching any config", async () => {
    const owners: string[] = [];
    const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
    const code = await runOntologyCli(
      [
        "github-repos",
        "--owner",
        "https://github.com/modapl",
        "--json",
        "--config",
        "/nonexistent/kontext.yaml",
      ],
      {
        listRepositories: async (owner) => {
          owners.push(owner);
          return listing;
        },
      },
    );
    const printed = out.mock.calls.map((call) => String(call[0])).join("");
    out.mockRestore();
    expect(code).toBe(0);
    expect(owners).toEqual(["https://github.com/modapl"]);
    expect(JSON.parse(printed)).toEqual({ command: "github-repos", ok: true, ...listing });
  });

  it("needs an owner", async () => {
    const { code, printed } = await run(["github-repos", "--json"]);
    expect(code).toBe(1);
    expect(JSON.parse(printed)).toMatchObject({
      ok: false,
      error: expect.stringContaining("--owner"),
    });
  });
});
