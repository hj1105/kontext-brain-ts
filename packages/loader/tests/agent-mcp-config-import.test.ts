import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  discoverAgentMCPServers,
  inferLayerType,
  mergeMCPConfigs,
  parseCodexMcpServers,
} from "../src/agent-mcp-config-import.js";

const roots: string[] = [];

function makeHome(): string {
  const root = mkdtempSync(join(tmpdir(), "kontext-import-"));
  roots.push(root);
  return root;
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("inferLayerType", () => {
  it("types a source only when the whole word names a supported layer", () => {
    expect(inferLayerType("notion")).toBe("notion");
    expect(inferLayerType("company-jira")).toBe("jira");
    expect(inferLayerType("github-pr")).toBe("github_pr");
    // A neighbouring product name must not inherit an adapter that reads a
    // different document shape.
    expect(inferLayerType("notionary")).toBeUndefined();
    expect(inferLayerType("github-actions-linter")).toBeUndefined();
  });
});

describe("parseCodexMcpServers", () => {
  it("reads stdio, url and env tables", () => {
    const servers = parseCodexMcpServers(
      [
        "[mcp_servers.node_repl]",
        'command = "/bin/node_repl"',
        "args = []",
        "startup_timeout_sec = 120",
        "",
        "[mcp_servers.node_repl.env]",
        'NODE_REPL_NODE_PATH = "/bin/node"',
        "",
        "[mcp_servers.slack]",
        'url = "https://mcp.slack.com/mcp"',
        "",
        "[other_table]",
        'command = "ignored"',
      ].join("\n"),
    );
    expect(Object.keys(servers).sort()).toEqual(["node_repl", "slack"]);
    expect(servers.node_repl?.command).toBe("/bin/node_repl");
    expect(servers.node_repl?.env).toEqual({ NODE_REPL_NODE_PATH: "/bin/node" });
    expect(servers.slack?.url).toBe("https://mcp.slack.com/mcp");
  });

  it("reads a quoted string array", () => {
    const servers = parseCodexMcpServers(
      ["[mcp_servers.cu]", 'command = "client"', 'args = ["mcp", "--quiet"]'].join("\n"),
    );
    expect(servers.cu?.args).toEqual(["mcp", "--quiet"]);
  });
});

describe("discoverAgentMCPServers", () => {
  it("imports Codex stdio servers with their environment", () => {
    const home = makeHome();
    mkdirSync(join(home, ".codex"));
    writeFileSync(
      join(home, ".codex", "config.toml"),
      [
        "[mcp_servers.notion]",
        'command = "notion-mcp"',
        "",
        "[mcp_servers.notion.env]",
        'TOKEN = "abc"',
      ].join("\n"),
    );
    const found = discoverAgentMCPServers({ homeDirectory: home, include: ["codex"] });
    expect(found).toHaveLength(1);
    expect(found[0]?.config).toMatchObject({
      name: "notion",
      command: "notion-mcp",
      transport: "stdio",
      type: "notion",
      env: { TOKEN: "abc" },
    });
    expect(found[0]?.origin).toBe("codex");
  });

  it("takes only the requested project's Claude servers", () => {
    const home = makeHome();
    writeFileSync(
      join(home, ".claude.json"),
      JSON.stringify({
        projects: {
          "/work/wanted": { mcpServers: { jira: { command: "jira-mcp" } } },
          "/work/other": { mcpServers: { slack: { url: "https://example.invalid" } } },
        },
      }),
    );
    const found = discoverAgentMCPServers({
      homeDirectory: home,
      include: ["claude"],
      projectDirectory: "/work/wanted",
    });
    expect(found.map((item) => item.config.name)).toEqual(["jira"]);
    expect(found[0]?.scope).toBe("/work/wanted");
  });

  it("ignores an entry that names neither a command nor a url", () => {
    const home = makeHome();
    writeFileSync(
      join(home, ".claude.json"),
      JSON.stringify({ mcpServers: { broken: { note: "no transport" } } }),
    );
    expect(discoverAgentMCPServers({ homeDirectory: home, include: ["claude"] })).toEqual([]);
  });

  it("returns nothing rather than throwing when no configuration exists", () => {
    expect(discoverAgentMCPServers({ homeDirectory: makeHome() })).toEqual([]);
  });
});

describe("mergeMCPConfigs", () => {
  it("adds new names and never rewrites an entry the user already owns", () => {
    const existing = [{ name: "notion", command: "mine", transport: "stdio" as const }];
    const result = mergeMCPConfigs(existing, [
      { name: "notion", command: "theirs", transport: "stdio" as const },
      { name: "jira", command: "jira-mcp", transport: "stdio" as const },
    ]);
    expect(result.added).toEqual(["jira"]);
    expect(result.kept).toEqual(["notion"]);
    expect(result.merged[0]).toEqual(existing[0]);
    expect(result.merged).toHaveLength(2);
  });
});
