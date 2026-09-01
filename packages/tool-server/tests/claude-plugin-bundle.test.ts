import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { resolvePluginDataDirectory } from "@kontext-brain/local";
import { describe, expect, it } from "vitest";

const pluginRoot = path.resolve(
  fileURLToPath(new URL("../../../plugins/kontext-brain", import.meta.url)),
);

async function readJson(relativePath: string): Promise<Record<string, unknown>> {
  return JSON.parse(await readFile(path.join(pluginRoot, relativePath), "utf8"));
}

describe("Claude plugin bundle", () => {
  it("declares the MCP server without an environment variable the server ignores", async () => {
    const config = (await readJson(".claude.mcp.json")) as {
      mcpServers: Record<string, { command: string; args: string[]; env?: Record<string, string> }>;
    };
    const server = config.mcpServers.kontext_brain;
    expect(server).toBeDefined();
    expect(server?.args).toEqual([expect.stringContaining("server.mjs")]);

    // resolvePluginDataDirectory only reads these names. Declaring anything
    // else looks like it scopes the private data directory while the server
    // silently falls back to the shared user-level location.
    const recognized = ["KONTEXT_PLUGIN_DATA", "PLUGIN_DATA", "CLAUDE_PLUGIN_DATA"];
    for (const name of Object.keys(server?.env ?? {})) {
      expect(recognized).toContain(name);
    }
  });

  it("keeps an explicit private data directory authoritative", () => {
    expect(
      resolvePluginDataDirectory({ KONTEXT_PLUGIN_DATA: "/tmp/kontext-private" }, "/home/user"),
    ).toBe(path.resolve("/tmp/kontext-private"));
    // KONTEXT_PLUGIN_ROOT is not one of the recognized names.
    expect(
      resolvePluginDataDirectory(
        { KONTEXT_PLUGIN_ROOT: "/plugins/kontext" },
        "/home/user",
        "linux",
      ),
    ).toBe(path.resolve("/home/user/.local/share/kontext-brain"));
  });

  it("publishes the repository license in both plugin manifests", async () => {
    for (const manifest of [".claude-plugin/plugin.json", ".codex-plugin/plugin.json"]) {
      expect((await readJson(manifest)).license).toBe("Apache-2.0");
    }
  });
});
