import { describe, expect, it } from "vitest";
import {
  type MCPConnector,
  type MCPToolAccess,
  ToolDrivenMCPConnector,
  hasToolAccess,
  readPath,
  resolveHeaderValues,
} from "../src/index.js";

/** A GitHub-like server: tools only, JSON in text blocks. */
class ToolsOnlyServer implements MCPConnector, MCPToolAccess {
  readonly name = "issues";
  readonly calls: Array<{ name: string; args: Record<string, unknown> }> = [];
  async listResources() {
    return [];
  }
  async fetchResource(): Promise<never> {
    throw new Error("resources/read is not supported");
  }
  async search() {
    return [];
  }
  async listTools() {
    return [
      { name: "list_issues", description: "List issues", inputSchema: {} },
      { name: "get_issue", description: "Read one issue", inputSchema: {} },
    ];
  }
  async callTool(name: string, args: Readonly<Record<string, unknown>>) {
    this.calls.push({ name, args: { ...args } });
    if (name === "list_issues") {
      const body = {
        items: [
          { number: 7, title: "Retry payments twice", body: "…" },
          { number: 9, title: "Round half up", body: "…" },
        ],
      };
      return { text: JSON.stringify(body), data: body, isError: false };
    }
    if (name === "get_issue") {
      const body = { number: args.number, body: `Decision for issue ${String(args.number)}` };
      return { text: JSON.stringify(body), data: body, isError: false };
    }
    return { text: "unknown tool", data: "unknown tool", isError: true };
  }
}

describe("readPath", () => {
  it("walks dotted keys and indexes, and returns undefined for a missing step", () => {
    const value = { a: { b: [{ c: 1 }, { c: 2 }] }, list: [1, 2] };
    expect(readPath(value, "a.b[1].c")).toBe(2);
    expect(readPath(value, "$.a.b[0].c")).toBe(1);
    expect(readPath(value, "list")).toEqual([1, 2]);
    expect(readPath(value, "")).toBe(value);
    expect(readPath(value, "a.x.y")).toBeUndefined();
  });
});

describe("resolveHeaderValues", () => {
  it("fills ${NAME} from the environment so a token never sits in kontext.yaml", () => {
    expect(
      resolveHeaderValues(
        { Authorization: "Bearer ${NOTION_TOKEN}", "X-Static": "1" },
        { NOTION_TOKEN: "secret" },
      ),
    ).toEqual({ Authorization: "Bearer secret", "X-Static": "1" });
    expect(resolveHeaderValues({ A: "${MISSING}" }, {})).toEqual({ A: "" });
  });
});

describe("ToolDrivenMCPConnector", () => {
  it("lists documents through the mapped tool and reads one by id", async () => {
    const server = new ToolsOnlyServer();
    expect(hasToolAccess(server)).toBe(true);
    const connector = new ToolDrivenMCPConnector(server, {
      list: {
        tool: "list_issues",
        arguments: { state: "open" },
        items: "items",
        id: "number",
        title: "title",
      },
      read: { tool: "get_issue", idArgument: "number", content: "body" },
    });
    const resources = await connector.listResources();
    expect(resources).toEqual([
      { id: "7", name: "Retry payments twice", description: "", mimeType: null },
      { id: "9", name: "Round half up", description: "", mimeType: null },
    ]);
    expect(server.calls[0]).toEqual({ name: "list_issues", args: { state: "open" } });
    const fetched = await connector.fetchResource("7");
    expect(fetched.content).toBe("Decision for issue 7");
    expect(server.calls[1]).toEqual({ name: "get_issue", args: { number: "7" } });
    // Why: a check on a mapped server still counts its tools through the wrapper.
    expect(hasToolAccess(connector)).toBe(true);
    expect((await connector.listTools()).map((tool) => tool.name)).toEqual([
      "list_issues",
      "get_issue",
    ]);
  });

  it("names the tool and path when a listing does not yield an array", async () => {
    const connector = new ToolDrivenMCPConnector(new ToolsOnlyServer(), {
      list: { tool: "list_issues", items: "nowhere", id: "number" },
      read: { tool: "get_issue", idArgument: "number" },
    });
    await expect(connector.listResources()).rejects.toThrow(
      /list_issues returned no array at "nowhere"/,
    );
  });
});
