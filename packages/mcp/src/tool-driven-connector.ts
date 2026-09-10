import type { MCPConnector, MCPData, MCPResource } from "./mcp-connector.js";

/**
 * Most MCP servers expose tools, not resources: GitHub, Notion, Linear, Slack
 * answer `tools/list`, and `resources/list` is empty. A mapping names the tool
 * that lists documents and the tool that reads one, with paths into their
 * results, so any such server becomes a document source without code changes.
 */

export interface ToolDocumentMapping {
  readonly list: {
    readonly tool: string;
    /** Fixed arguments for the listing call, e.g. a query or a database id. */
    readonly arguments?: Readonly<Record<string, unknown>>;
    /** Path to the array of items in the result; empty when the result is the array. */
    readonly items?: string;
    /** Paths inside one item. */
    readonly id: string;
    readonly title?: string;
    readonly description?: string;
  };
  readonly read: {
    readonly tool: string;
    /** The argument that receives the document id. */
    readonly idArgument: string;
    readonly arguments?: Readonly<Record<string, unknown>>;
    /** Path to the text in the result; empty to use the tool's text blocks as they are. */
    readonly content?: string;
  };
}

export interface MCPToolSummary {
  readonly name: string;
  readonly description: string;
  readonly inputSchema: unknown;
}

export interface MCPToolAccess {
  listTools(): Promise<readonly MCPToolSummary[]>;
  callTool(name: string, args: Readonly<Record<string, unknown>>): Promise<ToolCallOutcome>;
}

export interface ToolCallOutcome {
  /** Text blocks the tool returned, in order. */
  readonly text: string;
  /** Structured content when the server provides it; otherwise the text parsed as JSON when it is JSON. */
  readonly data: unknown;
  readonly isError: boolean;
}

export function hasToolAccess(connector: unknown): connector is MCPToolAccess {
  const candidate = connector as Partial<MCPToolAccess> | null;
  return typeof candidate?.listTools === "function" && typeof candidate?.callTool === "function";
}

/** `a.b[0].c` into `value`; `undefined` when any step is missing. */
export function readPath(value: unknown, path: string | undefined): unknown {
  if (!path || path.trim() === "" || path === "$") return value;
  let current: unknown = value;
  for (const segment of path.replace(/^\$\.?/, "").split(".")) {
    if (segment === "") continue;
    const match = /^([^[\]]*)((?:\[\d+\])*)$/.exec(segment);
    if (!match) return undefined;
    const [, key, indexes] = match;
    if (key) {
      if (typeof current !== "object" || current === null) return undefined;
      current = (current as Record<string, unknown>)[key];
    }
    for (const index of indexes?.match(/\d+/g) ?? []) {
      if (!Array.isArray(current)) return undefined;
      current = current[Number(index)];
    }
  }
  return current;
}

function asText(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}

export class ToolDrivenMCPConnector implements MCPConnector, MCPToolAccess {
  readonly name: string;

  constructor(
    private readonly base: MCPConnector & MCPToolAccess,
    private readonly mapping: ToolDocumentMapping,
  ) {
    this.name = base.name;
  }

  async listResources(): Promise<MCPResource[]> {
    const outcome = await this.base.callTool(
      this.mapping.list.tool,
      this.mapping.list.arguments ?? {},
    );
    if (outcome.isError) {
      throw new Error(
        `${this.name}: ${this.mapping.list.tool} failed: ${outcome.text.slice(0, 300)}`,
      );
    }
    const items = readPath(outcome.data, this.mapping.list.items);
    if (!Array.isArray(items)) {
      throw new Error(
        `${this.name}: ${this.mapping.list.tool} returned no array at "${this.mapping.list.items ?? "$"}"`,
      );
    }
    const resources: MCPResource[] = [];
    for (const item of items) {
      const id = asText(readPath(item, this.mapping.list.id));
      if (id === "") continue;
      // Why: an absent path means "not provided", not "the whole item".
      const title = this.mapping.list.title ? asText(readPath(item, this.mapping.list.title)) : "";
      const description = this.mapping.list.description
        ? asText(readPath(item, this.mapping.list.description))
        : "";
      resources.push({ id, name: title || id, description, mimeType: null });
    }
    return resources;
  }

  async fetchResource(resourceId: string): Promise<MCPData> {
    const outcome = await this.base.callTool(this.mapping.read.tool, {
      ...(this.mapping.read.arguments ?? {}),
      [this.mapping.read.idArgument]: resourceId,
    });
    if (outcome.isError) {
      throw new Error(
        `${this.name}: ${this.mapping.read.tool} failed: ${outcome.text.slice(0, 300)}`,
      );
    }
    const content = this.mapping.read.content
      ? asText(readPath(outcome.data, this.mapping.read.content))
      : outcome.text;
    return {
      resourceId,
      content,
      metadata: { source: this.name, tool: this.mapping.read.tool },
      fetchedAt: new Date(),
    };
  }

  async search(_query: string): Promise<MCPData[]> {
    return [];
  }

  // Why: a mapped server is still a tool server; inspection and checks must see its tools.
  listTools(): Promise<readonly MCPToolSummary[]> {
    return this.base.listTools();
  }

  callTool(name: string, args: Readonly<Record<string, unknown>>): Promise<ToolCallOutcome> {
    return this.base.callTool(name, args);
  }

  async close(): Promise<void> {
    const close = (this.base as { close?: () => Promise<void> }).close;
    if (typeof close === "function") await close.call(this.base);
  }
}
