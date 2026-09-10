import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import {
  StdioClientTransport,
  getDefaultEnvironment,
} from "@modelcontextprotocol/sdk/client/stdio.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import type { MCPToolAccess, MCPToolSummary, ToolCallOutcome } from "./tool-driven-connector.js";

export interface MCPResource {
  readonly id: string;
  readonly name: string;
  readonly description: string;
  readonly mimeType?: string | null;
}

export interface MCPData {
  readonly resourceId: string;
  readonly content: string;
  readonly metadata: Readonly<Record<string, string>>;
  readonly fetchedAt: Date;
}

/**
 * MCP client port. New connector implementations only need to implement this.
 */
export interface MCPConnector {
  readonly name: string;
  listResources(): Promise<MCPResource[]>;
  fetchResource(resourceId: string): Promise<MCPData>;
  search(query: string): Promise<MCPData[]>;
}

/** Tools and resources through the SDK client, shared by every transport. */
async function listToolsThrough(client: Client): Promise<readonly MCPToolSummary[]> {
  const result = await client.listTools();
  return result.tools.map((tool) => ({
    name: tool.name,
    description: tool.description ?? "",
    inputSchema: tool.inputSchema,
  }));
}

async function callToolThrough(
  client: Client,
  name: string,
  args: Readonly<Record<string, unknown>>,
): Promise<ToolCallOutcome> {
  const result = await client.callTool({ name, arguments: { ...args } });
  const blocks = Array.isArray(result.content) ? result.content : [];
  const text = blocks
    .map((block) => ("text" in block && typeof block.text === "string" ? block.text : ""))
    .filter((part) => part !== "")
    .join("\n");
  let data: unknown = result.structuredContent;
  if (data === undefined) {
    // Why: most servers return JSON as a text block; parsed, it can be walked by path.
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }
  return { text, data, isError: result.isError === true };
}

function resourcesThrough(result: Awaited<ReturnType<Client["listResources"]>>): MCPResource[] {
  return result.resources.map((r) => ({
    id: r.uri,
    name: r.name,
    description: r.description ?? "",
    mimeType: r.mimeType ?? null,
  }));
}

/** Header values may name environment variables as `${NAME}`; a token then never sits in a file. */
export function resolveHeaderValues(
  headers: Readonly<Record<string, string>> | undefined,
  env: NodeJS.ProcessEnv = process.env,
): Record<string, string> {
  const resolved: Record<string, string> = {};
  for (const [key, value] of Object.entries(headers ?? {})) {
    resolved[key] = value.replace(
      /\$\{([A-Za-z_][A-Za-z0-9_]*)\}/g,
      (_, name: string) => env[name] ?? "",
    );
  }
  return resolved;
}

/**
 * Connects to an MCP server over stdio (spawning a subprocess).
 * Standard transport for local MCP servers.
 */
export class StdioMCPConnector implements MCPConnector, MCPToolAccess {
  private client: Client | null = null;
  private readyPromise: Promise<void> | null = null;
  // Why: closing through the client only tears down a transport that finished
  // connecting. A server that never answers would otherwise outlive this process
  // as an orphan holding its pipes, one per failed probe.
  private transport: StdioClientTransport | null = null;

  constructor(
    public readonly name: string,
    private readonly command: string,
    private readonly args: readonly string[] = [],
    private readonly env?: Record<string, string>,
  ) {}

  private async ensureConnected(): Promise<Client> {
    if (this.client) return this.client;
    if (!this.readyPromise) {
      this.readyPromise = this.connect();
    }
    await this.readyPromise;
    if (!this.client) throw new Error("MCP client failed to connect");
    return this.client;
  }

  private async connect(): Promise<void> {
    const transport = new StdioClientTransport({
      command: this.command,
      args: [...this.args],
      // Why: the SDK replaces the child environment when `env` is given rather than
      // extending it, so passing a server's own few variables alone would strip PATH
      // and the command would not resolve.
      env: this.env ? { ...getDefaultEnvironment(), ...this.env } : undefined,
    });
    this.transport = transport;
    const client = new Client(
      { name: `kontext-client-${this.name}`, version: "0.1.0" },
      { capabilities: {} },
    );
    await client.connect(transport);
    this.client = client;
  }

  async listResources(): Promise<MCPResource[]> {
    const client = await this.ensureConnected();
    const result = await client.listResources();
    return result.resources.map((r) => ({
      id: r.uri,
      name: r.name,
      description: r.description ?? "",
      mimeType: r.mimeType ?? null,
    }));
  }

  async fetchResource(resourceId: string): Promise<MCPData> {
    const client = await this.ensureConnected();
    const result = await client.readResource({ uri: resourceId });
    const text = result.contents
      .map((c) => ("text" in c && typeof c.text === "string" ? c.text : ""))
      .join("\n");
    return {
      resourceId,
      content: text,
      metadata: {},
      fetchedAt: new Date(),
    };
  }

  async listTools(): Promise<readonly MCPToolSummary[]> {
    return listToolsThrough(await this.ensureConnected());
  }

  async callTool(name: string, args: Readonly<Record<string, unknown>>): Promise<ToolCallOutcome> {
    return callToolThrough(await this.ensureConnected(), name, args);
  }

  async search(_query: string): Promise<MCPData[]> {
    // Standard MCP doesn't define a search method for resources; return empty.
    return [];
  }

  async close(): Promise<void> {
    const { client, transport } = this;
    this.client = null;
    this.transport = null;
    this.readyPromise = null;
    if (client) {
      await client.close();
      return;
    }
    // The handshake never finished, so the spawned server is still running.
    await transport?.close();
  }
}

/**
 * Connects to an MCP server over SSE (HTTP).
 * Use for remote MCP servers.
 */
export class SseMCPConnector implements MCPConnector, MCPToolAccess {
  private client: Client | null = null;

  constructor(
    public readonly name: string,
    private readonly url: string,
    private readonly headers: Readonly<Record<string, string>> = {},
  ) {}

  private async ensureConnected(): Promise<Client> {
    if (this.client) return this.client;
    const headers = resolveHeaderValues(this.headers);
    const transport = new SSEClientTransport(new URL(this.url), {
      requestInit: { headers },
      // Why: the event stream is opened with its own fetch, which does not see requestInit.
      eventSourceInit: {
        fetch: (input, init) =>
          fetch(input, { ...init, headers: { ...headers, ...(init?.headers as object) } }),
      },
    });
    const client = new Client(
      { name: `kontext-client-${this.name}`, version: "0.1.0" },
      { capabilities: {} },
    );
    await client.connect(transport);
    this.client = client;
    return client;
  }

  async listResources(): Promise<MCPResource[]> {
    const client = await this.ensureConnected();
    const result = await client.listResources();
    return result.resources.map((r) => ({
      id: r.uri,
      name: r.name,
      description: r.description ?? "",
      mimeType: r.mimeType ?? null,
    }));
  }

  async fetchResource(resourceId: string): Promise<MCPData> {
    const client = await this.ensureConnected();
    const result = await client.readResource({ uri: resourceId });
    const text = result.contents
      .map((c) => ("text" in c && typeof c.text === "string" ? c.text : ""))
      .join("\n");
    return {
      resourceId,
      content: text,
      metadata: {},
      fetchedAt: new Date(),
    };
  }

  async listTools(): Promise<readonly MCPToolSummary[]> {
    return listToolsThrough(await this.ensureConnected());
  }

  async callTool(name: string, args: Readonly<Record<string, unknown>>): Promise<ToolCallOutcome> {
    return callToolThrough(await this.ensureConnected(), name, args);
  }

  async search(_query: string): Promise<MCPData[]> {
    return [];
  }

  async close(): Promise<void> {
    if (this.client) {
      await this.client.close();
      this.client = null;
    }
  }
}

/**
 * Connects over Streamable HTTP, the current remote MCP transport. Hosted
 * servers (Notion, Linear, GitHub) speak this; a bearer token or API key goes
 * in `headers`, with `${ENV_NAME}` values read from the environment.
 */
export class HttpMCPConnector implements MCPConnector, MCPToolAccess {
  private client: Client | null = null;

  constructor(
    public readonly name: string,
    private readonly url: string,
    private readonly headers: Readonly<Record<string, string>> = {},
  ) {}

  private async ensureConnected(): Promise<Client> {
    if (this.client) return this.client;
    const transport = new StreamableHTTPClientTransport(new URL(this.url), {
      requestInit: { headers: resolveHeaderValues(this.headers) },
    });
    const client = new Client(
      { name: `kontext-client-${this.name}`, version: "0.1.0" },
      { capabilities: {} },
    );
    await client.connect(transport);
    this.client = client;
    return client;
  }

  async listResources(): Promise<MCPResource[]> {
    const client = await this.ensureConnected();
    try {
      return resourcesThrough(await client.listResources());
    } catch (error) {
      // Why: a tools-only server answers "method not found"; that is no documents, not a failure.
      if (String(error).includes("-32601")) return [];
      throw error;
    }
  }

  async fetchResource(resourceId: string): Promise<MCPData> {
    const client = await this.ensureConnected();
    const result = await client.readResource({ uri: resourceId });
    const text = result.contents
      .map((c) => ("text" in c && typeof c.text === "string" ? c.text : ""))
      .join("\n");
    return { resourceId, content: text, metadata: {}, fetchedAt: new Date() };
  }

  async listTools(): Promise<readonly MCPToolSummary[]> {
    return listToolsThrough(await this.ensureConnected());
  }

  async callTool(name: string, args: Readonly<Record<string, unknown>>): Promise<ToolCallOutcome> {
    return callToolThrough(await this.ensureConnected(), name, args);
  }

  async search(_query: string): Promise<MCPData[]> {
    return [];
  }

  async close(): Promise<void> {
    if (this.client) {
      await this.client.close();
      this.client = null;
    }
  }
}

// ── Cache ─────────────────────────────────────────────────────

export class MCPDataCache {
  private readonly cache = new Map<string, { data: MCPData; cachedAt: number }>();

  constructor(private readonly ttlSeconds = 300) {}

  get(key: string): MCPData | null {
    const entry = this.cache.get(key);
    if (!entry) return null;
    const ageSec = (Date.now() - entry.cachedAt) / 1000;
    if (ageSec >= this.ttlSeconds) {
      this.cache.delete(key);
      return null;
    }
    return entry.data;
  }

  put(key: string, data: MCPData): void {
    this.cache.set(key, { data, cachedAt: Date.now() });
  }
}
