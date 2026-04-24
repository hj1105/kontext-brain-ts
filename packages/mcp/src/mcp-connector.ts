import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";

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

/**
 * Connects to an MCP server over stdio (spawning a subprocess).
 * Standard transport for local MCP servers.
 */
export class StdioMCPConnector implements MCPConnector {
  private client: Client | null = null;
  private readyPromise: Promise<void> | null = null;

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
      env: this.env,
    });
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

  async search(_query: string): Promise<MCPData[]> {
    // Standard MCP doesn't define a search method for resources; return empty.
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
 * Connects to an MCP server over SSE (HTTP).
 * Use for remote MCP servers.
 */
export class SseMCPConnector implements MCPConnector {
  private client: Client | null = null;

  constructor(
    public readonly name: string,
    private readonly url: string,
  ) {}

  private async ensureConnected(): Promise<Client> {
    if (this.client) return this.client;
    const transport = new SSEClientTransport(new URL(this.url));
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
