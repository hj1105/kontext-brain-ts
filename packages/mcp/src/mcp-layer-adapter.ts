import {
  type ContentFetcher,
  DataSource,
  type DocumentContent,
  type MetaDocument,
  createMetaDocument,
} from "@kontext-brain/core";
import type { MCPConnector, MCPResource } from "./mcp-connector.js";

/**
 * Connects an MCP source to L2 (meta) and L3 (content) layers of kontext.
 * Source-specific subclasses (Notion/Jira/...) inject extractors.
 */
export interface MCPLayerAdapter {
  readonly dataSource: DataSource;
  readonly connectorName: string;
  listMeta(ontologyNodeId: string): Promise<MetaDocument[]>;
  fetchContent(metaDoc: MetaDocument): Promise<string>;
}

/** Bridges MCPLayerAdapter to ContentFetcher interface. */
export class MCPContentFetcherBridge implements ContentFetcher {
  readonly source: DataSource;
  constructor(private readonly adapter: MCPLayerAdapter) {
    this.source = adapter.dataSource;
  }

  async fetch(metaDoc: MetaDocument): Promise<DocumentContent> {
    let body: string;
    try {
      body = await this.adapter.fetchContent(metaDoc);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      body = `Warning: fetch failed: ${msg}`;
    }
    return {
      metaDocumentId: metaDoc.id,
      title: metaDoc.title,
      body,
      source: this.adapter.dataSource,
      sectionContent: null,
      fetchedAt: new Date(),
    };
  }
}

// ── Generic adapter + factories ───────────────────────────────

export interface GenericAdapterOptions {
  titleExtractor?: (resource: MCPResource) => string;
  urlExtractor?: (resource: MCPResource) => string | null;
  metadataExtractor?: (resource: MCPResource) => Record<string, string>;
}

export class GenericMCPLayerAdapter implements MCPLayerAdapter {
  private readonly titleExtractor: (resource: MCPResource) => string;
  private readonly urlExtractor: (resource: MCPResource) => string | null;
  private readonly metadataExtractor: (resource: MCPResource) => Record<string, string>;

  constructor(
    public readonly dataSource: DataSource,
    public readonly connectorName: string,
    private readonly connector: MCPConnector,
    options: GenericAdapterOptions = {},
  ) {
    this.titleExtractor = options.titleExtractor ?? ((r) => r.name);
    this.urlExtractor = options.urlExtractor ?? (() => null);
    this.metadataExtractor = options.metadataExtractor ?? (() => ({}));
  }

  async listMeta(ontologyNodeId: string): Promise<MetaDocument[]> {
    try {
      const resources = await this.connector.listResources();
      return resources.map((r) =>
        createMetaDocument({
          id: r.id,
          title: this.titleExtractor(r),
          source: this.dataSource,
          ontologyNodeId,
          url: this.urlExtractor(r),
          metadata: this.metadataExtractor(r),
        }),
      );
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.warn(
        `Warning: MCPLayerAdapter '${this.connectorName}' listMeta failed: ${msg}`,
      );
      return [];
    }
  }

  async fetchContent(metaDoc: MetaDocument): Promise<string> {
    const data = await this.connector.fetchResource(metaDoc.id);
    return data.content;
  }
}

export const MCPLayerAdapterFactory = {
  notion(connector: MCPConnector): GenericMCPLayerAdapter {
    return new GenericMCPLayerAdapter(DataSource.NOTION, connector.name, connector, {
      titleExtractor: (r) => r.name,
      urlExtractor: (r) => (r.description.startsWith("http") ? r.description : null),
    });
  },

  jira(connector: MCPConnector): GenericMCPLayerAdapter {
    return new GenericMCPLayerAdapter(DataSource.JIRA, connector.name, connector, {
      metadataExtractor: (r) => {
        const meta: Record<string, string> = {};
        for (const part of r.description.split("|")) {
          const kv = part.trim().split(":");
          if (kv.length === 2 && kv[0] && kv[1]) meta[kv[0].trim()] = kv[1].trim();
        }
        return meta;
      },
    });
  },

  githubPr(connector: MCPConnector): GenericMCPLayerAdapter {
    return new GenericMCPLayerAdapter(DataSource.GITHUB_PR, connector.name, connector, {
      titleExtractor: (r) => {
        const idx = r.id.lastIndexOf("/");
        const num = idx >= 0 ? r.id.slice(idx + 1) : r.id;
        return `PR #${num}: ${r.name}`;
      },
      urlExtractor: (r) => (r.description.includes("github.com") ? r.description : null),
    });
  },

  slack(connector: MCPConnector): GenericMCPLayerAdapter {
    return new GenericMCPLayerAdapter(DataSource.SLACK, connector.name, connector, {
      titleExtractor: (r) => `#${r.name}`,
    });
  },
};
