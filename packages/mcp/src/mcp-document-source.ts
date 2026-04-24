import type { DocumentSource, SourceDocument } from "@kontext-brain/core";
import type { MCPConnector } from "./mcp-connector.js";

/**
 * MCPConnector -> DocumentSource bridge for OntologyAutoBuilder.
 */
export class MCPDocumentSource implements DocumentSource {
  constructor(private readonly connector: MCPConnector) {}

  async collect(): Promise<SourceDocument[]> {
    try {
      const resources = await this.connector.listResources();
      return resources.map((r) => {
        const metadata: Record<string, string> = { source: this.connector.name };
        if (r.description.trim()) metadata.description = r.description;
        if (r.mimeType) metadata.mimeType = r.mimeType;
        return { id: r.id, title: r.name, metadata };
      });
    } catch {
      return [];
    }
  }
}
