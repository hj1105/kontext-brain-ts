import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import type { KontextAgent } from "@kontext-brain/loader";
import { z } from "zod";

/**
 * Exposes a KontextAgent as an MCP server over stdio.
 * Compatible with Claude Desktop, Claude Code, and other MCP clients.
 */
export class KontextToolServer {
  private readonly server: McpServer;

  constructor(private readonly agent: KontextAgent) {
    this.server = new McpServer({
      name: "kontext-brain",
      version: "0.1.0",
    });
    this.registerTools();
  }

  async start(): Promise<void> {
    process.stderr.write("kontext-brain MCP server starting (stdio mode)\n");
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
  }

  private registerTools(): void {
    this.server.tool(
      "kontext_query",
      "Query the ontology-based knowledge base. Returns a fully reasoned answer with sources.",
      { question: z.string().describe("The question to answer") },
      async ({ question }) => {
        const result = await this.agent.query(question);
        const sourceLines = result.selectedMetaDocs
          .map((d) => `- ${d.title} (${d.source})`)
          .join("\n");
        const text = `${result.answer}\n\n--- Sources ---\n${sourceLines}\nTokens used: ${result.contextTokensUsed}`;
        return { content: [{ type: "text", text }] };
      },
    );

    this.server.tool(
      "kontext_query_context",
      "Retrieve relevant context from the knowledge base WITHOUT final LLM reasoning. Use when the calling agent wants to do its own reasoning.",
      { question: z.string().describe("The question to retrieve context for") },
      async ({ question }) => {
        const result = await this.agent.query(question);
        const nodes = result.usedOntologyNodes
          .map((n) => `## ${n.id}\n${n.description}`)
          .join("\n\n");
        const docs = result.selectedMetaDocs.map((d) => `- [${d.source}] ${d.title}`).join("\n");
        const text = `=== Retrieved Context ===\n\n${nodes}\n\n${docs}\n\nTokens used: ${result.contextTokensUsed}`;
        return { content: [{ type: "text", text }] };
      },
    );

    this.server.tool(
      "kontext_ingest",
      "Ingest new data into the knowledge graph. Extracts entities and relationships automatically.",
      {
        data: z.string().describe("The text data to ingest"),
        source: z.string().optional().describe("Source identifier"),
      },
      async ({ data, source }) => {
        await this.agent.ingest(data, source ?? "manual");
        return {
          content: [
            { type: "text", text: `Data ingested successfully from source: ${source ?? "manual"}` },
          ],
        };
      },
    );

    this.server.tool(
      "kontext_describe",
      "Describe the current ontology graph: nodes, edges, pipeline, and MCP adapters.",
      {},
      async () => ({
        content: [{ type: "text", text: this.agent.describeGraph() }],
      }),
    );

    this.server.tool(
      "kontext_sync",
      "Trigger MCP synchronization to update meta index from connected data sources.",
      {
        connectorName: z.string().optional().describe("Optional: sync only this connector"),
      },
      async ({ connectorName }) => {
        await this.agent.syncMCP(connectorName);
        const text = connectorName
          ? `Synced connector: ${connectorName}`
          : "Synced all MCP connectors";
        return { content: [{ type: "text", text }] };
      },
    );

    this.server.tool(
      "kontext_auto_setup",
      "Auto-setup: collect documents from all connected MCP sources, build or expand the ontology via LLM classification, and index documents. Run once after connecting MCP sources.",
      {
        targetNodeCount: z
          .number()
          .optional()
          .describe("Target number of ontology nodes if built from scratch (default 10)"),
      },
      async ({ targetNodeCount }) => {
        const result = await this.agent.autoSetup(targetNodeCount ?? 10);
        const text = [
          "Auto-setup complete",
          `  Nodes created:    ${result.nodesCreated}`,
          `  Nodes reused:     ${result.nodesReused}`,
          `  Docs classified:  ${result.documentsClassified}`,
          `  Docs unmapped:    ${result.documentsUnmapped}`,
          "",
          "Generated ontology (save as kontext.yaml):",
          result.ontologyYaml,
        ].join("\n");
        return { content: [{ type: "text", text }] };
      },
    );
  }
}
