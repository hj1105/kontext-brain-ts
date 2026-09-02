import { ABSOLUTE_MAX_AUTO_NODE_COUNT, MIN_AUTO_NODE_COUNT } from "@kontext-brain/core";
import {
  GitHubCanonicalOntologySource,
  GitHubOntologyProposalPublisher,
} from "@kontext-brain/github";
import type { KontextAgent } from "@kontext-brain/loader";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { OntologyMaintenanceService } from "./ontology-maintenance.js";
import { PeriodicMCPSyncService } from "./periodic-mcp-sync.js";
import { registerTaskWorkflowTools } from "./task-tool-server.js";
import { type KontextTaskWorkflowOperations } from "./task-workflow-tools.js";

/**
 * Exposes a KontextAgent as an MCP server over stdio.
 * Compatible with Claude Desktop, Claude Code, and other MCP clients.
 */
export class KontextToolServer {
  private readonly server: McpServer;
  private readonly periodicSync: PeriodicMCPSyncService;
  private readonly ontologyMaintenance: OntologyMaintenanceService | null;
  private stopPeriodicSync: (() => void) | null = null;

  constructor(
    private readonly agent: KontextAgent,
    workflow?: KontextTaskWorkflowOperations,
  ) {
    this.server = new McpServer({
      name: "kontext-brain",
      version: "0.1.0",
    });
    this.ontologyMaintenance = createOntologyMaintenance(agent);
    this.periodicSync = new PeriodicMCPSyncService({
      syncMCP: (connectorName) => this.syncWithOntologyMaintenance(connectorName),
    });
    this.registerTools();
    if (workflow) registerTaskWorkflowTools(this.server, workflow);
  }

  async start(): Promise<void> {
    process.stderr.write("kontext-brain MCP server starting (stdio mode)\n");
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    const refresh = this.agent.mcpRefreshConfiguration;
    if (refresh.enabled) {
      this.stopPeriodicSync = this.periodicSync.start(
        {
          intervalMilliseconds: refresh.intervalSeconds * 1_000,
          runOnStart: refresh.runOnStart,
        },
        (result) => {
          process.stderr.write(
            `kontext-brain sync complete: connectors=${result.connectorsSynced} added=${result.resourcesAdded} updated=${result.resourcesUpdated} removed=${result.resourcesRemoved}\n`,
          );
        },
        (error) => {
          process.stderr.write(
            `kontext-brain sync failed: ${error instanceof Error ? error.message : String(error)}\n`,
          );
        },
      );
    }
  }

  async close(): Promise<void> {
    this.stopPeriodicSync?.();
    this.stopPeriodicSync = null;
    await this.server.close();
  }

  private async syncWithOntologyMaintenance(connectorName?: string) {
    await this.refreshCanonicalOntology();
    const result = await this.agent.syncMCP(connectorName);
    await this.publishOntologyProposals();
    return result;
  }

  private async refreshCanonicalOntology(): Promise<void> {
    if (!this.ontologyMaintenance) return;
    try {
      const activation = await this.ontologyMaintenance.refreshCanonical();
      if (activation.changed) {
        process.stderr.write(
          `kontext-brain ontology activated: nodes=${activation.nodeCount} hash=${activation.contentHash}\n`,
        );
      }
    } catch (error) {
      process.stderr.write(
        `kontext-brain ontology refresh failed: ${error instanceof Error ? error.message : String(error)}\n`,
      );
    }
  }

  private async publishOntologyProposals(): Promise<void> {
    if (!this.ontologyMaintenance) return;
    try {
      const publication = await this.ontologyMaintenance.publishProposals();
      if (publication.changed) {
        process.stderr.write(
          `kontext-brain ontology proposal published: ${publication.url ?? "URL unavailable"}\n`,
        );
      }
    } catch (error) {
      process.stderr.write(
        `kontext-brain ontology proposal publish failed: ${error instanceof Error ? error.message : String(error)}\n`,
      );
    }
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
        const result = await this.agent.retrieve(question);
        const nodes = result.usedOntologyNodes
          .map((n) => `## ${n.id}\n${n.description}`)
          .join("\n\n");
        const docs = result.selectedMetaDocs.map((d) => `- [${d.source}] ${d.title}`).join("\n");
        const text = [
          "=== Retrieved Context ===",
          nodes,
          "=== Documents ===",
          docs,
          "=== Evidence ===",
          result.context,
          `Tokens used: ${result.contextTokensUsed}`,
        ].join("\n\n");
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
      "Trigger MCP synchronization, refresh a merged canonical ontology, and publish new ontology proposals when configured.",
      {
        connectorName: z.string().optional().describe("Optional: sync only this connector"),
      },
      async ({ connectorName }) => {
        const result = await this.syncWithOntologyMaintenance(connectorName);
        const text = [
          connectorName ? `Synced connector: ${connectorName}` : "Synced all MCP connectors",
          `Connectors: ${result.connectorsSynced}`,
          `Added: ${result.resourcesAdded}`,
          `Updated: ${result.resourcesUpdated}`,
          `Removed: ${result.resourcesRemoved}`,
          `Classified: ${result.resourcesClassified}`,
          `Unmapped: ${result.resourcesUnmapped}`,
        ].join("\n");
        return { content: [{ type: "text", text }] };
      },
    );

    this.server.tool(
      "kontext_ontology_status",
      "Show active ontology state, queued proposals, and canonical GitHub synchronization status.",
      {},
      async () => {
        const proposals = await this.agent.listPendingOntologyProposals();
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  enabled: this.ontologyMaintenance !== null,
                  activeContentHash: this.agent.activeOntologyContentHash,
                  activeNodeCount: this.agent.ontologyGraph.nodes.size,
                  proposals,
                  maintenance: this.ontologyMaintenance?.getStatus(),
                },
                null,
                2,
              ),
            },
          ],
        };
      },
    );

    this.server.tool(
      "kontext_publish_ontology_proposals",
      "Publish queued ontology proposals to the configured GitHub draft pull request.",
      {},
      async () => {
        if (!this.ontologyMaintenance) {
          return {
            content: [{ type: "text", text: "Ontology updates are disabled." }],
          };
        }
        await this.refreshCanonicalOntology();
        const result = await this.ontologyMaintenance.publishProposals();
        return {
          content: [
            {
              type: "text",
              text: result.changed
                ? `Ontology proposal published: ${result.url ?? "URL unavailable"}`
                : "No open ontology proposals.",
            },
          ],
        };
      },
    );

    this.server.tool(
      "kontext_auto_setup",
      "Auto-setup: collect documents from all connected MCP sources, build or expand the ontology via LLM classification, and index documents. Run once after connecting MCP sources.",
      {
        targetNodeCount: z
          .number()
          .int()
          .min(MIN_AUTO_NODE_COUNT)
          .max(ABSOLUTE_MAX_AUTO_NODE_COUNT)
          .optional()
          .describe(
            "Optional target node-count override; omit to infer it from corpus size and topic diversity",
          ),
      },
      async ({ targetNodeCount }) => {
        await this.refreshCanonicalOntology();
        const result = await this.agent.autoSetup(targetNodeCount);
        await this.publishOntologyProposals();
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

function createOntologyMaintenance(agent: KontextAgent): OntologyMaintenanceService | null {
  const configuration = agent.ontologyUpdateConfiguration;
  if (!configuration.enabled) return null;
  const github = configuration.github;
  if (!github) throw new Error("Ontology updates require GitHub configuration");
  const token = process.env[github.tokenEnv];
  if (!token) {
    throw new Error(`Ontology updates require the ${github.tokenEnv} environment variable`);
  }
  const common = {
    owner: github.owner,
    repository: github.repository,
    token,
    ontologyPath: github.ontologyPath,
    baseBranch: github.baseBranch,
    apiUrl: github.apiUrl,
  };
  return new OntologyMaintenanceService(
    agent,
    new GitHubCanonicalOntologySource(common),
    new GitHubOntologyProposalPublisher({
      ...common,
      proposalBranch: github.proposalBranch,
    }),
  );
}
