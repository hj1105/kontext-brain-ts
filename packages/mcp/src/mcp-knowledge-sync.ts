import { createHash } from "node:crypto";
import {
  type AccessControlList,
  type ChunkingStrategy,
  RecursiveChunkingStrategy,
  type ResourceSnapshot,
  type ResourceSnapshotEnricher,
  type ResourceSyncUseCase,
} from "@kontext-brain/core";
import type { MCPConnector, MCPData, MCPResource } from "./mcp-connector.js";

export interface MCPNormalizationInput {
  readonly organizationId: string;
  readonly connectorName: string;
  readonly resource: MCPResource;
  readonly data: MCPData;
  readonly ontologyNodeIds: readonly string[];
}

export interface MCPResourceSnapshotAdapter {
  readonly connectorName: string;
  readonly sourceType: string;
  normalize(input: MCPNormalizationInput): Promise<ResourceSnapshot>;
}

/** Fallback for sources that expose only a flat body through standard MCP. */
export class GenericMCPResourceSnapshotAdapter implements MCPResourceSnapshotAdapter {
  constructor(
    public readonly connectorName: string,
    public readonly sourceType: string,
    private readonly acl: AccessControlList,
    private readonly chunker: ChunkingStrategy = new RecursiveChunkingStrategy(),
  ) {}

  async normalize(input: MCPNormalizationInput): Promise<ResourceSnapshot> {
    const chunks = this.chunker.split(input.data.content);
    const contentHash = createHash("sha256")
      .update(
        JSON.stringify({
          title: input.resource.name,
          description: input.resource.description,
          body: input.data.content,
          metadata: input.data.metadata,
        }),
      )
      .digest("hex");
    return {
      organizationId: input.organizationId,
      source: {
        connectorId: input.connectorName,
        externalId: input.resource.id,
        type: this.sourceType,
      },
      title: input.resource.name,
      contentHash,
      body: input.data.content,
      acl: this.acl,
      ontologyNodeIds: input.ontologyNodeIds,
      chunks: chunks.map((chunk, index) => ({
        id: `chunk-${index}`,
        contentHash: createHash("sha256").update(chunk.fullText).digest("hex"),
        text: chunk.fullText,
        position: index,
        ontologyNodeIds: input.ontologyNodeIds,
      })),
    };
  }
}

export class MCPKnowledgeSynchronizer {
  private readonly adapters: ReadonlyMap<string, MCPResourceSnapshotAdapter>;

  constructor(
    private readonly resourceSync: ResourceSyncUseCase,
    adapters: readonly MCPResourceSnapshotAdapter[],
    private readonly snapshotEnricher?: ResourceSnapshotEnricher,
  ) {
    this.adapters = new Map(adapters.map((adapter) => [adapter.connectorName, adapter]));
  }

  async sync(
    organizationId: string,
    connector: MCPConnector,
    resource: MCPResource,
    ontologyNodeIds: readonly string[],
  ): Promise<void> {
    const adapter = this.adapters.get(connector.name);
    if (!adapter) throw new Error(`No MCP ResourceSnapshot adapter for "${connector.name}"`);
    const data = await connector.fetchResource(resource.id);
    const normalized = await adapter.normalize({
      organizationId,
      connectorName: connector.name,
      resource,
      data,
      ontologyNodeIds,
    });
    await this.resourceSync.execute(normalized, this.snapshotEnricher);
  }

  async remove(
    organizationId: string,
    connectorName: string,
    resourceId: string,
  ): Promise<boolean> {
    const adapter = this.adapters.get(connectorName);
    if (!adapter) throw new Error(`No MCP ResourceSnapshot adapter for "${connectorName}"`);
    return this.resourceSync.remove(organizationId, {
      connectorId: connectorName,
      externalId: resourceId,
      type: adapter.sourceType,
    });
  }
}
