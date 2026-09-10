import {
  HttpMCPConnector,
  LocalMarkdownConnector,
  type MCPConnector,
  type MCPToolAccess,
  SseMCPConnector,
  StdioMCPConnector,
  ToolDrivenMCPConnector,
} from "@kontext-brain/mcp";
import { materializeGitSource } from "./git-source-checkout.js";
import type { MCPConfigDto } from "./kontext-config.js";
import { LocalCodeConnector, LocalRepositoryConnector } from "./local-code-connector.js";

/**
 * Builds an ontology source connector from its configuration entry. Shared so
 * the loader, the CLI and any host GUI resolve a source the same way; a second
 * copy of this mapping is how a source starts behaving differently depending on
 * which surface connected it.
 */
export function createSourceConnector(dto: MCPConfigDto): MCPConnector {
  const transport = dto.transport ?? (dto.path ? "local" : dto.command ? "stdio" : "sse");
  if (transport === "local") {
    if (!dto.path) throw new Error(`MCP '${dto.name}': local transport requires 'path'`);
    return repositoryConnector(dto, dto.path);
  }
  if (transport === "git") {
    if (!dto.url) throw new Error(`MCP '${dto.name}': git transport requires 'url'`);
    // A checkout is just a directory once it exists, so the local connectors read
    // it and nothing downstream learns a new source kind.
    return repositoryConnector(dto, materializeGitSource(dto.url, dto.ref ? { ref: dto.ref } : {}));
  }
  if (transport === "stdio") {
    if (!dto.command) throw new Error(`MCP '${dto.name}': stdio transport requires 'command'`);
    return withDocumentMapping(
      dto,
      new StdioMCPConnector(dto.name, dto.command, dto.args ?? [], dto.env),
    );
  }
  if (!dto.url) throw new Error(`MCP '${dto.name}': ${transport} transport requires 'url'`);
  if (transport === "http") {
    return withDocumentMapping(dto, new HttpMCPConnector(dto.name, dto.url, dto.headers));
  }
  return withDocumentMapping(dto, new SseMCPConnector(dto.name, dto.url, dto.headers));
}

/** A server that lists documents through tools is read through its declared mapping. */
function withDocumentMapping(
  dto: MCPConfigDto,
  connector: MCPConnector & MCPToolAccess,
): MCPConnector {
  return dto.documents ? new ToolDrivenMCPConnector(connector, dto.documents) : connector;
}

function repositoryConnector(dto: MCPConfigDto, directory: string): MCPConnector {
  const include = dto.include ? { include: dto.include } : {};
  const markdown = new LocalMarkdownConnector(dto.name, directory, include);
  if (!dto.code) return markdown;
  return new LocalRepositoryConnector(
    dto.name,
    markdown,
    new LocalCodeConnector(dto.name, directory, include),
  );
}

export function connectorsFromEntries(entries: readonly MCPConfigDto[]): readonly MCPConnector[] {
  return entries.map((entry) => createSourceConnector(entry));
}
