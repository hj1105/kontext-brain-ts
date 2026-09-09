import {
  LocalMarkdownConnector,
  type MCPConnector,
  SseMCPConnector,
  StdioMCPConnector,
} from "@kontext-brain/mcp";
import { materializeGitSource } from "./git-source-checkout.js";
import type { MCPConfigDto } from "./kontext-config.js";

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
    return new LocalMarkdownConnector(dto.name, dto.path, {
      ...(dto.include ? { include: dto.include } : {}),
    });
  }
  if (transport === "git") {
    if (!dto.url) throw new Error(`MCP '${dto.name}': git transport requires 'url'`);
    // A checkout is just a directory of Markdown once it exists, so the local
    // connector reads it and nothing downstream learns a new source kind.
    const directory = materializeGitSource(dto.url, dto.ref ? { ref: dto.ref } : {});
    return new LocalMarkdownConnector(dto.name, directory, {
      ...(dto.include ? { include: dto.include } : {}),
    });
  }
  if (transport === "stdio") {
    if (!dto.command) throw new Error(`MCP '${dto.name}': stdio transport requires 'command'`);
    return new StdioMCPConnector(dto.name, dto.command, dto.args ?? [], dto.env);
  }
  if (!dto.url) throw new Error(`MCP '${dto.name}': sse transport requires 'url'`);
  return new SseMCPConnector(dto.name, dto.url);
}

export function connectorsFromEntries(entries: readonly MCPConfigDto[]): readonly MCPConnector[] {
  return entries.map((entry) => createSourceConnector(entry));
}
