import type { OntologyCliOptions, OntologyCliResult } from "./ontology-cli.js";
import type { EmbeddingSettings } from "./text-embedder-factory.js";

/** Text rendering for the ontology commands; `--json` callers get the result verbatim. */

function pad(value: string, width: number): string {
  return value.length >= width ? value : value + " ".repeat(width - value.length);
}

function renderList(result: Extract<OntologyCliResult, { command: "list" }>): string {
  if (result.sources.length === 0) {
    return "No sources configured. Run import-mcp or add.\n";
  }
  const lines = [`${result.sources.length} source(s):`];
  const width = Math.max(...result.sources.map((source) => source.name.length));
  for (const source of result.sources) {
    const type = source.type ? ` type=${source.type}` : "";
    lines.push(`  ${pad(source.name, width)}  ${source.transport}${type}  ${source.target}`);
  }
  lines.push(`Search embedding: ${describeEmbedding(result.embedding)}`);
  return `${lines.join("\n")}\n`;
}

function renderImport(result: Extract<OntologyCliResult, { command: "import-mcp" }>): string {
  if (result.discovered.length === 0) {
    return "No MCP servers found. Checked Claude Code and Codex; pass --project for project-scoped ones.\n";
  }
  const lines: string[] = [];
  for (const item of result.discovered) {
    const mark = item.alreadyPresent ? " " : "+";
    const scope = item.scope ? ` (${item.scope})` : "";
    const type = item.type ? ` type=${item.type}` : "";
    lines.push(`${mark} ${item.name}  from ${item.origin}${scope}${type}`);
  }
  lines.push("");
  lines.push(
    `Added ${result.added.length}, already present ${result.discovered.length - result.added.length}.`,
  );
  lines.push(result.written ? "Saved." : "Nothing written. Re-run with --write to save.");
  return `${lines.join("\n")}\n`;
}

function renderCheck(result: Extract<OntologyCliResult, { command: "check" }>): string {
  const lines = [`Checking ${result.sources.length} source(s):`];
  for (const source of result.sources) {
    lines.push(
      source.ok
        ? `  ok      ${source.name}  ${source.resourceCount} resources`
        : `  FAILED  ${source.name}  ${source.error ?? "unknown error"}`,
    );
  }
  lines.push("");
  const failures = result.sources.filter((source) => !source.ok).length;
  lines.push(
    failures === 0
      ? "All sources reachable."
      : `${failures} source(s) unreachable. Fix them before setup.`,
  );
  return `${lines.join("\n")}\n`;
}

function describeEmbedding(settings: EmbeddingSettings): string {
  if (settings.provider === "none") return "none (lexical search only)";
  const where = settings.baseUrl ? ` at ${settings.baseUrl}` : "";
  return `${settings.provider} ${settings.model}${where}`;
}

function renderSetup(result: Extract<OntologyCliResult, { command: "setup" }>): string {
  const lines = [
    `Nodes created ${result.nodesCreated}, reused ${result.nodesReused}`,
    `Documents classified ${result.documentsClassified}, unmapped ${result.documentsUnmapped}`,
    `Nodes: ${result.nodeIds.join(", ")}`,
    result.embeddingError
      ? `Embedding failed (${describeEmbedding(result.embedding)}): ${result.embeddingError}; search stays lexical.`
      : `Chunks embedded ${result.chunksEmbedded} (${describeEmbedding(result.embedding)})`,
    "",
    result.written
      ? `Saved ${result.nodeIds.length} ontology node(s).`
      : `${result.nodeIds.length} node(s) ready. Re-run with --write to save them.`,
  ];
  return `${lines.join("\n")}\n`;
}

function renderRepositories(
  result: Extract<OntologyCliResult, { command: "github-repos" }>,
): string {
  const lines = [
    `${result.owner} (${result.kind}) has ${result.repositories.length} repositor${result.repositories.length === 1 ? "y" : "ies"}:`,
  ];
  for (const repository of result.repositories) {
    const flags = [
      repository.private ? "private" : "public",
      ...(repository.archived ? ["archived"] : []),
      ...(repository.fork ? ["fork"] : []),
    ].join(", ");
    lines.push(
      `  ${repository.fullName}  ${repository.language ?? "-"}  ${flags}  ${repository.cloneUrl}`,
    );
  }
  lines.push(
    "",
    "Add one with: kontext-ontology add --name <name> --transport git --url <clone url> [--code] --write",
    "",
  );
  return lines.join("\n");
}

export function renderResult(result: OntologyCliResult, options: OntologyCliOptions): string {
  if (options.json) {
    return `${JSON.stringify(result, null, 2)}\n`;
  }
  if (!result.ok && "error" in result) {
    return `${result.error}\n`;
  }
  switch (result.command) {
    case "list":
      return renderList(result as Extract<OntologyCliResult, { command: "list" }>);
    case "import-mcp":
      return renderImport(result as Extract<OntologyCliResult, { command: "import-mcp" }>);
    case "add": {
      const added = result as Extract<OntologyCliResult, { command: "add" }>;
      return added.written
        ? `Added source '${added.name}'.\n`
        : `Source '${added.name}' is valid. Re-run with --write to save it.\n`;
    }
    case "embedding": {
      const chosen = result as Extract<OntologyCliResult, { command: "embedding" }>;
      return chosen.written
        ? `Search embedding set to ${describeEmbedding(chosen.embedding)}.\n`
        : `Search embedding would be ${describeEmbedding(chosen.embedding)}. Re-run with --write to save it.\n`;
    }
    case "embed": {
      const done = result as Extract<OntologyCliResult, { command: "embed" }>;
      return `Embedded ${done.chunksEmbedded} of ${done.chunksTotal} chunk(s) with ${describeEmbedding(done.embedding)}.\n`;
    }
    case "map": {
      const mapped = result as Extract<OntologyCliResult, { command: "map" }>;
      return mapped.written
        ? `Saved the document mapping for '${mapped.name}'.\n`
        : `Mapping for '${mapped.name}' is valid. Re-run with --write to save it.\n`;
    }
    case "inspect": {
      const inspected = result as Extract<OntologyCliResult, { command: "inspect" }>;
      const lines = [
        `${inspected.name} (${inspected.transport}): ${inspected.resourceCount} resource(s), ${inspected.tools.length} tool(s)`,
      ];
      for (const tool of inspected.tools) {
        lines.push(`  ${tool.name}  ${tool.description.replace(/\s+/g, " ").slice(0, 120)}`);
      }
      if (inspected.resourceCount === 0 && inspected.tools.length > 0) {
        lines.push(
          "",
          "This server exposes tools, not resources. Map documents with:",
          "  kontext-ontology add … --list-tool <tool> --items <path> --id <path> --title <path> --read-tool <tool> --read-arg <name> --content <path>",
        );
      }
      lines.push("");
      return lines.join("\n");
    }
    case "nodes": {
      const listing = result as Extract<OntologyCliResult, { command: "nodes" }>;
      const lines = [`${listing.nodes.length} ontology node(s):`];
      for (const node of listing.nodes) {
        const count = node.resourceCount === null ? "" : `  ${node.resourceCount} document(s)`;
        lines.push(`  ${node.id}${node.parentId ? ` (under ${node.parentId})` : ""}${count}`);
        for (const sample of node.samples.slice(0, 3)) {
          lines.push(`      - ${sample.connectorId}:${sample.externalId}`);
        }
      }
      lines.push("");
      return lines.join("\n");
    }
    case "query": {
      const query = result as Extract<OntologyCliResult, { command: "query" }>;
      const lines = [
        `${query.hits.length} hit(s) over ${query.chunksScanned} chunks in ${query.resourcesScanned} resources (${query.mode}${query.embeddingError ? `; ${query.embeddingError}` : ""}):`,
      ];
      for (const hit of query.hits) {
        lines.push(
          `  ${hit.score.toFixed(2)}  ${hit.source.connectorId}:${hit.source.externalId}  [${hit.ontologyNodeIds.join(", ")}]`,
          `      ${hit.text.replace(/\s+/g, " ").slice(0, 160)}`,
          `      evidence ${hit.evidenceId}`,
        );
      }
      lines.push("");
      return lines.join("\n");
    }
    case "github-repos":
      return renderRepositories(result as Extract<OntologyCliResult, { command: "github-repos" }>);
    case "check":
      return renderCheck(result as Extract<OntologyCliResult, { command: "check" }>);
    case "setup":
      return renderSetup(result as Extract<OntologyCliResult, { command: "setup" }>);
    default:
      return `${JSON.stringify(result)}\n`;
  }
}
