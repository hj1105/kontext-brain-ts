import { readFileSync, statSync } from "node:fs";
import { extname, join, relative, sep } from "node:path";
import {
  type CodeLanguage,
  type CodeSymbolRecord,
  type LanguageCodeProvider,
  PythonCodeProvider,
  TypeScriptCodeProvider,
} from "@kontext-brain/code";
import {
  LOCAL_SOURCE_DEFAULT_EXCLUDE,
  type MCPConnector,
  type MCPData,
  type MCPResource,
  walkSourceFiles,
} from "@kontext-brain/mcp";

/**
 * Exposes a repository's source files as ontology documents. The classifier
 * reasons about a file from its path and the behaviour it exports, which is
 * what the code providers extract; the body is read only when a file is fetched.
 * Test files are skipped by default because they describe the behaviour a
 * neighbouring module already carries and would double every node's membership.
 */

export interface LocalCodeOptions {
  readonly include?: readonly string[];
  readonly exclude?: readonly string[];
  readonly maxFiles?: number;
  /** Larger files are generated or vendored far more often than written; skip them. */
  readonly maxFileBytes?: number;
  readonly includeTests?: boolean;
}

const LANGUAGE_BY_EXTENSION: ReadonlyMap<string, CodeLanguage> = new Map([
  [".ts", "typescript"],
  [".tsx", "typescript"],
  [".mts", "typescript"],
  [".cts", "typescript"],
  [".js", "javascript"],
  [".jsx", "javascript"],
  [".mjs", "javascript"],
  [".cjs", "javascript"],
  [".py", "python"],
]);
const CODE_EXTENSIONS = Array.from(LANGUAGE_BY_EXTENSION.keys());
const DEFAULT_EXCLUDE = [
  ...LOCAL_SOURCE_DEFAULT_EXCLUDE,
  "__pycache__",
  ".venv",
  "venv",
  "site-packages",
];
const DEFAULT_MAX_FILES = 1500;
const DEFAULT_MAX_FILE_BYTES = 256 * 1024;
const TEST_FILE =
  /(\.(test|spec)\.[cm]?[jt]sx?$)|(^|\/)(__tests__|tests?)\/|(^|\/)test_[^/]+\.py$|_test\.py$|(^|\/)conftest\.py$/;
const DECLARATION_FILE = /\.d\.[cm]?ts$/;
/** Enough names to say what a module does; a barrel that re-exports hundreds is described by its path. */
const MAX_SYMBOL_NAMES = 40;

export function codeLanguageForPath(filePath: string): CodeLanguage | undefined {
  return LANGUAGE_BY_EXTENSION.get(extname(filePath).toLowerCase());
}

export class LocalCodeConnector implements MCPConnector {
  private readonly providers: ReadonlyMap<CodeLanguage, LanguageCodeProvider>;

  constructor(
    public readonly name: string,
    private readonly root: string,
    private readonly options: LocalCodeOptions = {},
  ) {
    const typescript = new TypeScriptCodeProvider();
    this.providers = new Map<CodeLanguage, LanguageCodeProvider>([
      ["typescript", typescript],
      ["javascript", typescript],
      ["python", new PythonCodeProvider()],
    ]);
  }

  private files(): string[] {
    const limit = this.options.maxFiles ?? DEFAULT_MAX_FILES;
    const maxBytes = this.options.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES;
    const found = walkSourceFiles(this.root, {
      extensions: CODE_EXTENSIONS,
      exclude: this.options.exclude ?? DEFAULT_EXCLUDE,
      // Why: the walker's cap counts every match; tests and declarations are dropped after it.
      maxFiles: limit * 3,
      ...(this.options.include ? { include: this.options.include } : {}),
    });
    const kept: string[] = [];
    for (const path of found) {
      if (kept.length >= limit) break;
      const id = this.idOf(path);
      if (DECLARATION_FILE.test(id)) continue;
      if (!this.options.includeTests && TEST_FILE.test(id)) continue;
      try {
        if (statSync(path).size > maxBytes) continue;
      } catch {
        continue;
      }
      kept.push(path);
    }
    return kept;
  }

  private idOf(path: string): string {
    return relative(this.root, path).split(sep).join("/");
  }

  private describe(id: string, text: string): string {
    const language = codeLanguageForPath(id);
    const provider = language ? this.providers.get(language) : undefined;
    if (!language || !provider) return "source module";
    try {
      const analysis = provider.analyze({
        codebaseId: this.name,
        targetPath: id,
        files: [{ path: id, content: text }],
      });
      const names = exportedNames(analysis.symbols);
      return names.length > 0
        ? `${language} · exports ${names.join(", ")}`
        : `${language} module without exported symbols`;
    } catch {
      // Why: a file the provider cannot parse is still a document at a path; the
      // classifier can place it by that path alone.
      return `${language} module`;
    }
  }

  async listResources(): Promise<MCPResource[]> {
    const resources: MCPResource[] = [];
    for (const path of this.files()) {
      let text: string;
      try {
        text = readFileSync(path, "utf8");
      } catch {
        continue;
      }
      const id = this.idOf(path);
      resources.push({
        id,
        name: id,
        description: this.describe(id, text),
        mimeType: `text/x-${codeLanguageForPath(id) ?? "source"}`,
      });
    }
    return resources;
  }

  async fetchResource(resourceId: string): Promise<MCPData> {
    const path = join(this.root, resourceId);
    const inside = relative(this.root, path);
    if (inside.startsWith("..") || inside === "") {
      throw new Error(`local code '${this.name}': resource outside root: ${resourceId}`);
    }
    return {
      resourceId,
      content: readFileSync(path, "utf8"),
      metadata: {
        source: this.name,
        path: resourceId,
        language: codeLanguageForPath(resourceId) ?? "unknown",
      },
      fetchedAt: new Date(),
    };
  }

  async search(query: string): Promise<MCPData[]> {
    const needle = query.trim().toLowerCase();
    if (needle === "") return [];
    const hits: MCPData[] = [];
    for (const path of this.files()) {
      let text: string;
      try {
        text = readFileSync(path, "utf8");
      } catch {
        continue;
      }
      if (!text.toLowerCase().includes(needle)) continue;
      const id = this.idOf(path);
      hits.push({
        resourceId: id,
        content: text,
        metadata: { source: this.name, path: id, language: codeLanguageForPath(id) ?? "unknown" },
        fetchedAt: new Date(),
      });
    }
    return hits;
  }
}

function exportedNames(symbols: readonly CodeSymbolRecord[]): string[] {
  const names: string[] = [];
  for (const symbol of symbols) {
    if (symbol.identity.kind === "module" || !symbol.exported) continue;
    if (names.includes(symbol.identity.qualifiedName)) continue;
    names.push(symbol.identity.qualifiedName);
    if (names.length >= MAX_SYMBOL_NAMES) break;
  }
  return names;
}

/**
 * One source that reads a repository's documents and its code, routing each
 * resource to the connector that understands its file type. Both halves share
 * the source name, so provenance still says which repository a fact came from.
 */
export class LocalRepositoryConnector implements MCPConnector {
  readonly name: string;

  constructor(
    name: string,
    private readonly documents: MCPConnector,
    private readonly code: LocalCodeConnector,
  ) {
    this.name = name;
  }

  private owner(resourceId: string): MCPConnector {
    return codeLanguageForPath(resourceId) ? this.code : this.documents;
  }

  async listResources(): Promise<MCPResource[]> {
    const [documents, code] = await Promise.all([
      this.documents.listResources(),
      this.code.listResources(),
    ]);
    return [...documents, ...code];
  }

  fetchResource(resourceId: string): Promise<MCPData> {
    return this.owner(resourceId).fetchResource(resourceId);
  }

  async search(query: string): Promise<MCPData[]> {
    const [documents, code] = await Promise.all([
      this.documents.search(query),
      this.code.search(query),
    ]);
    return [...documents, ...code];
  }
}
