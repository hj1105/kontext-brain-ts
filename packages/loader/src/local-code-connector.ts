import { readFileSync, statSync } from "node:fs";
import { dirname, extname, join, relative, sep } from "node:path";
import {
  CodeKnowledgeSynchronizer,
  type CodeLanguage,
  CodeResourceSnapshotAdapter,
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
import type {
  CodeKnowledgeSource,
  CodeKnowledgeSyncInput,
  CodeKnowledgeSyncReport,
} from "./code-knowledge-source.js";

/**
 * Exposes a repository's code as ontology documents, one per directory: a
 * module is the unit an ontology node governs, and a directory's exported
 * symbols say what it does. Per-file documents would put an organization's
 * whole tree through the classifier — tens of thousands of prompts — for a
 * placement that files in one directory almost always share.
 * Test files are skipped by default because they describe the behaviour a
 * neighbouring module already carries.
 */

export interface LocalCodeOptions {
  readonly include?: readonly string[];
  readonly exclude?: readonly string[];
  /** Source files scanned per source. */
  readonly maxFiles?: number;
  /** Directory modules exposed per source; the ones exporting most come first. */
  readonly maxModules?: number;
  /** Larger files are generated or vendored far more often than written; skip them. */
  readonly maxFileBytes?: number;
  readonly includeTests?: boolean;
}

interface AnalyzedFile {
  readonly id: string;
  readonly path: string;
  readonly language: CodeLanguage;
  readonly exportedNames: readonly string[];
}

interface CodeModule {
  /** Directory relative to the root, with a trailing slash; the root is `./`. */
  readonly id: string;
  readonly directory: string;
  readonly files: readonly AnalyzedFile[];
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
// Why 80: an organization of 70 repositories then classifies in tens of batches, not hundreds.
const DEFAULT_MAX_MODULES = 80;
const DEFAULT_MAX_FILE_BYTES = 256 * 1024;
const TEST_FILE =
  /(\.(test|spec)\.[cm]?[jt]sx?$)|(^|\/)(__tests__|tests?)\/|(^|\/)test_[^/]+\.py$|_test\.py$|(^|\/)conftest\.py$/;
const DECLARATION_FILE = /\.d\.[cm]?ts$/;
/** Enough names to say what a module does; a barrel that re-exports hundreds is described by its path. */
const MAX_SYMBOL_NAMES = 40;
const ROOT_MODULE_ID = "./";

export function codeLanguageForPath(filePath: string): CodeLanguage | undefined {
  return LANGUAGE_BY_EXTENSION.get(extname(filePath).toLowerCase());
}

/** Module ids end with a slash; document ids are file paths. */
export function isCodeModuleId(resourceId: string): boolean {
  return resourceId.endsWith("/");
}

export class LocalCodeConnector implements MCPConnector, CodeKnowledgeSource {
  private readonly providers: ReadonlyMap<CodeLanguage, LanguageCodeProvider>;
  private modules: readonly CodeModule[] | undefined;

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

  private analyze(path: string): AnalyzedFile | undefined {
    const id = this.idOf(path);
    const language = codeLanguageForPath(id);
    const provider = language ? this.providers.get(language) : undefined;
    if (!language || !provider) return undefined;
    let text: string;
    try {
      text = readFileSync(path, "utf8");
    } catch {
      return undefined;
    }
    try {
      const analysis = provider.analyze({
        codebaseId: this.name,
        targetPath: id,
        files: [{ path: id, content: text }],
      });
      return { id, path, language, exportedNames: exportedNames(analysis.symbols) };
    } catch {
      // Why: a file the provider cannot parse is still part of its module; the
      // module is then described by its path and the files that do parse.
      return { id, path, language, exportedNames: [] };
    }
  }

  /** Directory modules, most-exporting first, capped; computed once per connector. */
  private collect(): readonly CodeModule[] {
    if (this.modules) return this.modules;
    const byDirectory = new Map<string, AnalyzedFile[]>();
    for (const path of this.files()) {
      const file = this.analyze(path);
      if (!file) continue;
      const directory = dirname(file.id);
      const key = directory === "." ? "" : directory;
      const files = byDirectory.get(key);
      if (files) files.push(file);
      else byDirectory.set(key, [file]);
    }
    const limit = this.options.maxModules ?? DEFAULT_MAX_MODULES;
    const modules = Array.from(byDirectory.entries()).map(([directory, files]) => ({
      id: directory === "" ? ROOT_MODULE_ID : `${directory}/`,
      directory,
      files: [...files].sort((left, right) => left.id.localeCompare(right.id)),
    }));
    modules.sort(
      (left, right) =>
        exportCount(right) - exportCount(left) ||
        right.files.length - left.files.length ||
        left.id.length - right.id.length ||
        left.id.localeCompare(right.id),
    );
    this.modules = modules.slice(0, limit);
    return this.modules;
  }

  private describe(module: CodeModule): string {
    const languages = Array.from(new Set(module.files.map((file) => file.language))).join("/");
    const names: string[] = [];
    for (const file of module.files) {
      for (const name of file.exportedNames) {
        if (names.length >= MAX_SYMBOL_NAMES) break;
        if (!names.includes(name)) names.push(name);
      }
    }
    const count = `${module.files.length} file${module.files.length === 1 ? "" : "s"}`;
    return names.length > 0
      ? `${languages} · ${count} · exports ${names.join(", ")}`
      : `${languages} · ${count} without exported symbols`;
  }

  /** The module read as one document: each file with what it exports. */
  private render(module: CodeModule): string {
    const lines = [`# ${module.directory === "" ? "." : module.directory}`, ""];
    for (const file of module.files) {
      lines.push(
        file.exportedNames.length > 0
          ? `- ${file.id}: exports ${file.exportedNames.join(", ")}`
          : `- ${file.id}`,
      );
    }
    return `${lines.join("\n")}\n`;
  }

  private moduleOf(resourceId: string): CodeModule {
    const module = this.collect().find((candidate) => candidate.id === resourceId);
    if (!module) throw new Error(`local code '${this.name}': unknown module: ${resourceId}`);
    return module;
  }

  async listResources(): Promise<MCPResource[]> {
    return this.collect().map((module) => ({
      id: module.id,
      name: module.directory === "" ? "." : module.directory,
      description: this.describe(module),
      mimeType: "text/x-code-module",
    }));
  }

  async fetchResource(resourceId: string): Promise<MCPData> {
    const module = this.moduleOf(resourceId);
    return {
      resourceId,
      content: this.render(module),
      metadata: {
        source: this.name,
        path: module.directory,
        files: String(module.files.length),
      },
      fetchedAt: new Date(),
    };
  }

  /**
   * Projects every module's files into the knowledge graph at symbol level. A
   * module's files are analyzed together so imports between them resolve; each
   * file becomes its own Resource carrying the module's ontology nodes.
   */
  async syncCodeKnowledge(input: CodeKnowledgeSyncInput): Promise<CodeKnowledgeSyncReport> {
    const adapter = new CodeResourceSnapshotAdapter();
    let filesSynced = 0;
    let filesFailed = 0;
    const modules = this.collect();
    const total = modules.reduce((sum, module) => sum + module.files.length, 0);
    input.onProgress?.(0, total);
    for (const module of modules) {
      const ontologyNodeIds = input.nodeIdsFor(module.id);
      const byLanguage = new Map<CodeLanguage, AnalyzedFile[]>();
      for (const file of module.files) {
        const group = byLanguage.get(file.language);
        if (group) group.push(file);
        else byLanguage.set(file.language, [file]);
      }
      for (const [language, files] of byLanguage) {
        const provider = this.providers.get(language);
        if (!provider) continue;
        const synchronizer = new CodeKnowledgeSynchronizer(input.resourceSync, provider, adapter);
        const project = files.flatMap((file) => {
          try {
            return [{ path: file.id, content: readFileSync(file.path, "utf8") }];
          } catch {
            return [];
          }
        });
        for (const file of project) {
          try {
            await synchronizer.sync({
              codebaseId: this.name,
              targetPath: file.path,
              files: project,
              organizationId: input.organizationId,
              acl: { organizationWide: true },
              ontologyNodeIds,
            });
            filesSynced += 1;
          } catch {
            // Why: one file the provider cannot analyze must not stop the rest of the
            // repository from becoming knowledge; the count says how many were left out.
            filesFailed += 1;
          }
          input.onProgress?.(filesSynced + filesFailed, total);
        }
      }
    }
    return { filesSynced, filesFailed };
  }

  async search(query: string): Promise<MCPData[]> {
    const needle = query.trim().toLowerCase();
    if (needle === "") return [];
    const hits: MCPData[] = [];
    for (const module of this.collect()) {
      const matches = module.files.some((file) => {
        if (file.exportedNames.some((name) => name.toLowerCase().includes(needle))) return true;
        try {
          return readFileSync(file.path, "utf8").toLowerCase().includes(needle);
        } catch {
          return false;
        }
      });
      if (!matches) continue;
      hits.push({
        resourceId: module.id,
        content: this.render(module),
        metadata: { source: this.name, path: module.directory, files: String(module.files.length) },
        fetchedAt: new Date(),
      });
    }
    return hits;
  }
}

function exportCount(module: CodeModule): number {
  return module.files.reduce((sum, file) => sum + file.exportedNames.length, 0);
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
 * resource to the connector that understands its id. Both halves share the
 * source name, so provenance still says which repository a fact came from.
 */
export class LocalRepositoryConnector implements MCPConnector, CodeKnowledgeSource {
  readonly name: string;

  constructor(
    name: string,
    private readonly documents: MCPConnector,
    private readonly code: LocalCodeConnector,
  ) {
    this.name = name;
  }

  private owner(resourceId: string): MCPConnector {
    return isCodeModuleId(resourceId) ? this.code : this.documents;
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

  syncCodeKnowledge(input: CodeKnowledgeSyncInput): Promise<CodeKnowledgeSyncReport> {
    return this.code.syncCodeKnowledge(input);
  }

  async search(query: string): Promise<MCPData[]> {
    const [documents, code] = await Promise.all([
      this.documents.search(query),
      this.code.search(query),
    ]);
    return [...documents, ...code];
  }
}
