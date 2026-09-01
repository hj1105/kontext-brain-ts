import { createHash } from "node:crypto";
import path from "node:path";
import ts from "typescript";
import type {
  CodeAnalysisInput,
  CodeFileAnalysis,
  CodeLanguage,
  CodeRelationship,
  CodeRelationshipObject,
  CodeRelationshipPredicate,
  CodeSymbolIdentity,
  CodeSymbolKind,
  CodeSymbolRecord,
  LanguageCodeProvider,
  UnresolvedCodeRelationship,
} from "./domain.js";

const VIRTUAL_ROOT = "/__kontext_codebase__";
const printer = ts.createPrinter({ removeComments: true, newLine: ts.NewLineKind.LineFeed });

export class TypeScriptCodeProvider implements LanguageCodeProvider {
  readonly language = "typescript" as const;
  readonly semanticSupport = "certified" as const;

  analyze(input: CodeAnalysisInput): CodeFileAnalysis {
    const targetPath = normalizeRelativePath(input.targetPath);
    const files = new Map(
      input.files.map((file) => [virtualPath(file.path), file.content] as const),
    );
    const targetVirtualPath = virtualPath(targetPath);
    const targetText = files.get(targetVirtualPath);
    if (targetText === undefined) {
      throw new Error(`Target file "${targetPath}" is missing from the Codebase snapshot`);
    }

    const options: ts.CompilerOptions = {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.NodeNext,
      moduleResolution: ts.ModuleResolutionKind.NodeNext,
      allowJs: true,
      checkJs: false,
      jsx: ts.JsxEmit.ReactJSX,
      skipLibCheck: true,
      strict: true,
    };
    const host = createInMemoryCompilerHost(files, options);
    const program = ts.createProgram({ rootNames: Array.from(files.keys()), options, host });
    const sourceFile = program.getSourceFile(targetVirtualPath);
    if (!sourceFile) throw new Error(`TypeScript did not load target file "${targetPath}"`);
    const checker = program.getTypeChecker();
    const language = languageForPath(targetPath);

    const symbols: CodeSymbolRecord[] = [];
    const declarationSymbols = new Map<ts.Node, CodeSymbolRecord>();
    const moduleSymbol = createModuleSymbol(input.codebaseId, targetPath, language, sourceFile);
    symbols.push(moduleSymbol);
    declarationSymbols.set(sourceFile, moduleSymbol);

    let position = 1;
    const add = (description: DeclarationDescription): CodeSymbolRecord => {
      const record = createSymbolRecord(
        input.codebaseId,
        targetPath,
        language,
        sourceFile,
        checker,
        description,
        position++,
      );
      symbols.push(record);
      declarationSymbols.set(description.node, record);
      return record;
    };

    for (const statement of sourceFile.statements) {
      if (ts.isFunctionDeclaration(statement) && statement.name) {
        add({
          node: statement,
          kind: "function",
          qualifiedName: statement.name.text,
          behaviorBearing: Boolean(statement.body),
          exported: isExported(statement),
        });
      } else if (ts.isClassDeclaration(statement) && statement.name) {
        const classExported = isExported(statement);
        add({
          node: statement,
          kind: "class",
          qualifiedName: statement.name.text,
          behaviorBearing: false,
          exported: classExported,
        });
        for (const member of statement.members) {
          const memberDescription = describeClassMember(member, statement.name.text, classExported);
          if (memberDescription) add(memberDescription);
        }
      } else if (ts.isInterfaceDeclaration(statement)) {
        add({
          node: statement,
          kind: "interface",
          qualifiedName: statement.name.text,
          behaviorBearing: false,
          exported: isExported(statement),
        });
      } else if (ts.isTypeAliasDeclaration(statement)) {
        add({
          node: statement,
          kind: "type",
          qualifiedName: statement.name.text,
          behaviorBearing: false,
          exported: isExported(statement),
        });
      } else if (ts.isEnumDeclaration(statement)) {
        add({
          node: statement,
          kind: "enum",
          qualifiedName: statement.name.text,
          behaviorBearing: false,
          exported: isExported(statement),
        });
      } else if (ts.isVariableStatement(statement)) {
        for (const declaration of statement.declarationList.declarations) {
          if (!ts.isIdentifier(declaration.name)) continue;
          const behaviorBearing = Boolean(
            declaration.initializer &&
              (ts.isArrowFunction(declaration.initializer) ||
                ts.isFunctionExpression(declaration.initializer)),
          );
          add({
            node: declaration,
            signatureNode:
              declaration.initializer &&
              (ts.isArrowFunction(declaration.initializer) ||
                ts.isFunctionExpression(declaration.initializer))
                ? declaration.initializer
                : undefined,
            kind: behaviorBearing ? "named_arrow" : "constant",
            qualifiedName: declaration.name.text,
            behaviorBearing,
            exported: isExported(statement),
          });
        }
      }
    }

    const relationships: CodeRelationship[] = [];
    const unresolvedRelationships: UnresolvedCodeRelationship[] = [];
    const relationshipKeys = new Set<string>();
    const addRelationship = (
      subject: CodeSymbolRecord,
      predicate: CodeRelationshipPredicate,
      object: CodeRelationshipObject,
      evidence: CodeSymbolRecord = subject,
    ): void => {
      const key = JSON.stringify([subject.symbolId, predicate, object, evidence.symbolId]);
      if (relationshipKeys.has(key)) return;
      relationshipKeys.add(key);
      relationships.push({
        relationshipId: `code-fact:${sha256(key)}`,
        subjectSymbolId: subject.symbolId,
        predicate,
        object,
        evidenceSymbolId: evidence.symbolId,
      });
    };

    for (const statement of sourceFile.statements) {
      if (ts.isImportDeclaration(statement) && ts.isStringLiteral(statement.moduleSpecifier)) {
        addRelationship(
          moduleSymbol,
          "imports",
          resolveModuleObject(checker, statement.moduleSpecifier, input.codebaseId) ?? {
            kind: "literal",
            value: statement.moduleSpecifier.text,
          },
        );
      }
      if (ts.isClassDeclaration(statement) || ts.isInterfaceDeclaration(statement)) {
        const subject = declarationSymbols.get(statement);
        if (!subject) continue;
        for (const clause of statement.heritageClauses ?? []) {
          const predicate =
            clause.token === ts.SyntaxKind.ImplementsKeyword ? "implements" : "extends";
          for (const typeNode of clause.types) {
            const object = resolveRelationshipObject(
              checker,
              typeNode.expression,
              input.codebaseId,
              language,
            );
            if (object) addRelationship(subject, predicate, object);
          }
        }
      }
    }

    for (const [node, symbol] of declarationSymbols) {
      if (!symbol.behaviorBearing) continue;
      const signatureNode = signatureDeclarationFor(node);
      const signature = signatureNode
        ? checker.getSignatureFromDeclaration(signatureNode)
        : undefined;
      if (signature) {
        addRelationship(symbol, "returns", {
          kind: "literal",
          value: checker.typeToString(checker.getReturnTypeOfSignature(signature)),
        });
      }
    }

    const visit = (node: ts.Node): void => {
      const owner = findOwningSymbol(node, declarationSymbols) ?? moduleSymbol;
      if (ts.isCallExpression(node)) {
        const resolved = resolveCallObject(checker, node.expression, input.codebaseId, language);
        if (resolved.object) {
          addRelationship(owner, "calls", resolved.object);
        } else {
          unresolvedRelationships.push({
            subjectSymbolId: owner.symbolId,
            predicate: "calls",
            expression: node.expression.getText(sourceFile),
            reason: resolved.reason,
          });
        }
      }
      const environmentKey = readEnvironmentKey(node);
      if (environmentKey) {
        addRelationship(owner, "reads_env", { kind: "literal", value: environmentKey });
      }
      if (ts.isThrowStatement(node)) {
        addRelationship(owner, "throws", {
          kind: "literal",
          value: node.expression
            ? checker.typeToString(checker.getTypeAtLocation(node.expression))
            : "unknown",
        });
      }
      ts.forEachChild(node, visit);
    };
    visit(sourceFile);

    const diagnostics = [
      ...program.getSyntacticDiagnostics(sourceFile),
      ...program.getSemanticDiagnostics(sourceFile),
    ].map(formatDiagnostic);

    return {
      codebaseId: input.codebaseId,
      relativePath: targetPath,
      language,
      sourceText: targetText,
      contentHash: sha256(canonicalSyntax(sourceFile, sourceFile)),
      symbols,
      relationships,
      unresolvedRelationships: deduplicateUnresolved(unresolvedRelationships),
      diagnostics,
    };
  }
}

interface DeclarationDescription {
  readonly node: ts.Node;
  readonly signatureNode?: ts.SignatureDeclaration;
  readonly kind: CodeSymbolKind;
  readonly qualifiedName: string;
  readonly behaviorBearing: boolean;
  readonly exported: boolean;
}

function describeClassMember(
  member: ts.ClassElement,
  className: string,
  classExported: boolean,
): DeclarationDescription | undefined {
  const memberExported = classExported && !hasModifier(member, ts.SyntaxKind.PrivateKeyword);
  if (ts.isConstructorDeclaration(member)) {
    return {
      node: member,
      signatureNode: member,
      kind: "constructor",
      qualifiedName: `${className}.constructor`,
      behaviorBearing: Boolean(member.body),
      exported: memberExported,
    };
  }
  const name = propertyName(member.name);
  if (!name) return undefined;
  if (ts.isMethodDeclaration(member)) {
    return {
      node: member,
      signatureNode: member,
      kind: "method",
      qualifiedName: `${className}.${name}`,
      behaviorBearing: Boolean(member.body),
      exported: memberExported,
    };
  }
  if (ts.isGetAccessorDeclaration(member) || ts.isSetAccessorDeclaration(member)) {
    return {
      node: member,
      signatureNode: member,
      kind: ts.isGetAccessorDeclaration(member) ? "getter" : "setter",
      qualifiedName: `${className}.${name}`,
      behaviorBearing: Boolean(member.body),
      exported: memberExported,
    };
  }
  if (ts.isPropertyDeclaration(member)) {
    return {
      node: member,
      kind: "field",
      qualifiedName: `${className}.${name}`,
      behaviorBearing: false,
      exported: memberExported,
    };
  }
  return undefined;
}

function createSymbolRecord(
  codebaseId: string,
  relativePath: string,
  language: CodeLanguage,
  sourceFile: ts.SourceFile,
  checker: ts.TypeChecker,
  description: DeclarationDescription,
  position: number,
): CodeSymbolRecord {
  const signatureNode = description.signatureNode ?? signatureDeclarationFor(description.node);
  const signature = signatureNode
    ? signatureText(checker, signatureNode)
    : declarationSignature(description.node, sourceFile);
  const identity: CodeSymbolIdentity = {
    codebaseId,
    relativePath,
    language,
    kind: description.kind,
    qualifiedName: description.qualifiedName,
    signatureDiscriminator: signature,
  };
  return {
    symbolId: symbolId(identity),
    sourceChunkId: sourceChunkId(identity),
    identity,
    behaviorBearing: description.behaviorBearing,
    exported: description.exported,
    signature,
    contentHash: sha256(canonicalSyntax(description.node, sourceFile)),
    text: description.node.getText(sourceFile),
    position,
    semanticSupport: "certified",
  };
}

function createModuleSymbol(
  codebaseId: string,
  relativePath: string,
  language: CodeLanguage,
  sourceFile: ts.SourceFile,
): CodeSymbolRecord {
  const identity: CodeSymbolIdentity = {
    codebaseId,
    relativePath,
    language,
    kind: "module",
    qualifiedName: "<module>",
    signatureDiscriminator: relativePath,
  };
  const moduleStatements = sourceFile.statements.filter(
    (statement) => ts.isImportDeclaration(statement) || ts.isExportDeclaration(statement),
  );
  const normalized = moduleStatements
    .map((statement) => canonicalSyntax(statement, sourceFile))
    .join("\n");
  return {
    symbolId: symbolId(identity),
    sourceChunkId: sourceChunkId(identity),
    identity,
    behaviorBearing: false,
    exported: true,
    signature: relativePath,
    contentHash: sha256(normalized),
    text: moduleStatements.map((statement) => statement.getText(sourceFile)).join("\n"),
    position: 0,
    semanticSupport: "certified",
  };
}

function resolveModuleObject(
  checker: ts.TypeChecker,
  moduleSpecifier: ts.StringLiteral,
  codebaseId: string,
): CodeRelationshipObject | undefined {
  const target = checker.getSymbolAtLocation(moduleSpecifier);
  const declaration = target?.valueDeclaration ?? target?.declarations?.[0];
  if (!declaration || !ts.isSourceFile(declaration)) return undefined;
  const relativePath = relativeProjectPath(declaration.fileName);
  if (!relativePath) return undefined;
  const identity: CodeSymbolIdentity = {
    codebaseId,
    relativePath,
    language: languageForPath(relativePath),
    kind: "module",
    qualifiedName: "<module>",
    signatureDiscriminator: relativePath,
  };
  return {
    kind: "symbol",
    symbolId: symbolId(identity),
    qualifiedName: identity.qualifiedName,
    entityScope: "global",
  };
}

function resolveCallObject(
  checker: ts.TypeChecker,
  expression: ts.Expression,
  codebaseId: string,
  language: CodeLanguage,
): { object?: CodeRelationshipObject; reason: UnresolvedCodeRelationship["reason"] } {
  const lookup = ts.isPropertyAccessExpression(expression) ? expression.name : expression;
  let target = checker.getSymbolAtLocation(lookup);
  if (!target) return { reason: "no_symbol" };
  if (target.flags & ts.SymbolFlags.Alias) target = checker.getAliasedSymbol(target);
  const declaration = target.valueDeclaration ?? target.declarations?.[0];
  if (!declaration || !hasConcreteImplementation(declaration)) {
    return { reason: "no_concrete_declaration" };
  }
  const object = relationshipObjectForDeclaration(
    checker,
    declaration,
    codebaseId,
    language,
    target,
  );
  return object ? { object, reason: "no_symbol" } : { reason: "outside_project" };
}

function resolveRelationshipObject(
  checker: ts.TypeChecker,
  expression: ts.Expression,
  codebaseId: string,
  language: CodeLanguage,
): CodeRelationshipObject | undefined {
  let target = checker.getSymbolAtLocation(expression);
  if (!target) return undefined;
  if (target.flags & ts.SymbolFlags.Alias) target = checker.getAliasedSymbol(target);
  const declaration = target.valueDeclaration ?? target.declarations?.[0];
  return declaration
    ? relationshipObjectForDeclaration(checker, declaration, codebaseId, language, target)
    : undefined;
}

function relationshipObjectForDeclaration(
  checker: ts.TypeChecker,
  declaration: ts.Declaration,
  codebaseId: string,
  language: CodeLanguage,
  resolvedSymbol?: ts.Symbol,
): CodeRelationshipObject | undefined {
  const sourceFile = declaration.getSourceFile();
  const relativePath = relativeProjectPath(sourceFile.fileName);
  if (!relativePath) {
    return {
      kind: "literal",
      value: resolvedSymbol
        ? checker.symbolToString(resolvedSymbol)
        : declaration.getText(sourceFile),
    };
  }
  const description = describeAnyDeclaration(declaration);
  if (!description) return undefined;
  const signatureNode = description.signatureNode ?? signatureDeclarationFor(declaration);
  const signature = signatureNode
    ? signatureText(checker, signatureNode)
    : declarationSignature(declaration, sourceFile);
  const identity: CodeSymbolIdentity = {
    codebaseId,
    relativePath,
    language: languageForPath(relativePath) ?? language,
    kind: description.kind,
    qualifiedName: description.qualifiedName,
    signatureDiscriminator: signature,
  };
  return {
    kind: "symbol",
    symbolId: symbolId(identity),
    qualifiedName: identity.qualifiedName,
    entityScope: description.exported ? "global" : "resource",
  };
}

function describeAnyDeclaration(declaration: ts.Declaration): DeclarationDescription | undefined {
  if (ts.isFunctionDeclaration(declaration) && declaration.name) {
    return {
      node: declaration,
      signatureNode: declaration,
      kind: "function",
      qualifiedName: declaration.name.text,
      behaviorBearing: Boolean(declaration.body),
      exported: isExported(declaration),
    };
  }
  if (ts.isVariableDeclaration(declaration) && ts.isIdentifier(declaration.name)) {
    const initializer = declaration.initializer;
    return {
      node: declaration,
      signatureNode:
        initializer && (ts.isArrowFunction(initializer) || ts.isFunctionExpression(initializer))
          ? initializer
          : undefined,
      kind:
        initializer && (ts.isArrowFunction(initializer) || ts.isFunctionExpression(initializer))
          ? "named_arrow"
          : "constant",
      qualifiedName: declaration.name.text,
      behaviorBearing: Boolean(
        initializer && (ts.isArrowFunction(initializer) || ts.isFunctionExpression(initializer)),
      ),
      exported: Boolean(declaration.parent?.parent && isExported(declaration.parent.parent)),
    };
  }
  if (ts.isClassDeclaration(declaration) && declaration.name) {
    return {
      node: declaration,
      kind: "class",
      qualifiedName: declaration.name.text,
      behaviorBearing: false,
      exported: isExported(declaration),
    };
  }
  if (
    ts.isMethodDeclaration(declaration) ||
    ts.isConstructorDeclaration(declaration) ||
    ts.isGetAccessorDeclaration(declaration) ||
    ts.isSetAccessorDeclaration(declaration) ||
    ts.isPropertyDeclaration(declaration)
  ) {
    const parent = declaration.parent;
    if (ts.isClassDeclaration(parent) && parent.name) {
      return describeClassMember(declaration, parent.name.text, isExported(parent));
    }
  }
  return undefined;
}

function hasConcreteImplementation(declaration: ts.Declaration): boolean {
  if (ts.isFunctionDeclaration(declaration) || ts.isMethodDeclaration(declaration)) {
    return Boolean(declaration.body);
  }
  if (
    ts.isConstructorDeclaration(declaration) ||
    ts.isGetAccessorDeclaration(declaration) ||
    ts.isSetAccessorDeclaration(declaration)
  ) {
    return Boolean(declaration.body);
  }
  if (ts.isVariableDeclaration(declaration)) {
    return Boolean(declaration.initializer);
  }
  if (ts.isClassDeclaration(declaration)) return true;
  return false;
}

function findOwningSymbol(
  node: ts.Node,
  declarations: ReadonlyMap<ts.Node, CodeSymbolRecord>,
): CodeSymbolRecord | undefined {
  let current: ts.Node | undefined = node;
  while (current) {
    const record = declarations.get(current);
    if (record?.behaviorBearing) return record;
    current = current.parent;
  }
  return undefined;
}

function readEnvironmentKey(node: ts.Node): string | undefined {
  if (
    ts.isPropertyAccessExpression(node) &&
    ts.isPropertyAccessExpression(node.expression) &&
    ts.isIdentifier(node.expression.expression) &&
    node.expression.expression.text === "process" &&
    node.expression.name.text === "env"
  ) {
    return node.name.text;
  }
  if (
    ts.isElementAccessExpression(node) &&
    ts.isPropertyAccessExpression(node.expression) &&
    ts.isIdentifier(node.expression.expression) &&
    node.expression.expression.text === "process" &&
    node.expression.name.text === "env" &&
    node.argumentExpression &&
    ts.isStringLiteral(node.argumentExpression)
  ) {
    return node.argumentExpression.text;
  }
  return undefined;
}

function signatureDeclarationFor(node: ts.Node): ts.SignatureDeclaration | undefined {
  if (
    ts.isFunctionDeclaration(node) ||
    ts.isMethodDeclaration(node) ||
    ts.isConstructorDeclaration(node) ||
    ts.isGetAccessorDeclaration(node) ||
    ts.isSetAccessorDeclaration(node) ||
    ts.isArrowFunction(node) ||
    ts.isFunctionExpression(node)
  ) {
    return node;
  }
  if (
    ts.isVariableDeclaration(node) &&
    node.initializer &&
    (ts.isArrowFunction(node.initializer) || ts.isFunctionExpression(node.initializer))
  ) {
    return node.initializer;
  }
  return undefined;
}

function signatureText(checker: ts.TypeChecker, node: ts.SignatureDeclaration): string {
  const signature = checker.getSignatureFromDeclaration(node);
  return signature
    ? checker.signatureToString(signature, node, ts.TypeFormatFlags.NoTruncation)
    : "unknown";
}

function declarationSignature(node: ts.Node, sourceFile: ts.SourceFile): string {
  if ("name" in node && node.name && ts.isIdentifier(node.name as ts.Node)) {
    return (node.name as ts.Identifier).text;
  }
  return printer.printNode(ts.EmitHint.Unspecified, node, sourceFile).split("{")[0]?.trim() ?? "";
}

function createInMemoryCompilerHost(
  files: ReadonlyMap<string, string>,
  options: ts.CompilerOptions,
): ts.CompilerHost {
  const base = ts.createCompilerHost(options, true);
  const directories = virtualDirectories(files.keys());
  return {
    ...base,
    getCurrentDirectory: () => VIRTUAL_ROOT,
    fileExists: (fileName) =>
      files.has(normalizeVirtualPath(fileName)) || base.fileExists(fileName),
    directoryExists: (directoryName) =>
      directories.has(normalizeVirtualPath(directoryName)) ||
      Boolean(base.directoryExists?.(directoryName)),
    getDirectories: (directoryName) => {
      const normalized = normalizeVirtualPath(directoryName);
      const prefix = normalized.endsWith("/") ? normalized : `${normalized}/`;
      const children = Array.from(directories)
        .filter((candidate) => candidate.startsWith(prefix))
        .map((candidate) => candidate.slice(prefix.length).split("/")[0])
        .filter((candidate): candidate is string => Boolean(candidate));
      return Array.from(new Set([...(base.getDirectories?.(directoryName) ?? []), ...children]));
    },
    realpath: (fileName) => normalizeVirtualPath(fileName),
    readFile: (fileName) => files.get(normalizeVirtualPath(fileName)) ?? base.readFile(fileName),
    getSourceFile: (fileName, languageVersion, onError, shouldCreateNewSourceFile) => {
      const normalized = normalizeVirtualPath(fileName);
      const content = files.get(normalized);
      if (content !== undefined) {
        return ts.createSourceFile(
          normalized,
          content,
          languageVersion,
          true,
          scriptKindForPath(normalized),
        );
      }
      return base.getSourceFile(fileName, languageVersion, onError, shouldCreateNewSourceFile);
    },
    writeFile: () => undefined,
  };
}

function virtualDirectories(fileNames: Iterable<string>): ReadonlySet<string> {
  const directories = new Set<string>([VIRTUAL_ROOT]);
  for (const fileName of fileNames) {
    let current = path.posix.dirname(normalizeVirtualPath(fileName));
    while (current.startsWith(VIRTUAL_ROOT)) {
      directories.add(current);
      if (current === VIRTUAL_ROOT) break;
      current = path.posix.dirname(current);
    }
  }
  return directories;
}

function canonicalSyntax(node: ts.Node, sourceFile: ts.SourceFile): string {
  if (node.kind === ts.SyntaxKind.SemicolonToken || node.kind === ts.SyntaxKind.EndOfFileToken) {
    return "";
  }
  if (ts.isStringLiteralLike(node)) return `${node.kind}:${JSON.stringify(node.text)}`;
  if (ts.isNumericLiteral(node)) return `${node.kind}:${node.text.replaceAll("_", "")}`;
  if (ts.isIdentifier(node)) return `${node.kind}:${node.text}`;
  const children = node
    .getChildren(sourceFile)
    .map((child) => canonicalSyntax(child, sourceFile))
    .filter(Boolean);
  if (children.length === 0) return String(node.kind);
  return `${node.kind}[${children.join(",")}]`;
}

function isExported(node: ts.Node): boolean {
  return (
    hasModifier(node, ts.SyntaxKind.ExportKeyword) ||
    hasModifier(node, ts.SyntaxKind.DefaultKeyword)
  );
}

function hasModifier(node: ts.Node, kind: ts.SyntaxKind): boolean {
  return Boolean(
    ts.canHaveModifiers(node) && ts.getModifiers(node)?.some((item) => item.kind === kind),
  );
}

function propertyName(name: ts.PropertyName | undefined): string | undefined {
  if (!name) return undefined;
  if (ts.isIdentifier(name) || ts.isStringLiteral(name) || ts.isNumericLiteral(name)) {
    return name.text;
  }
  return undefined;
}

function languageForPath(filePath: string): CodeLanguage {
  return /\.[cm]?jsx?$/i.test(filePath) ? "javascript" : "typescript";
}

function scriptKindForPath(filePath: string): ts.ScriptKind {
  if (/\.tsx$/i.test(filePath)) return ts.ScriptKind.TSX;
  if (/\.[cm]?jsx$/i.test(filePath)) return ts.ScriptKind.JSX;
  if (/\.[cm]?js$/i.test(filePath)) return ts.ScriptKind.JS;
  return ts.ScriptKind.TS;
}

function virtualPath(filePath: string): string {
  return path.posix.join(VIRTUAL_ROOT, normalizeRelativePath(filePath));
}

function normalizeVirtualPath(filePath: string): string {
  return path.posix.normalize(filePath.replaceAll("\\", "/"));
}

function normalizeRelativePath(filePath: string): string {
  const normalized = path.posix.normalize(filePath.replaceAll("\\", "/")).replace(/^\.\//, "");
  if (normalized === ".." || normalized.startsWith("../") || path.posix.isAbsolute(normalized)) {
    throw new Error(`Codebase path must be relative: "${filePath}"`);
  }
  return normalized;
}

function relativeProjectPath(fileName: string): string | undefined {
  const normalized = normalizeVirtualPath(fileName);
  const relative = path.posix.relative(VIRTUAL_ROOT, normalized);
  return relative.startsWith("../") || path.posix.isAbsolute(relative) ? undefined : relative;
}

function symbolId(identity: CodeSymbolIdentity): string {
  return `code-symbol:${sha256(JSON.stringify(identity))}`;
}

function sourceChunkId(identity: CodeSymbolIdentity): string {
  return `symbol:${identity.kind}:${sha256(
    JSON.stringify([identity.qualifiedName, identity.signatureDiscriminator]),
  ).slice(0, 24)}`;
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function formatDiagnostic(diagnostic: ts.Diagnostic): string {
  return ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n");
}

function deduplicateUnresolved(
  unresolved: readonly UnresolvedCodeRelationship[],
): readonly UnresolvedCodeRelationship[] {
  const seen = new Set<string>();
  return unresolved.filter((item) => {
    const key = JSON.stringify(item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
