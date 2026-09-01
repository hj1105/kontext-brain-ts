import { createHash } from "node:crypto";
import type {
  CodeAnalysisInput,
  CodeFileAnalysis,
  CodeSymbolIdentity,
  CodeSymbolKind,
  CodeSymbolRecord,
  LanguageCodeProvider,
} from "./domain.js";

/**
 * Python has no type checker in this pipeline, so this provider reports
 * "syntactic" support rather than the "certified" support the TypeScript
 * provider claims. Behaviour identity therefore comes from a canonical token
 * stream instead of a typed syntax tree.
 *
 * The token stream drops comments, docstrings, blank lines, and inner
 * whitespace, and normalizes string and numeric literals, so a format-only edit
 * leaves contentHash unchanged exactly as it does for TypeScript.
 */
export class PythonCodeProvider implements LanguageCodeProvider {
  readonly language = "python" as const;
  readonly semanticSupport = "syntactic" as const;

  analyze(input: CodeAnalysisInput): CodeFileAnalysis {
    const relativePath = normalizeRelativePath(input.targetPath);
    const file = input.files.find(
      (candidate) => normalizeRelativePath(candidate.path) === relativePath,
    );
    if (file === undefined) {
      throw new Error(`Target file "${relativePath}" is missing from the Codebase snapshot`);
    }
    const lines = file.content.split(/\r?\n/);
    const declarations = parseDeclarations(lines);
    const symbols: CodeSymbolRecord[] = [];
    let position = 1;

    symbols.push(
      record({
        codebaseId: input.codebaseId,
        relativePath,
        kind: "module",
        qualifiedName: "<module>",
        signature: relativePath,
        bodyLines: lines,
        text: file.content,
        behaviorBearing: false,
        exported: true,
        position: 0,
      }),
    );

    for (const declaration of declarations) {
      const text = lines.slice(declaration.startLine, declaration.endLine + 1).join("\n");
      symbols.push(
        record({
          codebaseId: input.codebaseId,
          relativePath,
          kind: declaration.kind,
          qualifiedName: declaration.qualifiedName,
          signature: declaration.signature,
          bodyLines: lines.slice(declaration.startLine, declaration.endLine + 1),
          text,
          behaviorBearing: declaration.kind !== "class" && declaration.kind !== "constant",
          exported: declaration.exported,
          position: position++,
        }),
      );
    }

    return {
      codebaseId: input.codebaseId,
      relativePath,
      language: "python",
      sourceText: file.content,
      contentHash: sha256(canonicalTokens(lines)),
      symbols,
      relationships: [],
      unresolvedRelationships: [],
      diagnostics: [],
    };
  }
}

interface Declaration {
  readonly kind: CodeSymbolKind;
  readonly qualifiedName: string;
  readonly signature: string;
  readonly startLine: number;
  readonly endLine: number;
  readonly exported: boolean;
}

const definitionPattern = /^(\s*)(async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(.*)$/;
const constantPattern = /^([A-Z][A-Z0-9_]*)\s*(?::[^=]+)?=\s*(.+)$/;

function parseDeclarations(lines: readonly string[]): readonly Declaration[] {
  const declarations: Declaration[] = [];
  const open: { indent: number; name: string; kind: CodeSymbolKind }[] = [];

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index] ?? "";
    if (!line.trim() || line.trim().startsWith("#")) continue;
    const indent = line.length - line.trimStart().length;
    while (open.length > 0 && indent <= (open.at(-1)?.indent ?? 0)) open.pop();

    const match = definitionPattern.exec(line);
    if (match) {
      const [, , keyword, name, rest] = match;
      const enclosing = open.at(-1);
      const isClass = keyword === "class";
      // Only a function directly inside a class is a method; anything nested
      // deeper is a closure and is not addressable as its own logic unit.
      const kind: CodeSymbolKind = isClass
        ? "class"
        : enclosing?.kind === "class"
          ? "method"
          : "function";
      const qualifiedName = enclosing ? `${enclosing.name}.${name}` : (name as string);
      if (!isClass && open.some((entry) => entry.kind !== "class")) {
        continue;
      }
      declarations.push({
        kind,
        qualifiedName,
        signature: declarationSignature(lines, index, rest ?? ""),
        startLine: index,
        endLine: blockEnd(lines, index, indent),
        exported: !(name as string).startsWith("_"),
      });
      open.push({ indent, name: qualifiedName, kind });
      continue;
    }

    if (indent === 0) {
      const constant = constantPattern.exec(line.trim());
      if (constant) {
        declarations.push({
          kind: "constant",
          qualifiedName: constant[1] as string,
          signature: `${constant[1]} = <value>`,
          startLine: index,
          endLine: index,
          exported: true,
        });
      }
    }
  }
  return declarations;
}

/**
 * A definition header can wrap across lines, so the signature is read to its
 * balanced closing parenthesis and then normalized to a single spacing form.
 */
function declarationSignature(lines: readonly string[], startLine: number, rest: string): string {
  let text = rest;
  let depth = countDepth(rest);
  let index = startLine;
  while (depth > 0 && index + 1 < lines.length) {
    index += 1;
    const next = lines[index] ?? "";
    text += ` ${next.trim()}`;
    depth += countDepth(next);
  }
  return text.replace(/#.*$/, "").replace(/:\s*$/, "").replace(/\s+/g, " ").trim();
}

function countDepth(text: string): number {
  let depth = 0;
  for (const character of text) {
    if (character === "(" || character === "[" || character === "{") depth += 1;
    if (character === ")" || character === "]" || character === "}") depth -= 1;
  }
  return depth;
}

function blockEnd(lines: readonly string[], startLine: number, indent: number): number {
  let end = startLine;
  for (let index = startLine + 1; index < lines.length; index += 1) {
    const line = lines[index] ?? "";
    if (!line.trim()) continue;
    const lineIndent = line.length - line.trimStart().length;
    if (lineIndent <= indent) break;
    end = index;
  }
  return end;
}

function record(input: {
  readonly codebaseId: string;
  readonly relativePath: string;
  readonly kind: CodeSymbolKind;
  readonly qualifiedName: string;
  readonly signature: string;
  readonly bodyLines: readonly string[];
  readonly text: string;
  readonly behaviorBearing: boolean;
  readonly exported: boolean;
  readonly position: number;
}): CodeSymbolRecord {
  const identity: CodeSymbolIdentity = {
    codebaseId: input.codebaseId,
    relativePath: input.relativePath,
    language: "python",
    kind: input.kind,
    qualifiedName: input.qualifiedName,
    signatureDiscriminator: input.signature,
  };
  return {
    symbolId: `code-symbol:${sha256(JSON.stringify(identity))}`,
    sourceChunkId: `symbol:${identity.kind}:${sha256(
      JSON.stringify([identity.qualifiedName, identity.signatureDiscriminator]),
    ).slice(0, 24)}`,
    identity,
    behaviorBearing: input.behaviorBearing,
    exported: input.exported,
    signature: input.signature,
    contentHash: sha256(canonicalTokens(input.bodyLines)),
    text: input.text,
    position: input.position,
    semanticSupport: "syntactic",
  };
}

/**
 * Emits a canonical token stream. Indentation becomes relative depth markers so
 * re-indenting a block does not change its behaviour identity, while a real
 * nesting change still does.
 */
export function canonicalTokens(lines: readonly string[]): string {
  const tokens: string[] = [];
  const indents: number[] = [];
  let baseIndent: number | undefined;
  let expectDocstring = true;

  for (const rawLine of lines) {
    const stripped = stripComment(rawLine);
    if (!stripped.trim()) continue;
    const indent = stripped.length - stripped.trimStart().length;
    baseIndent ??= indent;
    const relative = Math.max(0, indent - baseIndent);
    while (indents.length > 0 && relative < (indents.at(-1) ?? 0)) {
      indents.pop();
      tokens.push("DEDENT");
    }
    if (relative > (indents.at(-1) ?? 0)) {
      indents.push(relative);
      tokens.push("INDENT");
    }
    const lineTokens = tokenizeLine(stripped.trim());
    // A bare string statement opening a block is documentation, matching how
    // the TypeScript provider drops comments before hashing.
    if (expectDocstring && lineTokens.length === 1 && lineTokens[0]?.startsWith("STR:")) {
      expectDocstring = false;
      continue;
    }
    expectDocstring =
      /^(?:def|class|async)\b/.test(stripped.trim()) || lineTokens.at(-1) === "OP::";
    tokens.push(...lineTokens, "NEWLINE");
  }
  return tokens.join(" ");
}

function stripComment(line: string): string {
  let quote: string | undefined;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (quote) {
      if (character === "\\") index += 1;
      else if (character === quote) quote = undefined;
      continue;
    }
    if (character === '"' || character === "'") quote = character;
    else if (character === "#") return line.slice(0, index);
  }
  return line;
}

function tokenizeLine(line: string): string[] {
  const tokens: string[] = [];
  let index = 0;
  while (index < line.length) {
    const character = line[index] as string;
    if (/\s/.test(character)) {
      index += 1;
      continue;
    }
    const triple = line.slice(index, index + 3);
    if (triple === '"""' || triple === "'''") {
      const close = line.indexOf(triple, index + 3);
      const end = close === -1 ? line.length : close + 3;
      tokens.push(`STR:${JSON.stringify(line.slice(index + 3, Math.max(index + 3, end - 3)))}`);
      index = end;
      continue;
    }
    if (character === '"' || character === "'") {
      let cursor = index + 1;
      let value = "";
      while (cursor < line.length && line[cursor] !== character) {
        if (line[cursor] === "\\") cursor += 1;
        value += line[cursor] ?? "";
        cursor += 1;
      }
      tokens.push(`STR:${JSON.stringify(value)}`);
      index = cursor + 1;
      continue;
    }
    const remainder = line.slice(index);
    const number = /^\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?/.exec(remainder);
    if (number) {
      tokens.push(`NUM:${Number(number[0].replaceAll("_", ""))}`);
      index += number[0].length;
      continue;
    }
    const name = /^[A-Za-z_][A-Za-z0-9_]*/.exec(remainder);
    if (name) {
      // A string prefix such as f or rb belongs to the literal that follows.
      const next = line[index + name[0].length];
      if (/^[A-Za-z]{1,2}$/.test(name[0]) && (next === '"' || next === "'")) {
        index += name[0].length;
        continue;
      }
      tokens.push(`NAME:${name[0]}`);
      index += name[0].length;
      continue;
    }
    const operator =
      /^(?:\*\*=|\/\/=|>>=|<<=|\.\.\.|==|!=|<=|>=|->|:=|\*\*|\/\/|<<|>>|[+\-*/%@&|^~<>()[\]{},:.;=])/.exec(
        remainder,
      );
    if (operator) {
      tokens.push(`OP:${operator[0]}`);
      index += operator[0].length;
      continue;
    }
    index += 1;
  }
  return tokens;
}

function normalizeRelativePath(value: string): string {
  return value.replaceAll("\\", "/").replace(/^\.\//, "");
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
