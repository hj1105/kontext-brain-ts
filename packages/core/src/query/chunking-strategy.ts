/**
 * Chunking strategy port. Chunk boundaries depend on domain/format:
 *   - Legal: article-level headers
 *   - Wikipedia/docs: paragraphs or fixed size
 *   - Markdown: `##`/`###` section headers
 */
export interface Chunk {
  readonly title: string;
  readonly fullText: string;
  readonly searchText: string;
}

export interface ChunkingStrategy {
  split(body: string): Chunk[];
}

function makeChunk(title: string, fullText: string): Chunk {
  return {
    title,
    fullText,
    searchText: `${title}\n${fullText.slice(0, 500)}`,
  };
}

/**
 * Regex header-based chunking. Boundaries are positions where splitPattern
 * matches (lookahead).
 */
export class RegexHeaderChunkingStrategy implements ChunkingStrategy {
  constructor(
    private readonly splitPattern: RegExp,
    private readonly titlePattern: RegExp | null = null,
    private readonly minChunkLength = 20,
  ) {
    if (!splitPattern.source.startsWith("(?=")) {
      // non-fatal warning — some regexes may not need lookahead
    }
  }

  split(body: string): Chunk[] {
    const parts = body
      .split(this.splitPattern)
      .map((p) => p.trim())
      .filter((p) => p.length >= this.minChunkLength);

    return parts.map((part, i) => {
      let title = `Chunk${i}`;
      if (this.titlePattern) {
        const match = this.titlePattern.exec(part);
        if (match?.[1]) title = match[1];
      }
      return makeChunk(title, part);
    });
  }

  /** Korean law article format: `### 제N조` */
  static readonly KOREAN_LAW_ARTICLE = new RegexHeaderChunkingStrategy(
    /(?=#{3,5}\s*제\d+조)/,
    /^#{3,5}\s*(제\d+조[^\n]*)/,
  );

  /** Markdown H2 sections. */
  static readonly MARKDOWN_H2 = new RegexHeaderChunkingStrategy(
    /(?=(?:^|\n)##\s)/,
    /(?:^|\n)##\s*([^\n]+)/,
  );
}

/**
 * Paragraph-based chunking. Splits on blank lines.
 * Suitable for Wikipedia/general documents.
 */
export class ParagraphChunkingStrategy implements ChunkingStrategy {
  constructor(
    private readonly minChunkLength = 100,
    private readonly maxChunkLength = 2000,
  ) {}

  split(body: string): Chunk[] {
    const paragraphs = body
      .split(/\n\s*\n/)
      .map((p) => p.trim())
      .filter((p) => p.length >= this.minChunkLength);

    const bounded = paragraphs.flatMap((p) =>
      p.length <= this.maxChunkLength ? [p] : chunkString(p, this.maxChunkLength),
    );

    return bounded.map((part, i) => {
      const firstLine = part.split("\n")[0]?.slice(0, 60) ?? `Chunk${i}`;
      return makeChunk(firstLine, part);
    });
  }
}

/**
 * Recursive size-based chunking (similar to LangChain RecursiveCharacterTextSplitter).
 */
export class RecursiveChunkingStrategy implements ChunkingStrategy {
  private readonly separators = ["\n\n", "\n", ". ", " "];

  constructor(
    private readonly targetSize = 1000,
    private readonly overlap = 200,
  ) {}

  split(body: string): Chunk[] {
    const chunks = this.recursiveSplit(body, this.separators);
    return chunks.map((text, i) => {
      const raw = text.slice(0, 60).replace(/\n/g, " ");
      const title = raw.trim() === "" ? `Chunk${i}` : raw;
      return makeChunk(title, text);
    });
  }

  private recursiveSplit(text: string, seps: readonly string[]): string[] {
    if (text.length <= this.targetSize) return [text];
    if (seps.length === 0) return chunkString(text, this.targetSize);

    const sep = seps[0] ?? " ";
    const parts = text.split(sep);
    const result: string[] = [];
    let buf = "";

    for (const part of parts) {
      if (buf.length + part.length + sep.length > this.targetSize) {
        if (buf.length > 0) {
          result.push(buf);
          const tail = buf.slice(Math.max(0, buf.length - this.overlap));
          buf = tail;
        }
        if (part.length > this.targetSize) {
          result.push(...this.recursiveSplit(part, seps.slice(1)));
        } else {
          buf += part + sep;
        }
      } else {
        buf += part + sep;
      }
    }
    if (buf.length > 0) result.push(buf);
    return result.filter((s) => s.trim().length > 0);
  }
}

function chunkString(s: string, size: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < s.length; i += size) {
    out.push(s.slice(i, i + size));
  }
  return out;
}
