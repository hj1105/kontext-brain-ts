export interface SourceChunk {
  readonly id: string;
  readonly sourceId: string;
  readonly ordinal: number;
  readonly text: string;
}

export interface SourceAnchor {
  readonly chunkId: string;
  readonly score: number;
  readonly rank: number;
}

export interface SourceContextPolicy {
  readonly windowCharacters: number;
  readonly maxContextCharacters: number;
}

export interface HydratedSourceContext {
  readonly id: string;
  readonly sourceId: string;
  readonly text: string;
  readonly score: number;
  readonly rank: number;
  readonly anchorIds: readonly string[];
  readonly chunkIds: readonly string[];
  readonly startOrdinal: number;
  readonly endOrdinal: number;
}

interface IndexedChunk extends SourceChunk {
  readonly sourceIndex: number;
}

interface SourceRange {
  readonly sourceId: string;
  readonly start: number;
  readonly end: number;
  readonly anchors: readonly SourceAnchor[];
}

/**
 * Restores source-native context around ranked retrieval anchors. Chunking stays
 * an indexing concern: callers receive de-duplicated source windows bounded by
 * one global context budget.
 */
export class SourceContextHydrator {
  private readonly chunksById = new Map<string, IndexedChunk>();
  private readonly chunksBySource = new Map<string, readonly SourceChunk[]>();

  constructor(
    chunks: readonly SourceChunk[],
    private readonly policy: SourceContextPolicy,
  ) {
    assertPolicy(policy);
    const grouped = new Map<string, SourceChunk[]>();
    for (const chunk of chunks) {
      const sourceChunks = grouped.get(chunk.sourceId) ?? [];
      sourceChunks.push(chunk);
      grouped.set(chunk.sourceId, sourceChunks);
    }
    for (const [sourceId, sourceChunks] of grouped) {
      const ordered = [...sourceChunks].sort(
        (left, right) => left.ordinal - right.ordinal || left.id.localeCompare(right.id),
      );
      this.chunksBySource.set(sourceId, ordered);
      ordered.forEach((chunk, sourceIndex) => {
        if (this.chunksById.has(chunk.id)) throw new Error(`Duplicate source chunk id ${chunk.id}`);
        this.chunksById.set(chunk.id, { ...chunk, sourceIndex });
      });
    }
  }

  hydrate(anchors: readonly SourceAnchor[]): HydratedSourceContext[] {
    const ranges = mergeRanges(
      anchors.flatMap((anchor) => {
        const chunk = this.chunksById.get(anchor.chunkId);
        if (!chunk) return [];
        const sourceChunks = this.chunksBySource.get(chunk.sourceId) ?? [];
        return [
          windowAround(sourceChunks, chunk.sourceIndex, anchor, this.policy.windowCharacters),
        ];
      }),
    ).sort(compareRanges);

    const output: HydratedSourceContext[] = [];
    let remainingCharacters = this.policy.maxContextCharacters;
    for (const range of ranges) {
      if (remainingCharacters <= 0) break;
      const sourceChunks = this.chunksBySource.get(range.sourceId) ?? [];
      const selectedChunks = sourceChunks.slice(range.start, range.end + 1);
      if (selectedChunks.length === 0) continue;
      const fullText = joinOverlappingText(selectedChunks.map((chunk) => chunk.text));
      const bestAnchor = [...range.anchors].sort(
        (left, right) => left.rank - right.rank || right.score - left.score,
      )[0];
      if (!bestAnchor) continue;
      const text =
        fullText.length <= remainingCharacters
          ? fullText
          : trimAroundAnchor(
              fullText,
              this.chunksById.get(bestAnchor.chunkId)?.text ?? "",
              remainingCharacters,
            );
      if (!text.trim()) continue;
      const startChunk = selectedChunks[0];
      const endChunk = selectedChunks[selectedChunks.length - 1];
      if (!startChunk || !endChunk) continue;
      output.push({
        id: `source-window:${range.sourceId}:${startChunk.ordinal}-${endChunk.ordinal}`,
        sourceId: range.sourceId,
        text,
        score: Math.max(...range.anchors.map((anchor) => anchor.score)),
        rank: output.length + 1,
        anchorIds: [...new Set(range.anchors.map((anchor) => anchor.chunkId))],
        chunkIds: selectedChunks.map((chunk) => chunk.id),
        startOrdinal: startChunk.ordinal,
        endOrdinal: endChunk.ordinal,
      });
      remainingCharacters -= text.length;
    }
    return output;
  }
}

function assertPolicy(policy: SourceContextPolicy): void {
  if (!Number.isInteger(policy.windowCharacters) || policy.windowCharacters <= 0) {
    throw new Error("windowCharacters must be a positive integer");
  }
  if (!Number.isInteger(policy.maxContextCharacters) || policy.maxContextCharacters <= 0) {
    throw new Error("maxContextCharacters must be a positive integer");
  }
}

function windowAround(
  chunks: readonly SourceChunk[],
  anchorIndex: number,
  anchor: SourceAnchor,
  targetCharacters: number,
): SourceRange {
  const anchorChunk = chunks[anchorIndex];
  if (!anchorChunk) throw new Error(`Unknown source anchor index ${anchorIndex}`);
  let start = anchorIndex;
  let end = anchorIndex;
  let characters = anchorChunk.text.length;
  let distance = 1;
  while (characters < targetCharacters) {
    let added = false;
    const left = anchorIndex - distance;
    const leftChunk = chunks[left];
    if (leftChunk && characters + leftChunk.text.length <= targetCharacters) {
      start = left;
      characters += leftChunk.text.length;
      added = true;
    }
    const right = anchorIndex + distance;
    const rightChunk = chunks[right];
    if (rightChunk && characters + rightChunk.text.length <= targetCharacters) {
      end = right;
      characters += rightChunk.text.length;
      added = true;
    }
    if (!added && left < 0 && right >= chunks.length) break;
    if (!added) {
      const candidates = [
        leftChunk ? { index: left, length: leftChunk.text.length } : null,
        rightChunk ? { index: right, length: rightChunk.text.length } : null,
      ].filter((candidate): candidate is { index: number; length: number } => candidate !== null);
      if (candidates.every((candidate) => characters + candidate.length > targetCharacters)) break;
    }
    distance += 1;
  }
  return { sourceId: anchorChunk.sourceId, start, end, anchors: [anchor] };
}

function mergeRanges(ranges: readonly SourceRange[]): SourceRange[] {
  const ordered = [...ranges].sort(
    (left, right) =>
      left.sourceId.localeCompare(right.sourceId) ||
      left.start - right.start ||
      left.end - right.end,
  );
  const output: SourceRange[] = [];
  for (const range of ordered) {
    const previous = output[output.length - 1];
    if (previous && previous.sourceId === range.sourceId && range.start <= previous.end) {
      output[output.length - 1] = {
        sourceId: previous.sourceId,
        start: Math.min(previous.start, range.start),
        end: Math.max(previous.end, range.end),
        anchors: [...previous.anchors, ...range.anchors],
      };
    } else {
      output.push(range);
    }
  }
  return output;
}

function compareRanges(left: SourceRange, right: SourceRange): number {
  const leftRank = Math.min(...left.anchors.map((anchor) => anchor.rank));
  const rightRank = Math.min(...right.anchors.map((anchor) => anchor.rank));
  const leftScore = Math.max(...left.anchors.map((anchor) => anchor.score));
  const rightScore = Math.max(...right.anchors.map((anchor) => anchor.score));
  return (
    leftRank - rightRank || rightScore - leftScore || left.sourceId.localeCompare(right.sourceId)
  );
}

function joinOverlappingText(parts: readonly string[]): string {
  let output = "";
  for (const rawPart of parts) {
    const part = rawPart.trim();
    if (!part) continue;
    if (!output) {
      output = part;
      continue;
    }
    const overlap = longestOverlap(output, part);
    output += overlap > 0 ? part.slice(overlap) : `\n\n${part}`;
  }
  return output;
}

function longestOverlap(left: string, right: string): number {
  const limit = Math.min(512, left.length, right.length);
  for (let length = limit; length >= 32; length -= 1) {
    if (left.endsWith(right.slice(0, length))) return length;
  }
  return 0;
}

function trimAroundAnchor(text: string, anchorText: string, limit: number): string {
  if (text.length <= limit) return text;
  const anchorStart = anchorText
    ? text.indexOf(anchorText.slice(0, Math.min(128, anchorText.length)))
    : -1;
  const center =
    anchorStart >= 0
      ? anchorStart + Math.floor(anchorText.length / 2)
      : Math.floor(text.length / 2);
  const start = Math.max(0, Math.min(text.length - limit, center - Math.floor(limit / 2)));
  return text.slice(start, start + limit).trim();
}
