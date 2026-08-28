import { describe, expect, it } from "vitest";
import { type SourceChunk, SourceContextHydrator } from "../src/index.js";

describe("SourceContextHydrator", () => {
  it("restores and de-duplicates source-native context around an anchor", () => {
    const hydrator = new SourceContextHydrator(
      [
        chunk("c-0", "doc", 0, `alpha ${"x".repeat(36)}`),
        chunk("c-1", "doc", 1, `${"x".repeat(36)} beta`),
        chunk("c-2", "doc", 2, "gamma"),
      ],
      { windowCharacters: 100, maxContextCharacters: 100 },
    );

    const result = hydrator.hydrate([{ chunkId: "c-1", score: 0.8, rank: 1 }]);

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      sourceId: "doc",
      anchorIds: ["c-1"],
      chunkIds: ["c-0", "c-1", "c-2"],
      startOrdinal: 0,
      endOrdinal: 2,
    });
    expect(result[0]?.text).toContain("alpha");
    expect(result[0]?.text).toContain("beta");
    expect(result[0]?.text.match(/x/g)).toHaveLength(36);
  });

  it("merges overlapping windows and keeps the best anchor rank", () => {
    const hydrator = new SourceContextHydrator(numberedChunks("doc", 8, 20), {
      windowCharacters: 70,
      maxContextCharacters: 200,
    });

    const result = hydrator.hydrate([
      { chunkId: "doc-2", score: 0.7, rank: 2 },
      { chunkId: "doc-3", score: 0.9, rank: 1 },
    ]);

    expect(result).toHaveLength(1);
    expect(result[0]?.anchorIds).toEqual(["doc-2", "doc-3"]);
    expect(result[0]?.score).toBe(0.9);
    expect(result[0]?.rank).toBe(1);
  });

  it("keeps sources separate and enforces one global character budget", () => {
    const hydrator = new SourceContextHydrator(
      [...numberedChunks("a", 4, 24), ...numberedChunks("b", 4, 24)],
      { windowCharacters: 80, maxContextCharacters: 90 },
    );

    const result = hydrator.hydrate([
      { chunkId: "a-1", score: 0.9, rank: 1 },
      { chunkId: "b-1", score: 0.8, rank: 2 },
    ]);

    expect(result.map((item) => item.sourceId)).toEqual(["a", "b"]);
    expect(result.reduce((total, item) => total + item.text.length, 0)).toBeLessThanOrEqual(90);
  });

  it("does not jump across an oversized adjacent chunk", () => {
    const hydrator = new SourceContextHydrator(
      [
        chunk("c-0", "doc", 0, "far-left"),
        chunk("c-1", "doc", 1, "x".repeat(100)),
        chunk("c-2", "doc", 2, "anchor"),
        chunk("c-3", "doc", 3, "right"),
      ],
      { windowCharacters: 40, maxContextCharacters: 40 },
    );

    const result = hydrator.hydrate([{ chunkId: "c-2", score: 1, rank: 1 }]);

    expect(result[0]?.chunkIds).toEqual(["c-2", "c-3"]);
    expect(result[0]?.text.length).toBeLessThanOrEqual(40);
    expect(result[0]?.text).not.toContain("far-left");
  });
});

function chunk(id: string, sourceId: string, ordinal: number, text: string): SourceChunk {
  return { id, sourceId, ordinal, text };
}

function numberedChunks(sourceId: string, count: number, characters: number): SourceChunk[] {
  return Array.from({ length: count }, (_, ordinal) =>
    chunk(`${sourceId}-${ordinal}`, sourceId, ordinal, `${ordinal}:${"x".repeat(characters - 2)}`),
  );
}
