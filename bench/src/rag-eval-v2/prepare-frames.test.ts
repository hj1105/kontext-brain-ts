import { describe, expect, it } from "vitest";
import { parseDelimited, splitWikipediaLinks } from "./prepare-frames.js";

describe("FRAMES TSV parser", () => {
  it("handles quoted tabs, newlines, and escaped quotes", () => {
    expect(parseDelimited('a\tb\n"x\ty"\t"line 1\nline ""2"""\n', "\t")).toEqual([
      ["a", "b"],
      ["x\ty", 'line 1\nline "2"'],
    ]);
  });

  it("expands malformed cells containing multiple comma-separated URLs", () => {
    expect(
      splitWikipediaLinks(
        "https://en.wikipedia.org/wiki/Tim_Salmon, https://en.wikipedia.org/wiki/Troy_Glaus, ",
      ),
    ).toEqual([
      "https://en.wikipedia.org/wiki/Tim_Salmon",
      "https://en.wikipedia.org/wiki/Troy_Glaus",
    ]);
  });
});
