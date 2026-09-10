import { describe, expect, it } from "vitest";
import { mapWithConcurrency, stratifiedSample } from "../src/index.js";

describe("stratifiedSample", () => {
  it("takes round-robin across groups so a small source is represented", () => {
    const items = [
      ...Array.from({ length: 50 }, (_, i) => ({ group: "a", i })),
      ...Array.from({ length: 2 }, (_, i) => ({ group: "b", i })),
    ];
    const sample = stratifiedSample(items, 6, (item) => item.group);
    expect(sample.filter((item) => item.group === "b")).toHaveLength(2);
    expect(sample).toHaveLength(6);
    expect(stratifiedSample(items.slice(0, 3), 10, (item) => item.group)).toHaveLength(3);
  });
});

describe("mapWithConcurrency", () => {
  it("keeps order and never exceeds the limit", async () => {
    let inFlight = 0;
    let peak = 0;
    const results = await mapWithConcurrency([1, 2, 3, 4, 5, 6, 7], 2, async (n) => {
      inFlight += 1;
      peak = Math.max(peak, inFlight);
      await new Promise((resolve) => setTimeout(resolve, 8 - n));
      inFlight -= 1;
      return n * 10;
    });
    expect(results).toEqual([10, 20, 30, 40, 50, 60, 70]);
    expect(peak).toBeLessThanOrEqual(2);
  });
});
