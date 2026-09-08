import type { Dir } from "node:fs";
import { opendir } from "node:fs/promises";
import { afterEach, expect, it, vi } from "vitest";
import { listHashedRecordFiles } from "../src/hashed-record-inventory.js";

vi.mock("node:fs/promises", () => ({ opendir: vi.fn() }));
afterEach(() => vi.resetAllMocks());

function directory(entries: { name: string; isFile: () => boolean }[]) {
  const closed = vi.fn();
  vi.mocked(opendir).mockResolvedValue({
    async *[Symbol.asyncIterator]() {
      try {
        yield* entries;
      } finally {
        closed();
      }
    },
  } as unknown as Dir);
  return closed;
}

it("returns empty only for a missing directory and propagates unavailable storage", async () => {
  vi.mocked(opendir).mockRejectedValueOnce(Object.assign(new Error("missing"), { code: "ENOENT" }));
  expect(await listHashedRecordFiles("/fixture")).toEqual([]);
  vi.mocked(opendir).mockRejectedValueOnce(Object.assign(new Error("denied"), { code: "EACCES" }));
  await expect(listHashedRecordFiles("/fixture")).rejects.toThrow("denied");
});

it("ignores temporary names, sorts canonical records and closes the iterator", async () => {
  const names = [`${"b".repeat(64)}.json`, "pending.tmp", `${"a".repeat(64)}.json`];
  const closed = directory(names.map((name) => ({ name, isFile: () => true })));
  expect(await listHashedRecordFiles("/fixture")).toEqual([names[2], names[0]]);
  expect(closed).toHaveBeenCalledOnce();
});

it("refuses non-file records and closes the iterator", async () => {
  const closed = directory([{ name: `${"a".repeat(64)}.json`, isFile: () => false }]);
  await expect(listHashedRecordFiles("/fixture")).rejects.toThrow("non-file record");
  expect(closed).toHaveBeenCalledOnce();
});

it("bounds all inspected entries, including ignored names, without returning a partial list", async () => {
  const closed = directory(
    Array.from({ length: 10_001 }, () => ({ name: "temporary", isFile: () => true })),
  );
  await expect(listHashedRecordFiles("/fixture")).rejects.toThrow("bounded scan limit");
  expect(closed).toHaveBeenCalledOnce();
});
