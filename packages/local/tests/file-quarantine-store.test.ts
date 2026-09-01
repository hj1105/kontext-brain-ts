import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { createQuarantineRecord } from "@kontext-brain/orchestrator";
import type { QuarantineRecord } from "@kontext-brain/spec";
import { afterEach, describe, expect, it } from "vitest";
import { FileQuarantineStore } from "../src/index.js";

const temporaryDirectories: string[] = [];
const record: QuarantineRecord = createQuarantineRecord({
  taskId: "task:quarantine",
  workItemId: "work-item:quarantine",
  codeRevision: "commit:result",
  contextDigest: "context:current",
  paths: ["src/outside.ts"],
  symbolIds: ["symbol:outside"],
  reasons: ["path_out_of_scope"],
  observedAt: "2026-08-28T07:00:00.000Z",
});

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("FileQuarantineStore", () => {
  it("persists private active quarantine and release audit state across restarts", async () => {
    const directory = await temporaryDirectory();
    const first = new FileQuarantineStore(directory);
    await first.put(record);
    const reopened = new FileQuarantineStore(directory);

    expect(await reopened.list("active")).toEqual([record]);
    const released = await reopened.release(
      record.quarantineId,
      "user:owner",
      "2026-08-28T07:30:00.000Z",
    );

    expect(released).toEqual(
      expect.objectContaining({ status: "released", releasedBy: "user:owner" }),
    );
    expect(await new FileQuarantineStore(directory).get(record.quarantineId)).toEqual(released);
    expect((await stat(reopened.filePath(record.quarantineId))).mode & 0o777).toBe(0o600);
  });

  it("detects payload tampering", async () => {
    const store = new FileQuarantineStore(await temporaryDirectory());
    await store.put(record);
    const filePath = store.filePath(record.quarantineId);
    const envelope = JSON.parse(await readFile(filePath, "utf8"));
    envelope.payload.paths = ["src/tampered.ts"];
    await writeFile(filePath, JSON.stringify(envelope), "utf8");

    await expect(store.get(record.quarantineId)).rejects.toThrow("digest mismatch");
  });
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-quarantine-"));
  temporaryDirectories.push(directory);
  return directory;
}
