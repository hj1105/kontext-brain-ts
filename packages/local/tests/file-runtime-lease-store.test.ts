import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { RuntimeLease } from "@kontext-brain/orchestrator";
import { afterEach, describe, expect, it } from "vitest";
import { FileRuntimeLeaseStore } from "../src/index.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("FileRuntimeLeaseStore", () => {
  it("persists active leases across instances and releases them durably", async () => {
    const directory = await temporaryDirectory();
    const first = new FileRuntimeLeaseStore(directory);
    expect(await first.acquire(lease("lease:one", "symbol:shared"))).toBe(true);

    const reopened = new FileRuntimeLeaseStore(directory);
    expect(await reopened.listActive("2026-08-29T00:01:00.000Z")).toHaveLength(1);
    expect(await reopened.acquire(lease("lease:two", "symbol:shared"))).toBe(false);

    await reopened.release("lease:one", "2026-08-29T00:02:00.000Z");
    expect(
      await new FileRuntimeLeaseStore(directory).acquire(lease("lease:two", "symbol:shared")),
    ).toBe(true);
  });

  it("serializes concurrent acquisition for conflicting scopes", async () => {
    const directory = await temporaryDirectory();
    const left = new FileRuntimeLeaseStore(directory);
    const right = new FileRuntimeLeaseStore(directory);
    const results = await Promise.all([
      left.acquire(lease("lease:left", "symbol:shared")),
      right.acquire(lease("lease:right", "symbol:shared")),
    ]);

    expect(results.filter(Boolean)).toHaveLength(1);
    expect(await left.listActive("2026-08-29T00:01:00.000Z")).toHaveLength(1);
  });
});

function lease(leaseId: string, symbolId: string): RuntimeLease {
  return {
    leaseId,
    taskId: "task:runtime",
    workItemId: `work-item:${leaseId}`,
    provider: "codex",
    workspacePath: `/workspace/${leaseId}`,
    symbolIds: [symbolId],
    paths: [`src/${leaseId}.ts`],
    acquiredAt: "2026-08-29T00:00:00.000Z",
    expiresAt: "2026-08-29T00:15:00.000Z",
  };
}

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-runtime-leases-"));
  temporaryDirectories.push(directory);
  return directory;
}
