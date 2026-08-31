import type { SyncMCPResult } from "@kontext-brain/loader";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PeriodicMCPSyncService } from "../src/index.js";

const noChanges: SyncMCPResult = {
  connectorsSynced: 3,
  resourcesAdded: 0,
  resourcesUpdated: 0,
  resourcesRemoved: 0,
  resourcesClassified: 0,
  resourcesUnmapped: 0,
};

describe("PeriodicMCPSyncService", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("runs immediately and never overlaps a slow refresh", async () => {
    let resolveFirst: ((result: SyncMCPResult) => void) | undefined;
    const first = new Promise<SyncMCPResult>((resolve) => {
      resolveFirst = resolve;
    });
    const syncMCP = vi
      .fn()
      .mockImplementationOnce(() => first)
      .mockResolvedValue(noChanges);
    const service = new PeriodicMCPSyncService({ syncMCP });
    const stop = service.start({ intervalMilliseconds: 100 });

    await vi.advanceTimersByTimeAsync(0);
    expect(syncMCP).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(500);
    expect(syncMCP).toHaveBeenCalledTimes(1);

    resolveFirst?.(noChanges);
    await vi.advanceTimersByTimeAsync(99);
    expect(syncMCP).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(syncMCP).toHaveBeenCalledTimes(2);
    stop();
  });

  it("reports an error and retries on the next interval", async () => {
    const error = new Error("connector unavailable");
    const syncMCP = vi.fn().mockRejectedValueOnce(error).mockResolvedValue(noChanges);
    const onError = vi.fn();
    const service = new PeriodicMCPSyncService({ syncMCP });
    const stop = service.start({ intervalMilliseconds: 100 }, undefined, onError);

    await vi.advanceTimersByTimeAsync(0);
    expect(onError).toHaveBeenCalledWith(error);
    await vi.advanceTimersByTimeAsync(100);
    expect(syncMCP).toHaveBeenCalledTimes(2);
    stop();
  });
});
