import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LocalWorkspaceObservationService } from "../src/index.js";

describe("LocalWorkspaceObservationService", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("keeps one observation in flight while a workspace poll is pending", async () => {
    let complete: (() => void) | undefined;
    const pending = new Promise<void>((resolve) => {
      complete = resolve;
    });
    const observe = vi
      .fn()
      .mockImplementationOnce(() => pending)
      .mockResolvedValue(undefined);
    const service = new LocalWorkspaceObservationService(
      { list: vi.fn().mockResolvedValue([{ workspacePath: "/workspace" }]) } as never,
      { observe } as never,
    );
    const stop = service.start(100);

    await vi.advanceTimersByTimeAsync(100);
    expect(observe).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(500);
    expect(observe).toHaveBeenCalledTimes(1);

    complete?.();
    await vi.advanceTimersByTimeAsync(100);
    expect(observe).toHaveBeenCalledTimes(2);
    stop();
  });

  it("isolates reporter failures and retries observation on the next interval", async () => {
    const observationError = new Error("observation failed");
    const observe = vi.fn().mockRejectedValueOnce(observationError).mockResolvedValue(undefined);
    const onError = vi.fn(() => {
      throw new Error("reporter failed");
    });
    const service = new LocalWorkspaceObservationService(
      { list: vi.fn().mockResolvedValue([{ workspacePath: "/workspace" }]) } as never,
      { observe } as never,
    );
    const stop = service.start(100, onError);

    await vi.advanceTimersByTimeAsync(100);
    expect(onError).toHaveBeenCalledWith(observationError);
    await vi.advanceTimersByTimeAsync(100);
    expect(observe).toHaveBeenCalledTimes(2);
    stop();
  });

  it("rejects a duplicate loop and permits an explicit stop followed by restart", () => {
    const service = new LocalWorkspaceObservationService(
      { list: vi.fn().mockResolvedValue([]) } as never,
      { observe: vi.fn() } as never,
    );
    const stop = service.start(100);

    expect(() => service.start(100)).toThrow("Workspace observation is already running");
    stop();

    const stopRestarted = service.start(100);
    stopRestarted();
  });
});
