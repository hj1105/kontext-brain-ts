import type { SyncMCPResult } from "@kontext-brain/loader";

export interface MCPSyncTarget {
  syncMCP(connectorName?: string): Promise<SyncMCPResult>;
}

export interface PeriodicMCPSyncOptions {
  readonly intervalMilliseconds: number;
  readonly runOnStart?: boolean;
}

/** Runs full-source refreshes serially, scheduling the next poll after the prior one settles. */
export class PeriodicMCPSyncService {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private stopped = true;
  private intervalMilliseconds = 0;
  private generation = 0;

  constructor(private readonly target: MCPSyncTarget) {}

  start(
    options: PeriodicMCPSyncOptions,
    onResult: (result: SyncMCPResult) => void = () => undefined,
    onError: (error: unknown) => void = () => undefined,
  ): () => void {
    if (!Number.isFinite(options.intervalMilliseconds) || options.intervalMilliseconds <= 0) {
      throw new Error("Periodic MCP sync interval must be greater than zero");
    }
    this.stop();
    this.stopped = false;
    this.intervalMilliseconds = options.intervalMilliseconds;
    const generation = this.generation;
    this.schedule(
      options.runOnStart === false ? options.intervalMilliseconds : 0,
      onResult,
      onError,
      generation,
    );
    return () => this.stop();
  }

  stop(): void {
    this.stopped = true;
    this.generation++;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  private schedule(
    delayMilliseconds: number,
    onResult: (result: SyncMCPResult) => void,
    onError: (error: unknown) => void,
    generation: number,
  ): void {
    if (this.stopped || generation !== this.generation) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.run(onResult, onError, generation);
    }, delayMilliseconds);
    this.timer.unref();
  }

  private async run(
    onResult: (result: SyncMCPResult) => void,
    onError: (error: unknown) => void,
    generation: number,
  ): Promise<void> {
    try {
      onResult(await this.target.syncMCP());
    } catch (error) {
      try {
        onError(error);
      } catch {
        // Observer failures must not disable future source refreshes.
      }
    } finally {
      this.schedule(this.intervalMilliseconds, onResult, onError, generation);
    }
  }
}
