import type { LocalPostWriteObserver } from "./local-post-write-observer.js";
import type { WriteAuthorizationBindingStore } from "./task-workflow-tools.js";

export class LocalWorkspaceObservationService {
  private inFlight: Promise<void> | undefined;
  private timer: NodeJS.Timeout | undefined;

  constructor(
    private readonly bindings: WriteAuthorizationBindingStore,
    private readonly observer: LocalPostWriteObserver,
  ) {}

  async observeAll(observedAt: string): Promise<void> {
    for (const { workspacePath } of await this.bindings.list()) {
      await this.observer.observe({
        cwd: workspacePath,
        toolName: "workspace_poll",
        observedAt,
      });
    }
  }

  start(
    intervalMilliseconds = 2_000,
    onError: (error: unknown) => void = () => undefined,
  ): () => void {
    if (this.timer) throw new Error("Workspace observation is already running");
    const observe = (): void => {
      if (this.inFlight) return;
      this.inFlight = this.observeAll(new Date().toISOString())
        .catch((error) => {
          try {
            onError(error);
          } catch {
            // Error reporting must not terminate the observation loop.
          }
        })
        .finally(() => {
          this.inFlight = undefined;
        });
    };
    const timer = setInterval(observe, intervalMilliseconds);
    timer.unref();
    this.timer = timer;
    return () => {
      if (this.timer !== timer) return;
      clearInterval(timer);
      this.timer = undefined;
    };
  }
}
