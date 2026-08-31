import type { LocalPostWriteObserver } from "./local-post-write-observer.js";
import type { WriteAuthorizationBindingStore } from "./task-workflow-tools.js";

export class LocalWorkspaceObservationService {
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
    const observe = (): void => {
      void this.observeAll(new Date().toISOString()).catch(onError);
    };
    const timer = setInterval(observe, intervalMilliseconds);
    timer.unref();
    return () => clearInterval(timer);
  }
}
