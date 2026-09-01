import type { PreparedTaskContextStore, TaskContextStateProvider } from "@kontext-brain/context";
import type {
  DurableVerificationCoordinator,
  TaskCompletionArtifactStore,
  VerificationExecution,
  VerificationRetryQueue,
} from "@kontext-brain/orchestrator";

export interface RecoverAvailableVerificationInput {
  readonly now: string;
  readonly nextAttemptAt: string;
  readonly leaseExpiresAt: string;
  readonly perTaskLimit?: number;
}

export class LocalVerificationRecoveryService {
  constructor(
    private readonly currentState: TaskContextStateProvider,
    private readonly preparedTasks: PreparedTaskContextStore,
    private readonly artifacts: TaskCompletionArtifactStore,
    private readonly retryQueue: VerificationRetryQueue,
    private readonly verification: DurableVerificationCoordinator,
  ) {}

  async recoverAvailable(
    input: RecoverAvailableVerificationInput,
  ): Promise<readonly VerificationExecution[]> {
    const queued = await this.retryQueue.list("queued");
    const taskIds = Array.from(
      new Set(queued.filter((job) => job.nextAttemptAt <= input.now).map((job) => job.taskId)),
    ).sort((left, right) => left.localeCompare(right));
    const recovered: VerificationExecution[] = [];
    for (const taskId of taskIds) {
      const prepared = await this.preparedTasks.get(taskId);
      if (!prepared) continue;
      let codeRevision: string;
      try {
        codeRevision = (await this.currentState.getCurrent(taskId)).codeRevision;
      } catch {
        continue;
      }
      const executions = await this.verification.retryAvailable({
        taskId,
        currentCodeRevision: codeRevision,
        currentContextDigest: prepared.snapshot.contextDigest,
        observedAt: input.now,
        nextAttemptAt: input.nextAttemptAt,
        leaseExpiresAt: input.leaseExpiresAt,
        limit: input.perTaskLimit,
      });
      if (executions.length > 0) {
        await this.artifacts.putVerificationRuns(
          taskId,
          executions.map((execution) => execution.run),
        );
        recovered.push(...executions);
      }
    }
    return recovered;
  }

  start(
    intervalMilliseconds = 30_000,
    onError: (error: unknown) => void = () => undefined,
  ): () => void {
    const recover = (): void => {
      const now = Date.now();
      void this.recoverAvailable({
        now: new Date(now).toISOString(),
        nextAttemptAt: new Date(now + intervalMilliseconds).toISOString(),
        leaseExpiresAt: new Date(now + intervalMilliseconds * 2).toISOString(),
      }).catch(onError);
    };
    recover();
    const timer = setInterval(recover, intervalMilliseconds);
    timer.unref();
    return () => clearInterval(timer);
  }
}
