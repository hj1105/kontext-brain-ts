import { spawn } from "node:child_process";

export interface RuntimeCommandInput {
  readonly executionId: string;
  readonly command: string;
  readonly args: readonly string[];
  readonly cwd: string;
  readonly stdin?: string;
  readonly environment: Readonly<Record<string, string>>;
  readonly timeoutMilliseconds: number;
  readonly signal?: AbortSignal;
}

export interface RuntimeCommandResult {
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
}

export interface RuntimeCommandRunner {
  run(input: RuntimeCommandInput): Promise<RuntimeCommandResult>;
  terminate(executionId: string): Promise<void>;
}

export class SpawnRuntimeCommandRunner implements RuntimeCommandRunner {
  private readonly active = new Map<string, ReturnType<typeof spawn>>();

  async run(input: RuntimeCommandInput): Promise<RuntimeCommandResult> {
    if (input.signal?.aborted) throw new Error("Runtime command was cancelled before start");
    return new Promise((resolve, reject) => {
      const child = spawn(input.command, input.args, {
        cwd: input.cwd,
        env: input.environment,
        shell: false,
        stdio: ["pipe", "pipe", "pipe"],
      });
      this.active.set(input.executionId, child);
      const stdout: Buffer[] = [];
      const stderr: Buffer[] = [];
      let bytes = 0;
      let settled = false;
      let terminating = false;
      let forceKill: NodeJS.Timeout | undefined;
      const abort = (): void => {
        if (terminating) return;
        terminating = true;
        child.kill("SIGTERM");
        forceKill = setTimeout(() => child.kill("SIGKILL"), 5_000);
        forceKill.unref();
      };
      const finish = (action: () => void): void => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        if (forceKill) clearTimeout(forceKill);
        input.signal?.removeEventListener("abort", abort);
        this.active.delete(input.executionId);
        action();
      };
      const capture = (target: Buffer[], chunk: Buffer): void => {
        bytes += chunk.byteLength;
        if (bytes > 32 * 1024 * 1024) {
          child.kill("SIGKILL");
          finish(() => reject(new Error("Runtime command output exceeded 32 MiB")));
          return;
        }
        target.push(chunk);
      };
      child.stdout.on("data", (chunk: Buffer) => capture(stdout, chunk));
      child.stderr.on("data", (chunk: Buffer) => capture(stderr, chunk));
      child.once("error", (error) => finish(() => reject(error)));
      child.once("close", (code) =>
        finish(() =>
          resolve({
            exitCode: code ?? 1,
            stdout: Buffer.concat(stdout).toString("utf8"),
            stderr: Buffer.concat(stderr).toString("utf8"),
          }),
        ),
      );
      child.stdin.on("error", () => undefined);
      input.signal?.addEventListener("abort", abort, { once: true });
      if (input.signal?.aborted) abort();
      child.stdin.end(input.stdin ?? "");
      const timeout = setTimeout(() => {
        child.kill("SIGKILL");
        finish(() => reject(new Error("Runtime command timed out")));
      }, input.timeoutMilliseconds);
      timeout.unref();
    });
  }

  async terminate(executionId: string): Promise<void> {
    const child = this.active.get(executionId);
    if (!child) return;
    child.kill("SIGTERM");
    const forceKill = setTimeout(() => child.kill("SIGKILL"), 5_000);
    forceKill.unref();
  }
}
