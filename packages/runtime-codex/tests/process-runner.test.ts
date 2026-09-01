import { describe, expect, it } from "vitest";
import { SpawnRuntimeCommandRunner } from "../src/index.js";

describe("SpawnRuntimeCommandRunner cancellation", () => {
  it("terminates an active child process when its signal is aborted", async () => {
    const controller = new AbortController();
    const runner = new SpawnRuntimeCommandRunner();
    const execution = runner.run({
      executionId: "codex-runner:cancel",
      command: process.execPath,
      args: ["-e", "setInterval(() => undefined, 1000)"],
      cwd: process.cwd(),
      environment: stringEnvironment(process.env),
      timeoutMilliseconds: 10_000,
      signal: controller.signal,
    });

    setTimeout(() => controller.abort(), 50);

    await expect(execution).resolves.toEqual(expect.objectContaining({ exitCode: 1 }));
  });
});

function stringEnvironment(environment: NodeJS.ProcessEnv): Record<string, string> {
  return Object.fromEntries(
    Object.entries(environment).filter(
      (entry): entry is [string, string] => entry[1] !== undefined,
    ),
  );
}
