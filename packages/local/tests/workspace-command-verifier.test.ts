import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { VerifierInfrastructureError } from "@kontext-brain/orchestrator";
import { afterEach, describe, expect, it } from "vitest";
import { WorkspaceCommandVerifierAdapter } from "../src/index.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("WorkspaceCommandVerifierAdapter", () => {
  it("executes the exact workspace-owned verifier without a shell", async () => {
    const workspacePath = await workspace([
      {
        kind: "test",
        ref: "feature:test",
        command: process.execPath,
        args: ["-e", "process.stdout.write('verified')"],
      },
    ]);
    const result = await new WorkspaceCommandVerifierAdapter().execute({
      workspacePath,
      requirement: {
        tier: "targeted",
        verifier: { kind: "test", ref: "feature:test" },
        subjectIds: ["symbol:feature"],
      },
      codeRevision: "commit:result",
      contextDigest: "context:current",
      observedAt: "2026-08-29T01:00:00.000Z",
    });

    expect(result).toEqual(
      expect.objectContaining({
        result: "passed",
        output: expect.objectContaining({ exitCode: 0, stdout: "verified" }),
      }),
    );
  });

  it("settles a non-zero command as failed and treats missing definitions as infrastructure", async () => {
    const workspacePath = await workspace([
      {
        kind: "test",
        ref: "feature:test",
        command: process.execPath,
        args: ["-e", "process.stderr.write('failed'); process.exit(3)"],
      },
    ]);
    const adapter = new WorkspaceCommandVerifierAdapter();
    const failed = await adapter.execute({
      workspacePath,
      requirement: {
        tier: "targeted",
        verifier: { kind: "test", ref: "feature:test" },
        subjectIds: ["symbol:feature"],
      },
      codeRevision: "commit:result",
      contextDigest: "context:current",
      observedAt: "2026-08-29T01:00:00.000Z",
    });

    expect(failed).toEqual(
      expect.objectContaining({
        result: "failed",
        output: expect.objectContaining({ exitCode: 3, stderr: "failed" }),
      }),
    );
    await expect(
      adapter.execute({
        workspacePath,
        requirement: {
          tier: "targeted",
          verifier: { kind: "query", ref: "missing:query" },
          subjectIds: ["symbol:feature"],
        },
        codeRevision: "commit:result",
        contextDigest: "context:current",
        observedAt: "2026-08-29T01:00:00.000Z",
      }),
    ).rejects.toBeInstanceOf(VerifierInfrastructureError);
  });
});

async function workspace(verifiers: readonly unknown[]): Promise<string> {
  const workspacePath = await mkdtemp(path.join(tmpdir(), "kontext-verifier-"));
  temporaryDirectories.push(workspacePath);
  await mkdir(path.join(workspacePath, ".kontext"));
  await writeFile(
    path.join(workspacePath, ".kontext", "verifiers.json"),
    JSON.stringify({ schemaVersion: 1, verifiers }),
    "utf8",
  );
  return workspacePath;
}
