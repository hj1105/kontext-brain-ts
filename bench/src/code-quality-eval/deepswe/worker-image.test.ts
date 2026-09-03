import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { WorkspaceCommandResult } from "../workspace.js";
import {
  CODEX_WORKER_IMAGE_RECIPE_VERSION,
  type WorkerImageCommand,
  ensureCodexWorkerImage,
  planCodexWorkerImage,
} from "./worker-image.js";

const cleanup = new Set<string>();
const spec = {
  baseImage: "registry.example.invalid/deepswe/task@sha256:abc123",
  codexVersion: "0.144.6",
  pierVersion: "0.3.1",
} as const;

afterEach(async () => {
  await Promise.all([...cleanup].map((entry) => rm(entry, { recursive: true, force: true })));
  cleanup.clear();
});

describe("DeepSWE pinned Codex worker image", () => {
  it("derives a stable identity from every build input", () => {
    const first = planCodexWorkerImage(spec);
    expect(planCodexWorkerImage(spec)).toEqual(first);
    expect(first.recipeVersion).toBe(CODEX_WORKER_IMAGE_RECIPE_VERSION);
    expect(first.tag).toMatch(/^kontext-brain\/deepswe-codex:[a-f0-9]{24}$/);
    expect(first.identitySha256).toMatch(/^[a-f0-9]{64}$/);
    expect(planCodexWorkerImage({ ...spec, codexVersion: "0.145.0" }).tag).not.toBe(first.tag);
    expect(planCodexWorkerImage({ ...spec, pierVersion: "0.3.2" }).tag).not.toBe(first.tag);
    expect(planCodexWorkerImage({ ...spec, baseImage: "task:v2" }).tag).not.toBe(first.tag);
  });

  it("reuses an image only when all identity labels match", async () => {
    const plan = planCodexWorkerImage(spec);
    const execute = vi
      .fn<WorkerImageCommand>()
      .mockResolvedValue(
        result(0, JSON.stringify({ Id: "sha256:existing", Config: { Labels: plan.labels } })),
      );
    const manifestsDirectory = await temporaryDirectory();
    const ensured = await ensureCodexWorkerImage({
      spec,
      manifestsDirectory,
      repositoryRoot: manifestsDirectory,
      execute,
    });
    expect(ensured).toMatchObject({ tag: plan.tag, imageId: "sha256:existing", reused: true });
    expect(execute).toHaveBeenCalledTimes(1);
    expect(execute.mock.calls[0]?.[2]).toEqual([
      "image",
      "inspect",
      "--format",
      "{{json .}}",
      plan.tag,
    ]);
  });

  it("builds a private labeled image on a cache miss without provider API keys", async () => {
    const plan = planCodexWorkerImage(spec);
    const execute = vi
      .fn<WorkerImageCommand>()
      .mockResolvedValueOnce(result(1, "", "not found"))
      .mockResolvedValueOnce(result(0, "built"))
      .mockResolvedValueOnce(
        result(0, JSON.stringify({ Id: "sha256:built", Config: { Labels: plan.labels } })),
      );
    const manifestsDirectory = await temporaryDirectory();
    const previousApiKey = process.env.OPENAI_API_KEY;
    process.env.OPENAI_API_KEY = "must-not-reach-docker";
    try {
      const ensured = await ensureCodexWorkerImage({
        spec,
        manifestsDirectory,
        repositoryRoot: manifestsDirectory,
        execute,
      });
      expect(ensured).toMatchObject({ imageId: "sha256:built", reused: false });
      const dockerfilePath = ensured.dockerfilePath;
      expect(dockerfilePath).toBeDefined();
      const dockerfile = await readFile(dockerfilePath as string, "utf8");
      expect(dockerfile).toContain(`FROM ${spec.baseImage}`);
      expect(dockerfile).toContain(`@openai/codex@${spec.codexVersion}`);
      expect(dockerfile).toContain(plan.identitySha256);
      expect(dockerfile).not.toContain("must-not-reach-docker");
      expect((await stat(dockerfilePath as string)).mode & 0o777).toBe(0o600);
      const buildCall = execute.mock.calls[1];
      expect(buildCall?.[2]).toEqual([
        "build",
        "--pull=false",
        "--tag",
        plan.tag,
        path.dirname(dockerfilePath as string),
      ]);
      expect(buildCall?.[3]?.OPENAI_API_KEY).toBeUndefined();
    } finally {
      if (previousApiKey === undefined) process.env.OPENAI_API_KEY = undefined;
      else process.env.OPENAI_API_KEY = previousApiKey;
    }
  });

  it("rebuilds a same-named image whose labels do not match", async () => {
    const plan = planCodexWorkerImage(spec);
    const execute = vi
      .fn<WorkerImageCommand>()
      .mockResolvedValueOnce(
        result(0, JSON.stringify({ Id: "sha256:stale", Config: { Labels: {} } })),
      )
      .mockResolvedValueOnce(result(0, "built"))
      .mockResolvedValueOnce(
        result(0, JSON.stringify({ Id: "sha256:fresh", Config: { Labels: plan.labels } })),
      );
    const manifestsDirectory = await temporaryDirectory();
    const ensured = await ensureCodexWorkerImage({
      spec,
      manifestsDirectory,
      repositoryRoot: manifestsDirectory,
      execute,
    });
    expect(ensured).toMatchObject({ imageId: "sha256:fresh", reused: false });
    expect(execute.mock.calls[1]?.[2]?.[0]).toBe("build");
  });

  it("coalesces concurrent requests for the same image into one build", async () => {
    const plan = planCodexWorkerImage(spec);
    let releaseInspect: (() => void) | undefined;
    const inspectGate = new Promise<void>((resolve) => {
      releaseInspect = resolve;
    });
    const execute = vi
      .fn<WorkerImageCommand>()
      .mockImplementationOnce(async () => {
        await inspectGate;
        return result(1, "", "not found");
      })
      .mockResolvedValueOnce(result(0, "built"))
      .mockResolvedValueOnce(
        result(0, JSON.stringify({ Id: "sha256:built-once", Config: { Labels: plan.labels } })),
      );
    const manifestsDirectory = await temporaryDirectory();
    const input = { spec, manifestsDirectory, repositoryRoot: manifestsDirectory, execute };
    const first = ensureCodexWorkerImage(input);
    const second = ensureCodexWorkerImage(input);
    releaseInspect?.();
    await expect(Promise.all([first, second])).resolves.toEqual([
      expect.objectContaining({ imageId: "sha256:built-once" }),
      expect.objectContaining({ imageId: "sha256:built-once" }),
    ]);
    expect(execute.mock.calls.filter((call) => call[2]?.[0] === "build")).toHaveLength(1);
  });

  it("rejects values that could inject Dockerfile instructions", () => {
    expect(() => planCodexWorkerImage({ ...spec, baseImage: "task:v1\nRUN env" })).toThrow(
      "Invalid base Docker image",
    );
  });
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-worker-image-"));
  cleanup.add(directory);
  return directory;
}

function result(exitCode: number, stdout = "", stderr = ""): WorkspaceCommandResult {
  return { exitCode, stdout, stderr };
}
