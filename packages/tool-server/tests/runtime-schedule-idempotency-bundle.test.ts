import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { expect, it, vi } from "vitest";
import { FileRuntimeScheduleJobStore, RuntimeScheduleJobManager } from "../src/index.js";

it("recovers the same durable enqueue through two real MCP sidecar processes without current context or provider credentials", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-enqueue-mcp-"));
  const request = {
    requestId: randomUUID(),
    taskId: "task:mcp-retry",
    repositoryPath: directory,
    work: [
      { workItemId: "logic:one", prompt: "Fixture only", eligibleProviders: ["codex" as const] },
    ],
  };
  const manager = new RuntimeScheduleJobManager(new FileRuntimeScheduleJobStore(directory));
  const execute = vi.fn(async () => ({ capabilities: [], results: [] }));
  try {
    const accepted = await manager.enqueue(request, async () => ({
      codeRevision: "fixture:revision",
      contextDigest: "fixture:digest",
      execute,
    }));
    await expect.poll(async () => (await manager.get(accepted.jobId)).status).toBe("completed");
    for (let restart = 0; restart < 2; restart++) {
      const client = new Client({ name: "kontext-idempotency-test", version: "1.0.0" });
      try {
        await client.connect(
          new StdioClientTransport({
            command: process.execPath,
            args: [path.resolve("plugins/kontext-brain/server.mjs")],
            cwd: directory,
            env: {
              KONTEXT_PLUGIN_DATA: directory,
              HOME: directory,
              USERPROFILE: directory,
              APPDATA: directory,
              LOCALAPPDATA: directory,
              XDG_CONFIG_HOME: directory,
              XDG_DATA_HOME: directory,
              XDG_CACHE_HOME: directory,
              CODEX_HOME: directory,
              CLAUDE_CONFIG_DIR: directory,
              PATH: "",
            },
            stderr: "pipe",
          }),
        );
        const recovered = await client.callTool({
          name: "kontext_schedule_logic",
          arguments: request,
        });
        expect(recovered.isError).not.toBe(true);
        expect(recovered.structuredContent).toMatchObject({
          jobId: accepted.jobId,
          requestId: request.requestId,
          status: "completed",
          contextDigest: "fixture:digest",
        });
        const conflict = await client.callTool({
          name: "kontext_schedule_logic",
          arguments: { ...request, taskId: "task:changed" },
        });
        expect(conflict.isError).toBe(true);
      } finally {
        await client.close();
      }
    }
    expect(execute).toHaveBeenCalledTimes(1);
  } finally {
    await rm(directory, { recursive: true });
  }
});
