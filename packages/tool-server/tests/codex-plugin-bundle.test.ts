import { spawn } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { FileQuarantineStore, FileTaskContextRepository } from "@kontext-brain/local";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { afterEach, describe, expect, it } from "vitest";
import {
  FileWriteAuthorizationBindingStore,
  captureWorkspaceSnapshot,
  changedPathsBetween,
} from "../src/index.js";

const temporaryDirectories: string[] = [];
const pluginRoot = path.resolve("plugins/kontext-brain");
const serverPath = path.join(pluginRoot, "server.mjs");

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("Codex plugin bundle", () => {
  it("initializes over stdio and exposes the complete Task workflow", async () => {
    const dataDirectory = await temporaryDataDirectory();
    const transport = new StdioClientTransport({
      command: process.execPath,
      args: [serverPath],
      cwd: pluginRoot,
      env: environment(dataDirectory),
      stderr: "pipe",
    });
    const client = new Client({ name: "kontext-plugin-test", version: "0.1.0" });
    try {
      await client.connect(transport);
      const tools = await client.listTools();
      expect(tools.tools.map((tool) => tool.name).sort()).toEqual([
        "kontext_authorize_write",
        "kontext_begin_logic",
        "kontext_cancel_schedule",
        "kontext_check_change",
        "kontext_get_schedule",
        "kontext_inspect_runtimes",
        "kontext_prepare_task",
        "kontext_propose_transition",
        "kontext_refresh_task_context",
        "kontext_schedule_logic",
        "kontext_submit_change_bundle",
      ]);
    } finally {
      await client.close();
    }
  });

  it("fails closed when the command hook has no current Context Receipt", async () => {
    const dataDirectory = await temporaryDataDirectory();
    const output = await runHook(dataDirectory, {
      cwd: path.join(dataDirectory, "workspace"),
      hook_event_name: "PreToolUse",
      tool_name: "apply_patch",
      tool_input: {
        command: "*** Begin Patch\n*** Add File: src/unreceipted.ts\n*** End Patch",
      },
    });

    expect(output.hookSpecificOutput.permissionDecision).toBe("deny");
    expect(output.hookSpecificOutput.permissionDecisionReason).toContain("No current Kontext");
  });

  it("carries a current receipt from MCP into an independent exact-path command hook", async () => {
    const dataDirectory = await temporaryDataDirectory();
    const workspace = path.join(dataDirectory, "workspace");
    await mkdir(path.join(workspace, "src"), { recursive: true });
    await writeFile(
      path.join(workspace, "src", "handler.ts"),
      "export function handler() { return 1; }\n",
    );
    const repository = new FileTaskContextRepository(dataDirectory);
    await repository.publishCurrent("task:vertical", {
      codeRevision: "commit:vertical",
      sourceFreshnessDigest: "sha256:fresh",
      effectiveScopes: [{ kind: "personal", subjectId: "user:owner" }],
      normativeRecords: [],
      normativeRevisionCatalog: [],
      conflicts: [],
      evidence: [],
      logicPlans: [
        {
          workItemId: "work-item:handler",
          plannedSymbolIds: ["planned-symbol:handler"],
          allowedPaths: ["src/handler.ts"],
        },
      ],
    });

    const { client } = await connectedClient(dataDirectory);
    try {
      await client.callTool({
        name: "kontext_prepare_task",
        arguments: {
          contract: {
            taskId: "task:vertical",
            intent: "Prove the plugin vertical slice.",
            acceptance: [
              {
                criterionId: "acceptance:hook",
                statement: "The exact path is authorized.",
                verifier: { kind: "test", ref: "codex-plugin-bundle.test.ts" },
              },
            ],
            nonGoals: [],
            targets: ["planned-symbol:handler"],
            risk: "low",
          },
          createdAt: new Date().toISOString(),
        },
      });
      const begun = await client.callTool({
        name: "kontext_begin_logic",
        arguments: {
          taskId: "task:vertical",
          workspacePath: workspace,
          logic: {
            workItemId: "work-item:handler",
            plannedSymbolIds: ["planned-symbol:handler"],
          },
          runtimeProvider: "codex",
          receiptTtlSeconds: 600,
          totalTokenBudget: 10_000,
          optionalEvidenceTokenBudget: 1_000,
        },
      });
      expect(begun.structuredContent).toMatchObject({
        status: "current",
        editingAllowed: true,
        receipt: { allowedPaths: ["src/handler.ts"] },
      });
    } finally {
      await client.close();
    }

    const allowedCommand = "*** Begin Patch\n*** Update File: src/handler.ts\n*** End Patch";
    const allowed = await runHook(dataDirectory, {
      cwd: workspace,
      hook_event_name: "PreToolUse",
      tool_name: "apply_patch",
      tool_use_id: "tool:plugin-allowed",
      tool_input: {
        command: allowedCommand,
      },
    });
    const denied = await runHook(dataDirectory, {
      cwd: workspace,
      hook_event_name: "PreToolUse",
      tool_name: "apply_patch",
      tool_input: {
        command: "*** Begin Patch\n*** Update File: src/outside.ts\n*** End Patch",
      },
    });
    expect(allowed.hookSpecificOutput.permissionDecision).toBe("allow");
    expect(denied.hookSpecificOutput.permissionDecision).toBe("deny");

    await writeFile(
      path.join(workspace, "src", "handler.ts"),
      "export function handler() { return 2; }\n",
    );
    const storedBinding = await new FileWriteAuthorizationBindingStore(dataDirectory).get(
      workspace,
    );
    if (!storedBinding) throw new Error("expected persisted write binding");
    const directAfter = await captureWorkspaceSnapshot(workspace, storedBinding.allowedPaths);
    expect(changedPathsBetween(storedBinding.baseline, directAfter)).toEqual(["src/handler.ts"]);
    const observed = await runHook(
      dataDirectory,
      {
        cwd: workspace,
        hook_event_name: "PostToolUse",
        tool_name: "apply_patch",
        tool_use_id: "tool:plugin-allowed",
        tool_input: { command: allowedCommand },
        tool_response: { exit_code: 0 },
      },
      "--observe-write-hook",
    );
    expect(observed.decision).toBeUndefined();
    expect(observed.hookSpecificOutput.additionalContext).toContain("1 changed path");
    expect(await new FileQuarantineStore(dataDirectory).list("active")).toEqual([]);

    const claudeAllowed = await runHook(dataDirectory, {
      cwd: workspace,
      hook_event_name: "PreToolUse",
      tool_name: "Write",
      tool_use_id: "tool:claude-write",
      tool_input: {
        file_path: path.join(workspace, "src", "handler.ts"),
        content: "export function handler() { return 3; }\n",
      },
    });
    expect(claudeAllowed.hookSpecificOutput.permissionDecision).toBe("allow");
    await writeFile(
      path.join(workspace, "src", "handler.ts"),
      "export function handler() { return 3; }\n",
    );
    const claudeObserved = await runHook(
      dataDirectory,
      {
        cwd: workspace,
        hook_event_name: "PostToolUse",
        tool_name: "Write",
        tool_use_id: "tool:claude-write",
        tool_input: {
          file_path: path.join(workspace, "src", "handler.ts"),
          content: "export function handler() { return 3; }\n",
        },
        tool_response: { success: true },
      },
      "--observe-write-hook",
    );
    expect(claudeObserved.decision).toBeUndefined();
    expect(claudeObserved.hookSpecificOutput.additionalContext).toContain("1 changed path");
    expect(await new FileQuarantineStore(dataDirectory).list("active")).toEqual([]);
  });
});

async function temporaryDataDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-plugin-data-"));
  temporaryDirectories.push(directory);
  return directory;
}

function environment(dataDirectory: string): Record<string, string> {
  return {
    ...Object.fromEntries(
      Object.entries(process.env).filter(
        (entry): entry is [string, string] => entry[1] !== undefined,
      ),
    ),
    KONTEXT_PLUGIN_DATA: dataDirectory,
  };
}

async function connectedClient(dataDirectory: string): Promise<{ client: Client }> {
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [serverPath],
    cwd: pluginRoot,
    env: environment(dataDirectory),
    stderr: "pipe",
  });
  const client = new Client({ name: "kontext-plugin-test", version: "0.1.0" });
  await client.connect(transport);
  return { client };
}

async function runHook(
  dataDirectory: string,
  event: Record<string, unknown>,
  mode = "--authorize-write-hook",
): Promise<{
  readonly decision?: string;
  readonly reason?: string;
  readonly hookSpecificOutput: {
    readonly permissionDecision?: string;
    readonly permissionDecisionReason?: string;
    readonly additionalContext?: string;
  };
}> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [serverPath, mode], {
      cwd: pluginRoot,
      env: environment(dataDirectory),
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.once("error", reject);
    child.once("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Hook exited ${code}: ${stderr}`));
        return;
      }
      resolve(JSON.parse(stdout));
    });
    child.stdin.end(JSON.stringify(event));
  });
}
