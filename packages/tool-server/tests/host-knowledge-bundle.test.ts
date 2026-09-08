import { access, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { expect, it } from "vitest";
import { z } from "zod";

const managementTools = [
  "kontext_list_tasks",
  "kontext_inspect_registered_schedule",
  "kontext_list_registered_schedules",
  "kontext_inspect_registered_integration",
  "kontext_integrate_registered_schedule",
  "kontext_resume_registered_schedule",
  "kontext_cancel_registered_schedule",
  "kontext_list_sources",
  "kontext_finalize_task",
  "kontext_inspect_finalization",
  "kontext_revalidate_finalization",
  "kontext_assess_completion",
  "kontext_create_task",
  "kontext_start_plan",
  "kontext_refine_plan",
  "kontext_inspect_plan",
  "kontext_cancel_plan",
  "kontext_approve_plan",
  "kontext_register_markdown_source",
  "kontext_register_session_source",
  "kontext_inspect_source",
  "kontext_refresh_source",
  "kontext_set_source_sharing",
];
it("registers source metadata through the real host sidecar but never exposes the tool to ordinary workers", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-host-source-mcp-"));
  const data = path.join(directory, "data");
  const request = { workspacePath: directory, relativePath: "notes.md" };
  await writeFile(
    path.join(directory, "notes.md"),
    "## Actual local source\nDo not invent an approval.\n",
  );
  let identity: string | undefined;
  const executable = process.env.KONTEXT_TEST_NODE_EXECUTABLE ?? process.execPath;
  if (!path.isAbsolute(executable)) throw new Error("Fixture runtime path must be absolute");
  try {
    for (const token of ["a".repeat(64), "b".repeat(64), undefined]) {
      const client = new Client({ name: "kontext-host-source-test", version: "1" });
      try {
        await client.connect(
          new StdioClientTransport({
            command: executable,
            args: [path.resolve("plugins/kontext-brain/server.mjs")],
            cwd: directory,
            env: {
              KONTEXT_PLUGIN_DATA: data,
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
              ELECTRON_RUN_AS_NODE: "1",
              ORCA_BACKGROUND_LAUNCH: "1",
              ...(token ? { KONTEXT_HOST_MANAGEMENT_TOKEN: token } : {}),
            },
            stderr: "pipe",
          }),
        );
        const names = (await client.listTools()).tools.map((tool) => tool.name);
        if (!token) {
          for (const name of managementTools) expect(names).not.toContain(name);
          const denied = await client.callTool({
            name: "kontext_register_markdown_source",
            arguments: { ...request, hostToken: "a".repeat(64) },
          });
          expect(denied.isError).toBe(true);
          continue;
        }
        for (const name of managementTools) expect(names).toContain(name);
        const refusedRefinement = await client.callTool({
          name: "kontext_refine_plan",
          arguments: {
            requestId: "10000000-0000-4000-8000-000000000001",
            parentRequestId: "10000000-0000-4000-8000-000000000002",
            expectedParentDigest: `sha256:${"0".repeat(64)}`,
            feedback: "Review zero inputs",
            hostToken: "c".repeat(64),
          },
        });
        expect(refusedRefinement.isError).toBe(true);
        const rejected = await client.callTool({
          name: "kontext_register_markdown_source",
          arguments: { ...request, hostToken: "c".repeat(64) },
        });
        expect(rejected.isError).toBe(true);
        if (!identity)
          await expect(
            access(path.join(data, "knowledge", "local-principal.json")),
          ).rejects.toThrow();
        const response = await client.callTool({
          name: "kontext_register_markdown_source",
          arguments: { ...request, hostToken: token },
        });
        expect(response.isError).not.toBe(true);
        expect(response.structuredContent).toMatchObject({
          title: "notes.md",
          changed: !identity,
          providerSharing: "not_granted",
          normativeApproval: "not_granted",
          evidence: [
            {
              resourceId: expect.any(String),
              evidenceId: expect.any(String),
              chunkId: expect.any(String),
            },
          ],
        });
        expect(JSON.stringify(response)).not.toContain("Do not invent an approval");
        expect(JSON.stringify(response)).not.toContain(token);
        const metadata = z
          .object({
            organizationId: z.string().uuid(),
            resourceId: z.string(),
            contentHash: z.string(),
          })
          .parse(response.structuredContent);
        const inspected = await client.callTool({
          name: "kontext_inspect_source",
          arguments: { hostToken: token, resourceId: metadata.resourceId },
        });
        expect(inspected.isError).not.toBe(true);
        const registration = z
          .object({ revision: z.number(), sharing: z.unknown() })
          .parse(inspected.structuredContent);
        expect(registration.sharing).toEqual(
          identity
            ? {
                dataClassification: "internal",
                allowedRuntimeProviders: ["codex"],
              }
            : null,
        );
        const inventory = await client.callTool({
          name: "kontext_list_sources",
          arguments: { hostToken: token, limit: 50 },
        });
        expect(inventory.isError).not.toBe(true);
        expect(inventory.structuredContent).toMatchObject({
          observation: "saved_metadata_only",
          nextCursor: null,
          sources: [{ resourceId: metadata.resourceId, sharing: registration.sharing }],
        });
        expect(JSON.stringify(inventory)).not.toContain("Do not invent an approval");
        expect(
          (
            await client.callTool({
              name: "kontext_list_sources",
              arguments: { hostToken: "c".repeat(64) },
            })
          ).isError,
        ).toBe(true);
        const sharingRequest = {
          resourceId: metadata.resourceId,
          expectedRevision: registration.revision,
          expectedContentHash: metadata.contentHash,
          dataClassification: "internal",
          allowedRuntimeProviders: identity ? [] : ["codex"],
        };
        const forbidden = await client.callTool({
          name: "kontext_set_source_sharing",
          arguments: { ...sharingRequest, hostToken: "c".repeat(64) },
        });
        expect(forbidden.isError).toBe(true);
        const sharing = await client.callTool({
          name: "kontext_set_source_sharing",
          arguments: { ...sharingRequest, hostToken: token },
        });
        expect(sharing.isError).not.toBe(true);
        expect(sharing.structuredContent).toMatchObject({
          revision: registration.revision + 1,
          normativeApproval: "not_granted",
          sharing: { allowedRuntimeProviders: sharingRequest.allowedRuntimeProviders },
        });
        expect(JSON.stringify(sharing)).not.toContain("Do not invent an approval");
        expect(JSON.stringify(sharing)).not.toContain(token);
        const duplicate = await client.callTool({
          name: "kontext_set_source_sharing",
          arguments: { ...sharingRequest, hostToken: token },
        });
        expect(duplicate.isError).toBe(true);
        const refreshed = await client.callTool({
          name: "kontext_refresh_source",
          arguments: { resourceId: metadata.resourceId, hostToken: token },
        });
        expect(refreshed.isError).not.toBe(true);
        expect(refreshed.structuredContent).toMatchObject({
          resourceId: metadata.resourceId,
          contentHash: metadata.contentHash,
          changed: false,
        });
        if (identity) expect(metadata.organizationId).toBe(identity);
        identity = metadata.organizationId;
      } finally {
        await client.close();
      }
    }
  } finally {
    await rm(directory, { recursive: true });
  }
});
