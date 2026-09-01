import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { PlannedSymbolRecord } from "@kontext-brain/code";
import type { ContextReceipt, LogicWorkItem } from "@kontext-brain/spec";
import { afterEach, describe, expect, it } from "vitest";
import {
  BoundWorkspaceChangeEvidenceProvider,
  InMemoryWriteAuthorizationBindingStore,
  captureWorkspaceCodeSymbols,
  captureWorkspaceSnapshot,
} from "../src/index.js";

const temporaryDirectories: string[] = [];
const taskId = "task:sidecar-evidence";
const workItem: LogicWorkItem = {
  workItemId: "work-item:handler",
  taskId,
  plannedSymbolIds: ["planned-symbol:handler"],
  dependsOn: [],
  allowedPaths: ["src/handler.ts"],
  requiredVerifiers: [],
  capabilityId: "capability:handler",
};

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("BoundWorkspaceChangeEvidenceProvider", () => {
  it("derives patch, result revision, receipt, and Planned Symbol binding from sidecar state", async () => {
    const { workspace, bindings } = await boundWorkspace();
    await writeFile(
      path.join(workspace, "src/handler.ts"),
      "export function handler(value: number) { return value + 2; }\n",
    );
    const planned = plannedHandler();

    const evidence = await new BoundWorkspaceChangeEvidenceProvider(bindings).observe({
      workspacePath: workspace,
      taskId,
      workItem,
      plannedSymbols: [planned],
    });

    expect(evidence.currentCodeRevision).toMatch(/^workspace-revision:/);
    expect(evidence.observedPatch).toMatchObject({
      patchDigest: expect.stringMatching(/^sha256:[a-f0-9]{64}$/),
      changedPaths: ["src/handler.ts"],
      changedSymbolIds: [expect.stringMatching(/^code-symbol:/)],
    });
    expect(evidence.receipts).toEqual([expect.objectContaining({ receiptId: "receipt:handler" })]);
    expect(evidence.plannedSymbolBindings).toEqual([
      expect.objectContaining({
        plannedSymbolId: planned.plannedSymbolId,
        symbolId: evidence.observedPatch.changedSymbolIds[0],
        boundBy: "intended_identity",
      }),
    ]);
    expect(evidence.plannedSymbolIssues).toEqual([]);
    expect(evidence.unauthorizedChangedSymbolIds).toEqual([]);
  });

  it("binds the Planned Symbol after a throwing stub is implemented", async () => {
    const { workspace, bindings } = await boundWorkspace(
      "export function handler(_value: number): number {\n  throw new Error('Not implemented');\n}\n",
    );
    // Implementing a throw-only stub changes the inferred signature, so the
    // symbol takes a new ID and the pre-edit and post-edit revisions used to
    // collide as two candidates sharing one intended identity.
    await writeFile(
      path.join(workspace, "src/handler.ts"),
      [
        "export function handler(value: number) {",
        "  if (value < 0) throw new RangeError('negative');",
        "  return Math.min(value * 3, 4500);",
        "}",
        "",
      ].join("\n"),
    );

    const evidence = await new BoundWorkspaceChangeEvidenceProvider(bindings).observe({
      workspacePath: workspace,
      taskId,
      workItem,
      plannedSymbols: [plannedHandler()],
    });

    expect(evidence.plannedSymbolIssues).toEqual([]);
    expect(evidence.plannedSymbolBindings).toEqual([
      expect.objectContaining({
        plannedSymbolId: "planned-symbol:handler",
        boundBy: "intended_identity",
      }),
    ]);
    // The superseded predecessor is still reported as changed, so it must not
    // count as an out-of-scope symbol.
    expect(evidence.unauthorizedChangedSymbolIds).toEqual([]);
  });

  it("reports an unbound plan instead of accepting a caller-named symbol", async () => {
    const { workspace, bindings } = await boundWorkspace();
    await writeFile(
      path.join(workspace, "src/handler.ts"),
      "export function handler(value: number) { return value + 3; }\n",
    );

    const evidence = await new BoundWorkspaceChangeEvidenceProvider(bindings).observe({
      workspacePath: workspace,
      taskId,
      workItem,
      plannedSymbols: [{ ...plannedHandler(), intendedIdentity: { qualifiedName: "missing" } }],
    });

    expect(evidence.plannedSymbolIssues).toEqual([
      expect.objectContaining({
        plannedSymbolId: "planned-symbol:handler",
        code: "identity_not_found",
      }),
    ]);
    expect(evidence.unauthorizedChangedSymbolIds).toEqual(evidence.observedPatch.changedSymbolIds);
  });
});

async function boundWorkspace(
  initialSource = "export function handler(value: number) { return value + 1; }\n",
) {
  const root = await mkdtemp(path.join(tmpdir(), "kontext-sidecar-evidence-"));
  temporaryDirectories.push(root);
  const workspace = path.join(root, "workspace");
  await mkdir(path.join(workspace, "src"), { recursive: true });
  await writeFile(path.join(workspace, "src/handler.ts"), initialSource);
  execFileSync("git", ["init", "-q"], { cwd: workspace });
  execFileSync("git", ["add", "."], { cwd: workspace });
  execFileSync(
    "git",
    [
      "-c",
      "user.name=Kontext Test",
      "-c",
      "user.email=test@example.invalid",
      "commit",
      "-qm",
      "base",
    ],
    { cwd: workspace },
  );
  const baseline = await captureWorkspaceSnapshot(workspace, workItem.allowedPaths);
  const symbolBaseline = await captureWorkspaceCodeSymbols(
    workspace,
    workItem.allowedPaths,
    baseline,
  );
  const bindings = new InMemoryWriteAuthorizationBindingStore();
  await bindings.put(workspace, {
    request: {
      taskId,
      logic: {
        workItemId: workItem.workItemId,
        plannedSymbolIds: workItem.plannedSymbolIds,
      },
      runtimeProvider: "codex",
      issuedAt: "2026-08-31T00:00:00.000Z",
      expiresAt: "2026-08-31T01:00:00.000Z",
      totalTokenBudget: 10_000,
      optionalEvidenceTokenBudget: 1_000,
    },
    allowedPaths: [path.join(workspace, "src/handler.ts")],
    receipt: receipt(),
    initialBaseline: baseline,
    baseline,
    symbolBaseline,
  });
  return { workspace, bindings };
}

function plannedHandler(): PlannedSymbolRecord {
  return {
    plannedSymbolId: "planned-symbol:handler",
    taskId,
    intendedIdentity: {
      relativePath: "src/handler.ts",
      language: "typescript",
      kind: "function",
      qualifiedName: "handler",
    },
    responsibility: "Handle one request",
  };
}

function receipt(): ContextReceipt {
  return {
    receiptId: "receipt:handler",
    taskId,
    workItemId: workItem.workItemId,
    plannedSymbolIds: workItem.plannedSymbolIds,
    allowedPaths: workItem.allowedPaths,
    contextDigest: "context:current",
    normativeRevisions: [],
    evidenceIds: [],
    issuedAt: "2026-08-31T00:00:00.000Z",
    expiresAt: "2026-08-31T01:00:00.000Z",
  };
}
