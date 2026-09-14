import { randomUUID } from "node:crypto";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { type CurrentTaskContextState, TaskContextWorkflow } from "@kontext-brain/context";
import type { DecisionRevision, EffectiveNormativeRecord, TaskContract } from "@kontext-brain/spec";
import { afterEach, describe, expect, it } from "vitest";
import { FileTaskContextRepository } from "../src/index.js";

const temporaryDirectories: string[] = [];
const contract: TaskContract = {
  taskId: "task:durable",
  intent: "Keep a frozen Task Context across MCP process restarts.",
  acceptance: [
    {
      criterionId: "acceptance:persistence",
      statement: "The prepared context can be read by a new repository instance.",
      verifier: { kind: "test", ref: "file-task-context-repository.test.ts" },
    },
  ],
  nonGoals: [],
  targets: ["planned-symbol:durable"],
  risk: "low",
};

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true })),
  );
});

describe("FileTaskContextRepository", () => {
  it("atomically persists private current and prepared Task context across instances", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "kontext-task-context-"));
    temporaryDirectories.push(directory);
    const repository = new FileTaskContextRepository(directory);
    await repository.publishCurrent(contract.taskId, currentState());

    const workflow = new TaskContextWorkflow(repository, repository);
    const prepared = await workflow.prepareTask({
      contract,
      createdAt: "2026-08-28T00:00:00.000Z",
    });
    const reopened = new FileTaskContextRepository(directory);

    expect(await reopened.getCurrent(contract.taskId)).toEqual(currentState());
    expect(await reopened.get(contract.taskId)).toEqual(prepared);
    expect(repository.currentStateFilePath(contract.taskId)).not.toContain(contract.taskId);
    expect((await stat(repository.currentStateFilePath(contract.taskId))).mode & 0o777).toBe(0o600);
    expect((await stat(repository.preparedTaskFilePath(contract.taskId))).mode & 0o777).toBe(0o600);
  });

  it("rejects stale publishers and detects payload tampering", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "kontext-task-context-"));
    temporaryDirectories.push(directory);
    const repository = new FileTaskContextRepository(directory);
    const written = await repository.publishCurrent(contract.taskId, currentState());

    await expect(
      repository.publishCurrent(contract.taskId, currentState(), {
        expectedDigest: `sha256:${randomUUID()}`,
      }),
    ).rejects.toThrow("changed since it was read");
    expect(
      (
        await repository.publishCurrent(contract.taskId, currentState(), {
          expectedDigest: written.digest,
        })
      ).created,
    ).toBe(false);

    const filePath = repository.currentStateFilePath(contract.taskId);
    const envelope = JSON.parse(await readFile(filePath, "utf8"));
    envelope.payload.codeRevision = "commit:tampered";
    await writeFile(filePath, JSON.stringify(envelope), "utf8");
    await expect(repository.getCurrent(contract.taskId)).rejects.toThrow("digest mismatch");
  });
});

function currentState(): CurrentTaskContextState {
  const revision: DecisionRevision = {
    kind: "decision",
    organizationId: "personal:owner",
    recordId: "decision:context",
    revisionId: "revision:context:1",
    scope: { kind: "personal", subjectId: "user:owner" },
    evidence: [{ evidenceId: "evidence:session:1" }],
    egress: {
      dataClassification: "internal",
      allowedRuntimeProviders: ["codex", "claude"],
    },
    authoredBy: "user:owner",
    authoredAt: "2026-08-28T00:00:00.000Z",
    statement: "Obtain current Kontext context before editing.",
  };
  const effective: EffectiveNormativeRecord = {
    origin: "local",
    revision,
    activation: {
      organizationId: revision.organizationId,
      kind: revision.kind,
      recordId: revision.recordId,
      revisionId: revision.revisionId,
      scope: revision.scope,
      state: "accepted_local",
      acceptedBy: "user:owner",
      acceptedAt: "2026-08-28T00:01:00.000Z",
    },
  };
  return {
    codeRevision: "commit:1",
    sourceFreshnessDigest: "freshness:1",
    effectiveScopes: [revision.scope],
    normativeRecords: [effective],
    normativeRevisionCatalog: [revision],
    conflicts: [],
    evidence: [
      {
        evidenceId: "evidence:session:1",
        text: "The user confirmed this decision in the local session.",
        sourceSpan: "session decision",
        availability: "current",
        allowedRuntimeProviders: ["codex", "claude"],
        provenance: {
          resourceId: "resource:session:1",
          chunkId: "chunk:session:1",
          resourceTitle: "Local session",
          source: {
            connectorId: "codex",
            externalId: "codex://session/1",
            type: "session",
          },
          observedAt: "2026-08-28T00:00:00.000Z",
          contentHash: "sha256:session",
          ontologyNodeIds: ["engineering"],
        },
      },
    ],
    logicPlans: [
      {
        workItemId: "work-item:durable",
        plannedSymbolIds: ["planned-symbol:durable"],
        allowedPaths: ["src/durable.ts"],
      },
    ],
    governanceLinks: [
      {
        plannedSymbolId: "planned-symbol:durable",
        recordId: revision.recordId,
        revisionId: revision.revisionId,
        origin: "curated",
      },
    ],
  };
}
