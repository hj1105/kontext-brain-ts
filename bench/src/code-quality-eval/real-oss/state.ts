import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { CodeResourceSnapshotAdapter, PythonCodeProvider } from "@kontext-brain/code";
import {
  InMemoryKnowledgeGraphRepository,
  InMemoryResourceContentStore,
  SyncResourceUseCase,
} from "@kontext-brain/core";
import { ExternalIdNormativeResourceReader, SymbolGovernanceResolver } from "@kontext-brain/loader";
import type { CodeQualityRuntime } from "../contracts.js";
import { runWorkspaceCommand } from "../workspace.js";
import type {
  RealOssLogicTarget,
  RealOssNormativeRecord,
  RealOssOntologyStats,
  RealOssTask,
  RealOssWorkspace,
} from "./contracts.js";

const organizationId = "personal:real-oss-code-quality";

export interface RealOssStateAssembly {
  readonly assembly: unknown;
  readonly target: RealOssLogicTarget;
  readonly ontology: RealOssOntologyStats;
}

/**
 * Ingests the real checkout rather than a hand-written code surrogate. Python
 * modules become Resources, behavior-bearing AST-like symbols become chunks and
 * entities, and public GitHub/docs provenance plus extracted normative records
 * are joined to the target through shared Ontology Nodes.
 */
export async function buildRealOssStateAssembly(input: {
  readonly task: RealOssTask;
  readonly workspace: RealOssWorkspace;
  readonly runtime: CodeQualityRuntime;
}): Promise<RealOssStateAssembly> {
  const repository = new InMemoryKnowledgeGraphRepository();
  const sync = new SyncResourceUseCase(repository, new InMemoryResourceContentStore());
  const provider = new PythonCodeProvider();
  const adapter = new CodeResourceSnapshotAdapter();
  const pythonFiles = (await listFiles(path.join(input.workspace.workspacePath, "src")))
    .filter((file) => file.endsWith(".py"))
    .map((file) => path.relative(input.workspace.workspacePath, file).replaceAll(path.sep, "/"))
    .sort();
  let codeSymbols = 0;
  let behaviorBearingSymbols = 0;
  let targetSymbolId: string | undefined;

  for (const relativePath of pythonFiles) {
    const content = await readFile(path.join(input.workspace.workspacePath, relativePath), "utf8");
    const analysis = provider.analyze({
      codebaseId: input.task.codebaseId,
      targetPath: relativePath,
      files: [{ path: relativePath, content }],
    });
    codeSymbols += analysis.symbols.length;
    behaviorBearingSymbols += analysis.symbols.filter((symbol) => symbol.behaviorBearing).length;
    const ontologyNodeIds =
      relativePath === input.task.target.relativePath
        ? input.task.target.ontologyNodeIds
        : [`code-module:${input.task.repository}:${relativePath}`];
    const normalized = adapter.normalize({
      analysis,
      organizationId,
      acl: { organizationWide: true },
      ontologyNodeIds,
    });
    await sync.execute({
      ...normalized,
      // SymbolGovernanceResolver intentionally has one language-independent
      // source identity for code modules.
      source: { ...normalized.source, type: "code-module" },
    });
    if (relativePath === input.task.target.relativePath) {
      const target = analysis.symbols.find(
        (symbol) =>
          symbol.identity.qualifiedName === input.task.target.qualifiedName &&
          symbol.behaviorBearing,
      );
      if (!target) {
        throw new Error(
          `Behavior-bearing target ${input.task.target.qualifiedName} was not found in ${relativePath}`,
        );
      }
      targetSymbolId = target.symbolId;
    }
  }
  if (!targetSymbolId) throw new Error("Real OSS target symbol was not indexed");

  for (const document of input.task.sourceDocuments) {
    await sync.execute({
      organizationId,
      source: {
        connectorId: document.kind === "documentation" ? "github-repository" : "github",
        externalId: document.documentId,
        type: document.kind,
      },
      title: document.title,
      contentHash: `sha256:${sha256(document.body)}`,
      body: document.body,
      acl: { organizationWide: true },
      ontologyNodeIds: document.ontologyNodeIds,
      chunks: [
        {
          id: `${document.documentId}:0`,
          contentHash: `sha256:${sha256(`${document.documentId}\n${document.body}`)}`,
          text: document.body,
          position: 0,
        },
      ],
    });
  }

  const revisionByRecord = new Map<string, string>();
  for (const record of input.task.normativeRecords) {
    revisionByRecord.set(record.recordId, record.revisionId);
    const body = normativeText(record);
    await sync.execute({
      organizationId,
      source: { connectorId: "normative", externalId: record.recordId, type: record.kind },
      title: record.recordId,
      contentHash: `sha256:${sha256(`${record.revisionId}\n${body}`)}`,
      body,
      acl: { organizationWide: true },
      ontologyNodeIds: record.ontologyNodeIds,
      chunks: [
        {
          id: `${record.recordId}:0`,
          contentHash: `sha256:${sha256(body)}`,
          text: body,
          position: 0,
        },
      ],
    });
  }

  const resolver = new SymbolGovernanceResolver(
    repository,
    new ExternalIdNormativeResourceReader((recordId) => revisionByRecord.get(recordId)),
  );
  const resolution = await resolver.resolve({
    organizationId,
    codebaseId: input.task.codebaseId,
    relativePath: input.task.target.relativePath,
    plannedSymbolId: input.task.target.plannedSymbolId,
  });
  const expected = input.task.normativeRecords.map((record) => record.recordId).sort();
  const actual = resolution.records.map((record) => record.recordId).sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(
      `Real OSS ontology resolution mismatch: expected ${expected.join(", ")}; received ${actual.join(", ")}`,
    );
  }

  const scope = { kind: "codebase", codebaseId: input.task.codebaseId } as const;
  const acceptedAt = latestObservedAt(input.task);
  const target = {
    workItemId: input.task.target.workItemId,
    plannedSymbolId: input.task.target.plannedSymbolId,
  };
  return {
    target,
    ontology: {
      codeResources: pythonFiles.length,
      codeSymbols,
      behaviorBearingSymbols,
      provenanceResources: input.task.sourceDocuments.length,
      normativeRecords: input.task.normativeRecords.length,
      targetSymbolId,
      targetQualifiedName: input.task.target.qualifiedName,
      governingRecordIds: actual,
    },
    assembly: {
      taskId: input.task.taskId,
      organizationId,
      codeRevision: input.workspace.baseRevision,
      baseScopes: [scope],
      localManifest: {
        schemaVersion: 1,
        organizationId,
        revisions: input.task.normativeRecords.map((record) =>
          normativeRevision(record, input.task, input.runtime, scope),
        ),
        activations: input.task.normativeRecords.map((record) => ({
          organizationId,
          kind: record.kind,
          recordId: record.recordId,
          revisionId: record.revisionId,
          scope,
          state: "accepted_local",
          acceptedBy: "benchmark-curator:swe-bench-verified",
          acceptedAt,
        })),
      },
      evidence: input.task.sourceDocuments.map((document) => ({
        evidenceId: document.documentId,
        text: document.body,
        sourceSpan: `${document.sourceUrl} — ${document.sourceSpan}`,
        availability: "current",
        allowedRuntimeProviders: [input.runtime],
      })),
      logicPlans: [
        {
          workItemId: input.task.target.workItemId,
          plannedSymbolIds: [input.task.target.plannedSymbolId],
          plannedSymbols: [
            {
              plannedSymbolId: input.task.target.plannedSymbolId,
              taskId: input.task.taskId,
              intendedIdentity: {
                codebaseId: input.task.codebaseId,
                relativePath: input.task.target.relativePath,
                language: "python",
                kind: "method",
                qualifiedName: input.task.target.qualifiedName,
              },
              responsibility: input.task.target.responsibility,
              boundSymbolId: targetSymbolId,
            },
          ],
          allowedPaths: input.task.allowedPaths,
          dependsOn: [],
          requiredVerifiers: [
            { kind: "test", ref: "swe-bench:FAIL_TO_PASS" },
            { kind: "test", ref: "swe-bench:PASS_TO_PASS" },
          ],
          capabilityId: "capability:flask-blueprint-construction",
        },
      ],
      governanceLinks: resolution.records.map((record) => ({
        plannedSymbolId: input.task.target.plannedSymbolId,
        recordId: record.recordId,
        revisionId: record.revisionId,
        origin: record.origin,
      })),
    },
  };
}

export async function publishRealOssState(input: {
  readonly task: RealOssTask;
  readonly workspace: RealOssWorkspace;
  readonly runtime: CodeQualityRuntime;
  readonly repositoryRoot: string;
  readonly pluginDataDirectory: string;
}): Promise<RealOssStateAssembly> {
  const state = await buildRealOssStateAssembly(input);
  await mkdir(input.pluginDataDirectory, { recursive: true, mode: 0o700 });
  const assemblyPath = path.join(input.pluginDataDirectory, "real-oss-assembly.json");
  await writeFile(assemblyPath, `${JSON.stringify(state.assembly, null, 2)}\n`, { mode: 0o600 });
  const result = await runWorkspaceCommand(
    input.repositoryRoot,
    process.execPath,
    [
      path.join(input.repositoryRoot, "packages", "local", "dist", "cli.js"),
      "publish-task-state",
      assemblyPath,
    ],
    { ...process.env, KONTEXT_PLUGIN_DATA: input.pluginDataDirectory },
  );
  if (result.exitCode !== 0) {
    throw new Error(`Cannot publish real OSS Kontext state: ${result.stderr || result.stdout}`);
  }
  return state;
}

function normativeRevision(
  record: RealOssNormativeRecord,
  task: RealOssTask,
  runtime: CodeQualityRuntime,
  scope: { readonly kind: "codebase"; readonly codebaseId: string },
): unknown {
  const sourceById = new Map(
    task.sourceDocuments.map((document) => [document.documentId, document]),
  );
  const common = {
    kind: record.kind,
    organizationId,
    recordId: record.recordId,
    revisionId: record.revisionId,
    scope,
    evidence: record.evidenceIds.map((evidenceId) => ({
      evidenceId,
      sourceSpan: sourceById.get(evidenceId)?.sourceSpan,
    })),
    egress: { dataClassification: "public", allowedRuntimeProviders: [runtime] },
    authoredBy: `github:${task.repository}`,
    authoredAt: latestObservedAt(task, record.evidenceIds),
  };
  switch (record.kind) {
    case "decision":
      return { ...common, statement: record.statement };
    case "domain_term":
      return {
        ...common,
        term: record.term,
        definition: record.definition,
        avoid: record.avoid,
      };
    case "invariant":
      return {
        ...common,
        statement: record.statement,
        verifiers: [
          { kind: "test", ref: "swe-bench:FAIL_TO_PASS" },
          { kind: "test", ref: "swe-bench:PASS_TO_PASS" },
        ],
      };
  }
}

function normativeText(record: RealOssNormativeRecord): string {
  switch (record.kind) {
    case "decision":
    case "invariant":
      return record.statement;
    case "domain_term":
      return `${record.term}: ${record.definition}`;
  }
}

function latestObservedAt(task: RealOssTask, evidenceIds?: readonly string[]): string {
  const allowed = evidenceIds ? new Set(evidenceIds) : undefined;
  return (
    task.sourceDocuments
      .filter((document) => !allowed || allowed.has(document.documentId))
      .map((document) => document.observedAt)
      .sort()
      .at(-1) ?? "1970-01-01T00:00:00.000Z"
  );
}

async function listFiles(root: string): Promise<readonly string[]> {
  const entries = await readdir(root, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map(async (entry) => {
      const fullPath = path.join(root, entry.name);
      return entry.isDirectory() ? listFiles(fullPath) : [fullPath];
    }),
  );
  return nested.flat();
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
