import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  InMemoryKnowledgeGraphRepository,
  InMemoryResourceContentStore,
  SyncResourceUseCase,
} from "@kontext-brain/core";
import { ExternalIdNormativeResourceReader, SymbolGovernanceResolver } from "@kontext-brain/loader";
import type { CodeQualityRuntime } from "../contracts.js";
import { runWorkspaceCommand } from "../workspace.js";
import { governedPolicy } from "./generator.js";
import { type LargeScaleRule, allRules } from "./rules.js";
import type { LargeScaleWorkspace } from "./workspace.js";

export const largeScaleTaskId = "task:code-quality:large-scale-retry";
export const largeScaleCodebaseId = "codebase:large-scale-retry";
const organizationId = "personal:code-quality-large-scale";
const scope = { kind: "codebase", codebaseId: largeScaleCodebaseId } as const;

export interface LargeScaleLogicTarget {
  readonly workItemId: string;
  readonly plannedSymbolId: string;
}

export interface LargeScaleStateAssembly {
  readonly assembly: unknown;
  readonly targets: readonly LargeScaleLogicTarget[];
  readonly governingRecordIds: readonly string[];
}

/**
 * Builds the private sidecar state through the same ontology hop production
 * uses. The benchmark never hands a preselected record list to the compiler:
 * code and normative Resources are placed on Ontology Nodes, then
 * SymbolGovernanceResolver derives the links.
 */
export async function buildLargeScaleStateAssembly(input: {
  readonly workspace: LargeScaleWorkspace;
  readonly runtime: CodeQualityRuntime;
}): Promise<LargeScaleStateAssembly> {
  const rules = allRules();
  const repository = new InMemoryKnowledgeGraphRepository();
  const sync = new SyncResourceUseCase(repository, new InMemoryResourceContentStore());
  const revisionByRecord = new Map(rules.map((rule) => [rule.recordId, rule.revisionId]));

  for (const item of input.workspace.repository.functions) {
    await sync.execute({
      organizationId,
      source: {
        connectorId: "code",
        externalId: `${largeScaleCodebaseId}:${item.file}`,
        type: "code-module",
      },
      title: item.name,
      contentHash: `sha256:code:${item.file}`,
      body: item.name,
      acl: { organizationWide: true },
      ontologyNodeIds: [item.subsystem],
      chunks: [
        {
          id: `code:${item.file}:0`,
          contentHash: `sha256:code:${item.file}:0`,
          text: item.name,
          position: 0,
        },
      ],
    });
  }
  for (const rule of rules) {
    await sync.execute({
      organizationId,
      source: { connectorId: "sidecar", externalId: rule.recordId, type: "normative" },
      title: rule.recordId,
      contentHash: `sha256:normative:${rule.revisionId}`,
      body: `${rule.text}\n${rule.evidenceText}`,
      acl: { organizationWide: true },
      ontologyNodeIds: [rule.subsystem ?? `noise:${rule.recordId}`],
      chunks: [
        {
          id: `${rule.recordId}:0`,
          contentHash: `sha256:normative:${rule.revisionId}:0`,
          text: rule.text,
          position: 0,
        },
      ],
    });
  }

  const resolver = new SymbolGovernanceResolver(
    repository,
    new ExternalIdNormativeResourceReader((recordId) => revisionByRecord.get(recordId)),
  );
  const governed = input.workspace.repository.functions
    .filter((item) => item.governed)
    .sort((left, right) => left.name.localeCompare(right.name));
  const targets = governed.map((_, index) => ({
    workItemId: `work-item:target-${String(index + 1).padStart(2, "0")}`,
    plannedSymbolId: `planned-symbol:target-${String(index + 1).padStart(2, "0")}`,
  }));
  const resolutions = await Promise.all(
    governed.map((item, index) => {
      const target = targets[index];
      if (!target) throw new Error(`Missing target for ${item.name}`);
      return resolver.resolve({
        organizationId,
        codebaseId: largeScaleCodebaseId,
        relativePath: item.file,
        plannedSymbolId: target.plannedSymbolId,
      });
    }),
  );
  const governingRecordIds = [
    ...new Set(
      resolutions.flatMap((resolution) => resolution.records.map((record) => record.recordId)),
    ),
  ].sort();
  const expectedGoverningIds = rules
    .filter((rule) => rule.subsystem === "billing")
    .map((rule) => rule.recordId)
    .sort();
  if (JSON.stringify(governingRecordIds) !== JSON.stringify(expectedGoverningIds)) {
    throw new Error(
      `Ontology resolution mismatch: expected ${expectedGoverningIds.join(", ")}; received ${governingRecordIds.join(", ")}`,
    );
  }

  const authoredAt = "2026-08-31T00:00:00.000Z";
  const acceptedAt = "2026-08-31T00:01:00.000Z";
  return {
    targets,
    governingRecordIds,
    assembly: {
      taskId: largeScaleTaskId,
      organizationId,
      codeRevision: input.workspace.baseRevision,
      baseScopes: [scope],
      localManifest: {
        schemaVersion: 1,
        organizationId,
        revisions: rules.map((rule) => normativeRevision(rule, authoredAt, input.runtime)),
        activations: rules.map((rule) => ({
          organizationId,
          kind: rule.kind,
          recordId: rule.recordId,
          revisionId: rule.revisionId,
          scope,
          state: "accepted_local",
          acceptedBy: "user:code-quality-eval",
          acceptedAt,
        })),
      },
      evidence: rules.map((rule) => ({
        evidenceId: rule.evidenceId,
        text: rule.evidenceText,
        sourceSpan: `large-scale fixture:${rule.recordId}`,
        availability: "current",
        allowedRuntimeProviders: [input.runtime],
      })),
      logicPlans: governed.map((item, index) => {
        const target = targets[index];
        if (!target) throw new Error(`Missing target for ${item.name}`);
        return {
          workItemId: target.workItemId,
          plannedSymbolIds: [target.plannedSymbolId],
          plannedSymbols: [
            {
              plannedSymbolId: target.plannedSymbolId,
              taskId: largeScaleTaskId,
              intendedIdentity: {
                relativePath: item.file,
                language: "javascript",
                kind: "function",
                qualifiedName: item.name,
              },
              responsibility: "Apply the current approved retry policy to one logic unit.",
            },
          ],
          allowedPaths: [item.file, governedPolicy.sharedModule],
          dependsOn: [],
          requiredVerifiers: [{ kind: "test", ref: "workspace:test" }],
          capabilityId: "capability:large-scale-retry-policy",
        };
      }),
      governanceLinks: resolutions.flatMap((resolution) =>
        resolution.records.map((record) => ({
          plannedSymbolId: resolution.plannedSymbolId,
          recordId: record.recordId,
          revisionId: record.revisionId,
          origin: record.origin,
        })),
      ),
    },
  };
}

export async function publishLargeScaleState(input: {
  readonly workspace: LargeScaleWorkspace;
  readonly runtime: CodeQualityRuntime;
  readonly repositoryRoot: string;
  readonly pluginDataDirectory: string;
}): Promise<LargeScaleStateAssembly> {
  const state = await buildLargeScaleStateAssembly(input);
  await mkdir(input.pluginDataDirectory, { recursive: true, mode: 0o700 });
  const assemblyPath = path.join(input.pluginDataDirectory, "large-scale-assembly.json");
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
    throw new Error(`Cannot publish large-scale Kontext state: ${result.stderr || result.stdout}`);
  }
  return state;
}

function normativeRevision(
  rule: LargeScaleRule,
  authoredAt: string,
  runtime: CodeQualityRuntime,
): unknown {
  const common = {
    kind: rule.kind,
    organizationId,
    recordId: rule.recordId,
    revisionId: rule.revisionId,
    scope,
    evidence: [{ evidenceId: rule.evidenceId, sourceSpan: `fixture:${rule.recordId}` }],
    egress: { dataClassification: "internal", allowedRuntimeProviders: [runtime] },
    authoredBy: "user:code-quality-eval",
    authoredAt,
  };
  switch (rule.kind) {
    case "decision":
      return { ...common, statement: rule.text };
    case "domain_term":
      return {
        ...common,
        term: "Recovery Ceiling",
        definition: rule.text,
        avoid: ["max delay", "timeout", "backoff limit"],
      };
    case "invariant":
      return {
        ...common,
        statement: rule.text,
        verifiers: [{ kind: "test", ref: "workspace:test" }],
      };
  }
}
