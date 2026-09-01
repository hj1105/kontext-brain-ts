import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import type {
  CodeQualityNormativeRule,
  CodeQualityRuntime,
  CodeQualityScenario,
} from "./contracts.js";
import { runWorkspaceCommand } from "./workspace.js";

const organizationId = "personal:code-quality-eval";
const scope = { kind: "workspace", workspaceId: "workspace:code-quality-eval" } as const;

export async function publishScenarioState(input: {
  readonly scenario: CodeQualityScenario;
  readonly baseRevision: string;
  readonly repositoryRoot: string;
  readonly pluginDataDirectory: string;
  readonly runtime?: CodeQualityRuntime;
}): Promise<void> {
  const runtime = input.runtime ?? "codex";
  await mkdir(input.pluginDataDirectory, { recursive: true, mode: 0o700 });
  const assemblyPath = path.join(input.pluginDataDirectory, "assembly.json");
  await writeFile(
    assemblyPath,
    `${JSON.stringify(assembly(input.scenario, input.baseRevision, runtime), null, 2)}\n`,
    { mode: 0o600 },
  );
  const cliPath = path.join(input.repositoryRoot, "packages", "local", "dist", "cli.js");
  const result = await runWorkspaceCommand(
    input.repositoryRoot,
    process.execPath,
    [cliPath, "publish-task-state", assemblyPath],
    { ...process.env, KONTEXT_PLUGIN_DATA: input.pluginDataDirectory },
  );
  if (result.exitCode !== 0) {
    throw new Error(`Cannot publish Kontext state: ${result.stderr || result.stdout}`);
  }
}

function assembly(
  scenario: CodeQualityScenario,
  baseRevision: string,
  runtime: CodeQualityRuntime,
): unknown {
  const authoredAt = "2026-08-31T00:00:00.000Z";
  const acceptedAt = "2026-08-31T00:01:00.000Z";
  return {
    taskId: scenario.taskId,
    organizationId,
    codeRevision: baseRevision,
    baseScopes: [scope],
    localManifest: {
      schemaVersion: 1,
      organizationId,
      revisions: scenario.rules.map((rule) => revision(rule, authoredAt, runtime)),
      activations: scenario.rules.map((rule) => ({
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
    evidence: scenario.rules.map((rule) => ({
      evidenceId: rule.evidenceId,
      text: rule.evidenceText,
      sourceSpan: `held-out policy fixture:${scenario.scenarioId}`,
      availability: "current",
      allowedRuntimeProviders: [runtime],
    })),
    logicPlans: [
      {
        workItemId: scenario.workItemId,
        plannedSymbolIds: [scenario.plannedSymbolId],
        plannedSymbols: [
          {
            plannedSymbolId: scenario.plannedSymbolId,
            taskId: scenario.taskId,
            intendedIdentity: {
              relativePath: scenario.sourceFile,
              language: languageForPath(scenario.sourceFile),
              kind: "function",
              qualifiedName: scenario.qualifiedName,
            },
            responsibility: scenario.intent,
          },
        ],
        allowedPaths: [scenario.sourceFile],
        dependsOn: [],
        requiredVerifiers: [{ kind: "test", ref: "workspace:test" }],
        capabilityId: scenario.capabilityId,
      },
    ],
  };
}

function languageForPath(filePath: string): "typescript" | "javascript" | "python" {
  if (/\.py$/i.test(filePath)) return "python";
  return /\.[cm]?jsx?$/i.test(filePath) ? "javascript" : "typescript";
}

function revision(
  rule: CodeQualityNormativeRule,
  authoredAt: string,
  runtime: CodeQualityRuntime,
): unknown {
  const common = {
    kind: rule.kind,
    organizationId,
    recordId: rule.recordId,
    revisionId: rule.revisionId,
    scope,
    evidence: [{ evidenceId: rule.evidenceId, sourceSpan: "held-out policy fixture" }],
    egress: {
      dataClassification: "internal",
      allowedRuntimeProviders: [runtime],
    },
    authoredBy: "user:code-quality-eval",
    authoredAt,
  };
  switch (rule.kind) {
    case "decision":
      return { ...common, statement: rule.statement };
    case "domain_term":
      return {
        ...common,
        term: rule.term,
        definition: rule.definition,
        avoid: rule.avoid,
      };
    case "invariant":
      return {
        ...common,
        statement: rule.statement,
        verifiers: [{ kind: "test", ref: "workspace:test" }],
      };
  }
}
