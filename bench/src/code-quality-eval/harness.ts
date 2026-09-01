import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { ClaudeCodeQualityRunner } from "./claude-runner.js";
import {
  CodexCodeQualityRunner,
  type CodexExecutionInput,
  type CodexExecutionResult,
} from "./codex-runner.js";
import type {
  CodeQualityArm,
  CodeQualityReport,
  CodeQualityRunConfig,
  CodeQualityRunResult,
  CodeQualityScenario,
} from "./contracts.js";
import { publishScenarioState } from "./kontext-state.js";
import { buildCodeQualityReport } from "./report.js";
import { createScenarioWorkspace, evaluateWorkspace } from "./workspace.js";

export interface CodeQualityHarnessDependencies {
  readonly execute?: (input: CodexExecutionInput) => Promise<CodexExecutionResult>;
  readonly publishState?: typeof publishScenarioState;
  readonly onProgress?: (message: string) => void;
}

export async function runCodeQualityEvaluation(input: {
  readonly repositoryRoot: string;
  readonly scenarios: readonly CodeQualityScenario[];
  readonly config: CodeQualityRunConfig;
  readonly dependencies?: CodeQualityHarnessDependencies;
}): Promise<CodeQualityReport> {
  const runner =
    input.config.runtime === "claude"
      ? new ClaudeCodeQualityRunner()
      : new CodexCodeQualityRunner();
  const execute = input.dependencies?.execute ?? ((execution) => runner.execute(execution));
  const publishState = input.dependencies?.publishState ?? publishScenarioState;
  const progress = input.dependencies?.onProgress ?? (() => undefined);
  const runs: CodeQualityRunResult[] = [];

  for (let repetition = 1; repetition <= input.config.repetitions; repetition += 1) {
    for (const [scenarioIndex, scenario] of input.scenarios.entries()) {
      const arms: readonly CodeQualityArm[] =
        (scenarioIndex + repetition) % 2 === 0 ? ["baseline", "kontext"] : ["kontext", "baseline"];
      for (const arm of arms) {
        progress(`[${scenario.scenarioId} r${repetition}] ${arm} starting`);
        const run = await runArm({
          repositoryRoot: input.repositoryRoot,
          scenario,
          repetition,
          arm,
          config: input.config,
          execute,
          publishState,
        });
        runs.push(run);
        const passed = run.hiddenAssertions.filter((assertion) => assertion.passed).length;
        progress(
          `[${scenario.scenarioId} r${repetition}] ${arm} finished: hidden ${passed}/${run.hiddenAssertions.length}, public ${run.publicTestsPassed ? "pass" : "fail"}`,
        );
      }
    }
  }

  return buildCodeQualityReport({
    config: input.config,
    scenarios: input.scenarios.map((scenario) => scenario.scenarioId),
    runs,
  });
}

async function runArm(input: {
  readonly repositoryRoot: string;
  readonly scenario: CodeQualityScenario;
  readonly repetition: number;
  readonly arm: CodeQualityArm;
  readonly config: CodeQualityRunConfig;
  readonly execute: (input: CodexExecutionInput) => Promise<CodexExecutionResult>;
  readonly publishState: typeof publishScenarioState;
}): Promise<CodeQualityRunResult> {
  const workspace = await createScenarioWorkspace(input.scenario);
  const pluginDataDirectory = await mkdtemp(
    path.join(tmpdir(), `kontext-code-eval-state-${input.scenario.scenarioId}-`),
  );
  const startedAt = new Date();
  let execution: CodexExecutionResult;
  try {
    if (input.arm === "kontext") {
      await input.publishState({
        scenario: input.scenario,
        baseRevision: workspace.baseRevision,
        repositoryRoot: input.repositoryRoot,
        pluginDataDirectory,
        runtime: input.config.runtime,
      });
    }
    execution = await input.execute({
      arm: input.arm,
      scenario: input.scenario,
      workspacePath: workspace.workspacePath,
      repositoryRoot: input.repositoryRoot,
      ...(input.arm === "kontext" ? { pluginDataDirectory } : {}),
      config: input.config,
    });
  } catch (error) {
    execution = {
      exitCode: 1,
      stdout: "",
      stderr: errorMessage(error),
      durationMilliseconds: Date.now() - startedAt.getTime(),
      kontextToolsObserved: [],
    };
  }

  try {
    const evaluation = await evaluateWorkspace(input.scenario, workspace.workspacePath);
    const canonicalTermsPresent = input.scenario.canonicalTerms.filter((term) =>
      evaluation.source.includes(term),
    );
    const canonicalTermsMissing = input.scenario.canonicalTerms.filter(
      (term) => !canonicalTermsPresent.includes(term),
    );
    const outOfScopePaths = evaluation.changedPaths.filter(
      (changedPath) => changedPath !== input.scenario.sourceFile,
    );
    const contextConsulted =
      input.arm === "kontext" &&
      execution.kontextToolsObserved.includes("kontext_prepare_task") &&
      execution.kontextToolsObserved.includes("kontext_begin_logic");
    const evaluationEligible =
      execution.exitCode === 0 && (input.arm === "baseline" || contextConsulted);
    const finishedAt = new Date();
    const diagnostic = [
      execution.stderr.trim(),
      ...(evaluationEligible ? [] : [execution.stdout.trim()]),
    ]
      .filter(Boolean)
      .join("\n");
    return {
      runId: `${input.scenario.scenarioId}:r${input.repetition}:${input.arm}`,
      scenarioId: input.scenario.scenarioId,
      repetition: input.repetition,
      arm: input.arm,
      model: input.config.model,
      reasoningEffort: input.config.reasoningEffort,
      startedAt: startedAt.toISOString(),
      finishedAt: finishedAt.toISOString(),
      durationMilliseconds: execution.durationMilliseconds,
      runtimeExitCode: execution.exitCode,
      ...(diagnostic ? { runtimeDiagnostic: diagnostic.slice(-12_000) } : {}),
      ...(execution.inputTokens === undefined ? {} : { inputTokens: execution.inputTokens }),
      ...(execution.outputTokens === undefined ? {} : { outputTokens: execution.outputTokens }),
      publicTestsPassed: evaluation.publicTestsPassed,
      hiddenAssertions: evaluation.hidden.assertions,
      canonicalTermsPresent,
      canonicalTermsMissing,
      changedPaths: evaluation.changedPaths,
      outOfScopePaths,
      kontextToolsObserved: execution.kontextToolsObserved,
      contextConsulted,
      evaluationEligible,
      source: evaluation.source,
      patch: evaluation.patch,
    };
  } finally {
    await Promise.all([
      rm(workspace.workspacePath, { recursive: true, force: true }),
      rm(pluginDataDirectory, { recursive: true, force: true }),
    ]);
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}
