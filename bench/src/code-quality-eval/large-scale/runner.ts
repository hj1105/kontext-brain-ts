import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  claudeArguments,
  extractClaudeToolCalls,
  extractClaudeUsage,
  writeClaudeMcpConfig,
} from "../claude-runner.js";
import {
  type KontextToolCall,
  codexRuntimeArguments,
  codexSubscriptionEnvironment,
  extractCodexUsage,
  extractKontextToolCalls,
  runCodexCommand,
} from "../codex-runner.js";
import type { CodeQualityArm, CodeQualityRunConfig } from "../contracts.js";
import type { LargeScaleReport, LargeScaleRunConfig, LargeScaleRunResult } from "./contracts.js";
import { buildLargeScaleReport } from "./report.js";
import {
  type LargeScaleRetrieval,
  renderLargeScaleContext,
  retrieveLargeScaleContext,
} from "./retrieval.js";
import { type LargeScaleLogicTarget, largeScaleTaskId, publishLargeScaleState } from "./state.js";
import {
  type LargeScaleWorkspace,
  createLargeScaleWorkspace,
  gradeLargeScaleWorkspace,
} from "./workspace.js";

export interface LargeScaleExecutionInput {
  readonly arm: CodeQualityArm;
  readonly workspacePath: string;
  readonly repositoryRoot: string;
  readonly pluginDataDirectory?: string;
  readonly prompt: string;
  readonly config: CodeQualityRunConfig;
}

export interface LargeScaleExecutionResult {
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
  readonly durationMilliseconds: number;
  readonly inputTokens?: number;
  readonly outputTokens?: number;
  readonly kontextToolCalls: readonly KontextToolCall[];
}

export interface LargeScaleHarnessDependencies {
  readonly execute?: (input: LargeScaleExecutionInput) => Promise<LargeScaleExecutionResult>;
  readonly retrieve?: () => Promise<LargeScaleRetrieval>;
  readonly publishState?: typeof publishLargeScaleState;
  readonly onProgress?: (message: string) => void;
}

export async function runLargeScaleEvaluation(input: {
  readonly repositoryRoot: string;
  readonly config: LargeScaleRunConfig;
  readonly existingRuns?: readonly LargeScaleRunResult[];
  readonly dependencies?: LargeScaleHarnessDependencies;
}): Promise<LargeScaleReport> {
  const progress = input.dependencies?.onProgress ?? (() => undefined);
  const execute = input.dependencies?.execute ?? executeLargeScaleRuntime;
  const publishState = input.dependencies?.publishState ?? publishLargeScaleState;
  const existingRuns = input.existingRuns ?? [];
  const missing = (repetition: number, arm: CodeQualityArm): boolean =>
    !existingRuns.some(
      (run) => run.repetition === repetition && run.arm === arm && run.evaluationEligible,
    );
  const needsRag = Array.from({ length: input.config.repetitions }, (_, index) => index + 1).some(
    (repetition) => input.config.arms.includes("rag") && missing(repetition, "rag"),
  );
  const retrieval = needsRag
    ? await (input.dependencies?.retrieve ?? retrieveLargeScaleContext)()
    : undefined;
  const runs: LargeScaleRunResult[] = existingRuns.filter(
    (run) => run.repetition <= input.config.repetitions && input.config.arms.includes(run.arm),
  );

  for (let repetition = 1; repetition <= input.config.repetitions; repetition += 1) {
    for (const arm of rotateArms(input.config.arms, repetition - 1)) {
      if (!missing(repetition, arm)) {
        progress(`[large-scale r${repetition}] ${arm} already eligible; skipping`);
        continue;
      }
      progress(`[large-scale r${repetition}] ${arm} starting`);
      const run = await runLargeScaleArm({
        repositoryRoot: input.repositoryRoot,
        config: input.config,
        repetition,
        arm,
        execute,
        publishState,
        ...(arm === "rag" && retrieval ? { retrieval } : {}),
      });
      runs.push(run);
      progress(
        `[large-scale r${repetition}] ${arm} finished: eligible=${run.evaluationEligible}, success=${run.taskSuccess}, recall=${run.grade.targetRecall.toFixed(2)}, precision=${run.grade.collateralPrecision.toFixed(2)}, hidden=${run.grade.hiddenPassed}/${run.grade.hiddenTotal}`,
      );
    }
  }

  return buildLargeScaleReport({
    config: input.config,
    runs: runs.sort(
      (left, right) => left.repetition - right.repetition || left.arm.localeCompare(right.arm),
    ),
  });
}

async function runLargeScaleArm(input: {
  readonly repositoryRoot: string;
  readonly config: LargeScaleRunConfig;
  readonly repetition: number;
  readonly arm: CodeQualityArm;
  readonly execute: (input: LargeScaleExecutionInput) => Promise<LargeScaleExecutionResult>;
  readonly publishState: typeof publishLargeScaleState;
  readonly retrieval?: LargeScaleRetrieval;
}): Promise<LargeScaleRunResult> {
  const workspace = await createLargeScaleWorkspace();
  const pluginDataDirectory = await mkdtemp(path.join(tmpdir(), "kontext-large-scale-state-"));
  const startedAt = new Date();
  let targets: readonly LargeScaleLogicTarget[] = [];
  let execution: LargeScaleExecutionResult;
  try {
    if (input.arm === "kontext") {
      const state = await input.publishState({
        workspace,
        runtime: input.config.runtime,
        repositoryRoot: input.repositoryRoot,
        pluginDataDirectory,
      });
      targets = state.targets;
    }
    const prompt = largeScalePrompt({
      arm: input.arm,
      workspacePath: workspace.workspacePath,
      runtime: input.config.runtime,
      targets,
      ...(input.retrieval ? { retrieval: input.retrieval } : {}),
    });
    execution = await input.execute({
      arm: input.arm,
      workspacePath: workspace.workspacePath,
      repositoryRoot: input.repositoryRoot,
      ...(input.arm === "kontext" ? { pluginDataDirectory } : {}),
      prompt,
      config: input.config,
    });
  } catch (error) {
    execution = {
      exitCode: 1,
      stdout: "",
      stderr: errorMessage(error),
      durationMilliseconds: Date.now() - startedAt.getTime(),
      kontextToolCalls: [],
    };
  }

  try {
    const grade = await gradeLargeScaleWorkspace(workspace);
    const consultation = assessKontextConsultation(execution.kontextToolCalls, targets.length);
    const evaluationEligible =
      execution.exitCode === 0 && (input.arm !== "kontext" || consultation.complete);
    const taskSuccess = evaluationEligible && isSuccessfulGrade(grade);
    const finishedAt = new Date();
    const diagnostic = [
      execution.stderr.trim(),
      ...(evaluationEligible ? [] : [execution.stdout.trim()]),
    ]
      .filter(Boolean)
      .join("\n");
    return {
      runId: `large-scale:r${input.repetition}:${input.arm}`,
      repetition: input.repetition,
      arm: input.arm,
      model: input.config.model,
      reasoningEffort: input.config.reasoningEffort,
      startedAt: startedAt.toISOString(),
      finishedAt: finishedAt.toISOString(),
      durationMilliseconds: execution.durationMilliseconds,
      runtimeExitCode: execution.exitCode,
      ...(diagnostic ? { runtimeDiagnostic: diagnostic.slice(-20_000) } : {}),
      ...(execution.inputTokens === undefined ? {} : { inputTokens: execution.inputTokens }),
      ...(execution.outputTokens === undefined ? {} : { outputTokens: execution.outputTokens }),
      kontextToolCalls: execution.kontextToolCalls,
      expectedLogicConsultations: targets.length,
      observedLogicConsultations: consultation.beginCalls,
      contextConsulted: consultation.complete,
      evaluationEligible,
      taskSuccess,
      ...(input.retrieval
        ? {
            retrieval: {
              documentIds: input.retrieval.documents.map((document) => document.documentId),
              governingRetrieved: input.retrieval.governingRetrieved,
              governingTotal: input.retrieval.governingTotal,
            },
          }
        : {}),
      grade,
    };
  } finally {
    await Promise.all([
      rm(workspace.workspacePath, { recursive: true, force: true }),
      rm(pluginDataDirectory, { recursive: true, force: true }),
    ]);
  }
}

export function largeScalePrompt(input: {
  readonly arm: CodeQualityArm;
  readonly workspacePath: string;
  readonly runtime: CodeQualityRunConfig["runtime"];
  readonly targets: readonly LargeScaleLogicTarget[];
  readonly retrieval?: LargeScaleRetrieval;
  readonly createdAt?: string;
}): string {
  const shared = [
    "Complete the coding task in TASK.md autonomously.",
    `Work only inside ${input.workspacePath}.`,
    "Inspect the repository to find every source location affected by the issue.",
    "Edit only files under src/. Do not change TASK.md, package.json, or tests.",
    "Do not inspect files outside this workspace and do not use the network for implementation guidance.",
    "Run npm test before finishing. Do not ask questions.",
  ].join("\n");
  if (input.arm === "baseline") return `${shared}\n`;
  if (input.arm === "rag") {
    return `${shared}\n\nThe following source documents were retrieved independently from the documentation corpus. They may contain unrelated or superseded material. Reconcile their dates and scope, follow the current applicable policy, and use its exact domain language.\n\n${input.retrieval ? renderLargeScaleContext(input.retrieval) : "No documentation was retrieved."}\n`;
  }

  const contract = {
    taskId: largeScaleTaskId,
    intent: "Implement the retry-delay change requested in TASK.md.",
    acceptance: [
      {
        criterionId: "criterion:large-scale:current-policy",
        statement: "Every affected behavior-bearing symbol follows the current applicable policy.",
        verifier: { kind: "test", ref: "workspace:test" },
      },
      {
        criterionId: "criterion:large-scale:no-collateral",
        statement: "Unrelated retry behavior remains unchanged.",
        verifier: { kind: "test", ref: "workspace:test" },
      },
    ],
    nonGoals: ["Changing tests", "Changing unrelated retry behavior", "Changing public APIs"],
    targets: input.targets.map((target) => target.plannedSymbolId),
    risk: "medium",
  } as const;
  const requests = input.targets.map((target) => ({
    taskId: largeScaleTaskId,
    workspacePath: input.workspacePath,
    logic: {
      workItemId: target.workItemId,
      plannedSymbolIds: [target.plannedSymbolId],
    },
    runtimeProvider: input.runtime,
    receiptTtlSeconds: 1_800,
    totalTokenBudget: 12_000,
    optionalEvidenceTokenBudget: 3_000,
  }));

  return `${shared}\n\nThe kontext_brain MCP server is the only source of private current policy and governed-symbol scope. Follow this protocol exactly:\n1. Call kontext_prepare_task once with contract=${JSON.stringify(contract)} and createdAt=${input.createdAt ?? "the current ISO timestamp"}.\n2. Process the following Logic Work Items in order, one at a time: ${JSON.stringify(requests)}. Before editing each item, call kontext_begin_logic with that exact object. Each call must return current status, editingAllowed=true, and a Context Receipt.\n3. For that one behavior-bearing symbol, obey the receipt's exact allowed paths and its mandatory Decisions, Domain Terms, and Invariants. Evidence is provenance, not a competing instruction. Do not infer policy or scope from another item's receipt.\n4. Make that item's edit, then call kontext_check_change first with tier=fast and then tier=targeted, using the exact taskId, workItemId, workspacePath, a current ISO observedAt, and a slightly later ISO nextAttemptAt.\n5. Only then begin the next Logic Work Item. If context is stale, refresh and begin again. If it is conflict, inaccessible, unavailable, or editing is denied, stop instead of guessing.\n6. After all items, run npm test.\n`;
}

export function assessKontextConsultation(
  calls: readonly KontextToolCall[],
  expectedLogicConsultations: number,
): { readonly prepareCalls: number; readonly beginCalls: number; readonly complete: boolean } {
  const prepareCalls = new Set(
    calls.filter((call) => call.name === "kontext_prepare_task").map((call) => call.callId),
  ).size;
  const beginCalls = new Set(
    calls.filter((call) => call.name === "kontext_begin_logic").map((call) => call.callId),
  ).size;
  return {
    prepareCalls,
    beginCalls,
    complete: prepareCalls >= 1 && beginCalls >= expectedLogicConsultations,
  };
}

export function isSuccessfulGrade(
  grade: Awaited<ReturnType<typeof gradeLargeScaleWorkspace>>,
): boolean {
  return (
    grade.targetRecall === 1 &&
    grade.collateralPrecision === 1 &&
    grade.hiddenPassed === grade.hiddenTotal &&
    grade.regressionFailures === 0 &&
    grade.publicTestsPassed &&
    grade.canonicalTermPresent &&
    grade.sharedConstantHonoured
  );
}

export async function executeLargeScaleRuntime(
  input: LargeScaleExecutionInput,
): Promise<LargeScaleExecutionResult> {
  if (input.config.runtime === "claude") return executeClaude(input);
  const result = await runCodexCommand({
    command: "codex",
    args: codexRuntimeArguments(input),
    stdin: input.prompt,
    timeoutMilliseconds: input.config.timeoutMilliseconds,
    environment: codexSubscriptionEnvironment(),
  });
  const usage = extractCodexUsage(result.stdout);
  return {
    ...result,
    ...usage,
    kontextToolCalls: extractKontextToolCalls(result.stdout),
  };
}

async function executeClaude(input: LargeScaleExecutionInput): Promise<LargeScaleExecutionResult> {
  const mcpConfigPath =
    input.arm === "kontext" && input.pluginDataDirectory
      ? await writeClaudeMcpConfig(input.repositoryRoot, input.pluginDataDirectory)
      : undefined;
  try {
    const result = await runCodexCommand({
      command: "claude",
      args: claudeArguments({
        arm: input.arm,
        workspacePath: input.workspacePath,
        config: input.config,
        maxTurns: 100,
        ...(mcpConfigPath ? { mcpConfigPath } : {}),
      }),
      stdin: input.prompt,
      timeoutMilliseconds: input.config.timeoutMilliseconds,
      environment: codexSubscriptionEnvironment(),
      cwd: input.workspacePath,
    });
    const usage = extractClaudeUsage(result.stdout);
    return {
      ...result,
      ...usage,
      kontextToolCalls: extractClaudeToolCalls(result.stdout),
    };
  } finally {
    if (mcpConfigPath) await rm(path.dirname(mcpConfigPath), { recursive: true, force: true });
  }
}

function rotateArms(arms: readonly CodeQualityArm[], offset: number): readonly CodeQualityArm[] {
  if (arms.length === 0) return [];
  const start = ((offset % arms.length) + arms.length) % arms.length;
  return [...arms.slice(start), ...arms.slice(0, start)];
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}
