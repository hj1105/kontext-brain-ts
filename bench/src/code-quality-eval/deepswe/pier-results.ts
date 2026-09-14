import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import type {
  DeepSweArm,
  DeepSweCapabilityFailureReason,
  DeepSweContextTelemetry,
  DeepSweExclusionReason,
  DeepSwePreparedArm,
  DeepSweTrialResult,
} from "./contracts.js";
import { sha256 } from "./corpus.js";

interface PierAgentContext {
  readonly n_input_tokens?: number;
  readonly n_cache_tokens?: number;
  readonly n_output_tokens?: number;
  readonly cost_usd?: number;
  readonly peak_context_tokens?: number;
  readonly n_agent_steps?: number;
}

interface PierTrialResult {
  readonly task_name?: string;
  readonly trial_name?: string;
  readonly verifier_result?: { readonly rewards?: Readonly<Record<string, number>> };
  readonly exception_info?: {
    readonly exception_type?: string;
    readonly exception_message?: string;
  };
  readonly agent_result?: PierAgentContext;
  readonly step_results?: readonly { readonly agent_result?: PierAgentContext }[];
  readonly started_at?: string;
  readonly finished_at?: string;
  readonly agent_execution?: { readonly started_at?: string; readonly finished_at?: string };
  readonly n_agent_steps?: number;
}

interface PierJobResult {
  readonly n_total_trials?: number;
  readonly trial_results?: readonly PierTrialResult[];
}

interface MiniSweMessage {
  readonly role?: string;
  readonly object?: string;
  readonly usage?: Readonly<Record<string, unknown>>;
  readonly extra?: {
    readonly response?: { readonly usage?: Readonly<Record<string, unknown>> };
  };
}

interface MiniSweTrajectory {
  readonly info?: {
    readonly model_stats?: { readonly instance_cost?: number };
  };
  readonly messages?: readonly MiniSweMessage[];
}

export async function readPierArmResults(arm: DeepSwePreparedArm): Promise<DeepSweTrialResult[]> {
  const parsed = JSON.parse(await readFile(arm.expectedJobResultPath, "utf8")) as PierJobResult;
  const pierTrials = parsed.trial_results?.length
    ? parsed.trial_results
    : await readTrialDirectoryResults(path.dirname(arm.expectedJobResultPath));
  if (parsed.n_total_trials !== undefined && pierTrials.length !== parsed.n_total_trials) {
    throw new Error(
      `Pier result in ${arm.jobName} declares ${parsed.n_total_trials} trials but ${pierTrials.length} trial results were found`,
    );
  }
  const grouped = new Map<string, PierTrialResult[]>();
  for (const trial of pierTrials) {
    const taskName = trial.task_name?.trim();
    if (!taskName) throw new Error(`Pier result in ${arm.jobName} has no task_name`);
    const taskId = taskName.split("/").at(-1);
    if (!taskId) throw new Error(`Pier result in ${arm.jobName} has invalid task_name`);
    const current = grouped.get(taskId) ?? [];
    current.push(trial);
    grouped.set(taskId, current);
  }
  const results: DeepSweTrialResult[] = [];
  for (const [taskId, trials] of [...grouped.entries()].sort(([a], [b]) => a.localeCompare(b))) {
    const ordered = [...trials].sort((left, right) =>
      `${left.started_at ?? ""}\0${left.trial_name ?? ""}`.localeCompare(
        `${right.started_at ?? ""}\0${right.trial_name ?? ""}`,
      ),
    );
    for (const [index, trial] of ordered.entries()) {
      results.push(await normalizeTrial(arm, taskId, trial, index + 1));
    }
  }
  return results;
}

async function readTrialDirectoryResults(
  jobDirectory: string,
): Promise<readonly PierTrialResult[]> {
  const entries = await readdir(jobDirectory, { withFileTypes: true });
  const trials = await Promise.all(
    entries
      .filter((entry) => entry.isDirectory())
      .map(async (entry) => {
        const resultPath = path.join(jobDirectory, entry.name, "result.json");
        try {
          return JSON.parse(await readFile(resultPath, "utf8")) as PierTrialResult;
        } catch (error) {
          if (isMissingFile(error)) return undefined;
          throw new Error(`Cannot read Pier trial result ${resultPath}`, { cause: error });
        }
      }),
  );
  return trials.filter((trial): trial is PierTrialResult => trial !== undefined);
}

async function normalizeTrial(
  arm: DeepSwePreparedArm,
  taskId: string,
  trial: PierTrialResult,
  rolloutIndex: number,
): Promise<DeepSweTrialResult> {
  const trialName = trial.trial_name?.trim();
  if (!trialName) throw new Error(`Pier result for ${taskId} has no trial_name`);
  const reward = trial.verifier_result?.rewards?.reward;
  const exceptionType = trial.exception_info?.exception_type;
  const exceptionMessage = trial.exception_info?.exception_message ?? "";
  const classification = classifyOutcome(reward, exceptionType, exceptionMessage);
  const trialDirectory = path.join(path.dirname(arm.expectedJobResultPath), trialName);
  const context = await readContextTelemetry([
    path.join(trialDirectory, "agent", "kontext-calls.jsonl"),
    path.join(trialDirectory, "artifacts", "kontext-agent", "kontext-calls.jsonl"),
  ]);
  const patchSha256 = await optionalFileHash(path.join(trialDirectory, "artifacts", "model.patch"));
  const trajectory = await firstFileIdentity([
    path.join(trialDirectory, "agent", "trajectory.json"),
    path.join(trialDirectory, "artifacts", "kontext-agent", "mini-swe-agent.trajectory.json"),
  ]);
  const nativeMetrics = await readMiniSweTrajectoryMetrics(
    path.join(trialDirectory, "artifacts", "kontext-agent", "mini-swe-agent.trajectory.json"),
  );
  const metrics = mergeMetrics(aggregateContexts(trial), nativeMetrics);
  const startedAt = trial.agent_execution?.started_at ?? trial.started_at;
  const finishedAt = trial.agent_execution?.finished_at ?? trial.finished_at;
  const durationMilliseconds = duration(startedAt, finishedAt);
  return {
    arm: arm.arm,
    taskId,
    trialName,
    rolloutIndex,
    eligible: classification.eligible,
    success: classification.success,
    ...(reward === undefined ? {} : { reward }),
    ...(classification.exclusionReason ? { exclusionReason: classification.exclusionReason } : {}),
    ...(classification.capabilityFailureReason
      ? { capabilityFailureReason: classification.capabilityFailureReason }
      : {}),
    ...(exceptionType ? { exceptionType } : {}),
    ...(startedAt ? { startedAt } : {}),
    ...(finishedAt ? { finishedAt } : {}),
    ...(durationMilliseconds === undefined ? {} : { durationMilliseconds }),
    ...metrics,
    ...(patchSha256 ? { patchSha256 } : {}),
    ...(trajectory ? { trajectoryPath: trajectory.path, trajectorySha256: trajectory.sha256 } : {}),
    context,
  };
}

export function classifyOutcome(
  reward: number | undefined,
  exceptionType?: string,
  exceptionMessage = "",
): {
  readonly eligible: boolean;
  readonly success: boolean;
  readonly exclusionReason?: DeepSweExclusionReason;
  readonly capabilityFailureReason?: DeepSweCapabilityFailureReason;
} {
  if (reward === 1) return { eligible: true, success: true };
  if (reward === 0) {
    return {
      eligible: true,
      success: false,
      capabilityFailureReason: "verifier_rejected",
    };
  }
  const combined = `${exceptionType ?? ""} ${exceptionMessage}`.toLowerCase();
  if (/agenttimeouterror|agent timeout|timed out/.test(combined)) {
    return { eligible: true, success: false, capabilityFailureReason: "agent_timeout" };
  }
  if (/context.{0,20}(exhaust|length|window)|maximum context/.test(combined)) {
    return { eligible: true, success: false, capabilityFailureReason: "context_exhausted" };
  }
  if (reward !== undefined && reward < 0) {
    return { eligible: false, success: false, exclusionReason: "verifier_error" };
  }
  if (/verifier|rewardfile|grading|grader/.test(combined)) {
    return { eligible: false, success: false, exclusionReason: "verifier_error" };
  }
  if (/provider|ratelimit|rate.limit|apierror|authentication|overloaded/.test(combined)) {
    return { eligible: false, success: false, exclusionReason: "provider_error" };
  }
  if (/network|connection|dns|socket|gateway/.test(combined)) {
    return { eligible: false, success: false, exclusionReason: "network_error" };
  }
  if (/agentsetup|agent install|setup timeout/.test(combined)) {
    return { eligible: false, success: false, exclusionReason: "agent_setup_error" };
  }
  if (/docker|modal|environment|image|container/.test(combined)) {
    return { eligible: false, success: false, exclusionReason: "environment_error" };
  }
  return {
    eligible: false,
    success: false,
    exclusionReason: "unclassified_infrastructure_error",
  };
}

function aggregateContexts(trial: PierTrialResult): {
  readonly inputTokens?: number;
  readonly cachedTokens?: number;
  readonly outputTokens?: number;
  readonly costUsd?: number;
  readonly agentSteps?: number;
  readonly peakContextTokens?: number;
} {
  const contexts = trial.agent_result
    ? [trial.agent_result]
    : (trial.step_results ?? []).flatMap((step) => (step.agent_result ? [step.agent_result] : []));
  const sum = (key: keyof PierAgentContext): number | undefined => {
    const values = contexts.map((context) => context[key]).filter(isNumber);
    return values.length ? values.reduce((total, value) => total + value, 0) : undefined;
  };
  const peaks = contexts.map((context) => context.peak_context_tokens).filter(isNumber);
  const inputTokens = sum("n_input_tokens");
  const cachedTokens = sum("n_cache_tokens");
  const outputTokens = sum("n_output_tokens");
  const costUsd = sum("cost_usd");
  const agentSteps = trial.n_agent_steps ?? sum("n_agent_steps");
  return {
    ...(inputTokens === undefined ? {} : { inputTokens }),
    ...(cachedTokens === undefined ? {} : { cachedTokens }),
    ...(outputTokens === undefined ? {} : { outputTokens }),
    ...(costUsd === undefined ? {} : { costUsd }),
    ...(agentSteps === undefined ? {} : { agentSteps }),
    ...(peaks.length ? { peakContextTokens: Math.max(...peaks) } : {}),
  };
}

async function readContextTelemetry(
  filePaths: readonly string[],
): Promise<DeepSweContextTelemetry> {
  const empty = {
    prepareCalls: 0,
    searchCalls: 0,
    beginLogicCalls: 0,
    fastCheckCalls: 0,
    targetedCheckCalls: 0,
    logicSymbols: [],
    fullyCheckedLogicSymbols: [],
    protocolComplete: false,
  };
  let contents: string;
  for (const filePath of filePaths) {
    try {
      contents = await readFile(filePath, "utf8");
      return parseContextTelemetry(contents);
    } catch {
      // Try Pier's convention-artifact fallback next.
    }
  }
  return empty;
}

function parseContextTelemetry(contents: string): DeepSweContextTelemetry {
  const events = contents
    .split(/\r?\n/)
    .filter(Boolean)
    .flatMap((line) => {
      try {
        return [
          JSON.parse(line) as {
            command?: string;
            arguments?: { tier?: string; path?: string; symbol?: string };
          },
        ];
      } catch {
        return [];
      }
    });
  const prepareCalls = events.filter((event) => event.command === "prepare-task").length;
  const logicSymbols = uniqueSymbols(events.filter((event) => event.command === "begin-logic"));
  const fastChecked = new Set(
    uniqueSymbols(
      events.filter(
        (event) => event.command === "check-change" && event.arguments?.tier === "fast",
      ),
    ),
  );
  const targetedChecked = new Set(
    uniqueSymbols(
      events.filter(
        (event) => event.command === "check-change" && event.arguments?.tier === "targeted",
      ),
    ),
  );
  const fullyCheckedLogicSymbols = logicSymbols.filter(
    (symbol) => fastChecked.has(symbol) && targetedChecked.has(symbol),
  );
  return {
    prepareCalls,
    searchCalls: events.filter((event) => event.command === "search").length,
    beginLogicCalls: events.filter((event) => event.command === "begin-logic").length,
    fastCheckCalls: events.filter(
      (event) => event.command === "check-change" && event.arguments?.tier === "fast",
    ).length,
    targetedCheckCalls: events.filter(
      (event) => event.command === "check-change" && event.arguments?.tier === "targeted",
    ).length,
    logicSymbols,
    fullyCheckedLogicSymbols,
    protocolComplete:
      prepareCalls === 1 &&
      logicSymbols.length > 0 &&
      fullyCheckedLogicSymbols.length === logicSymbols.length,
  };
}

async function readMiniSweTrajectoryMetrics(
  filePath: string,
): Promise<ReturnType<typeof aggregateContexts>> {
  let parsed: MiniSweTrajectory;
  try {
    parsed = JSON.parse(await readFile(filePath, "utf8")) as MiniSweTrajectory;
  } catch {
    return {};
  }
  const messages = parsed.messages ?? [];
  const usages = messages.map(messageUsage);
  const sum = (keys: readonly string[]): number | undefined => {
    const values = usages.flatMap((usage) => {
      for (const key of keys) {
        const value = finiteNumber(usage[key]);
        if (value !== undefined) return [value];
      }
      return [];
    });
    return values.length ? values.reduce((total, value) => total + value, 0) : undefined;
  };
  const inputTokens = sum(["prompt_tokens", "input_tokens"]);
  const cachedTokens = sumNested(
    usages,
    ["cache_read_input_tokens"],
    [
      ["input_tokens_details", "cached_tokens"],
      ["prompt_tokens_details", "cached_tokens"],
    ],
  );
  const outputTokens = sum(["completion_tokens", "output_tokens"]);
  const usageCost = sumNested(usages, ["cost"], [["cost_details", "upstream_inference_cost"]]);
  const instanceCost = finiteNumber(parsed.info?.model_stats?.instance_cost);
  const promptTokens = usages.flatMap((usage) => {
    const value = finiteNumber(usage.prompt_tokens) ?? finiteNumber(usage.input_tokens);
    return value === undefined ? [] : [value];
  });
  const agentSteps = messages.filter(
    (message) => message.role === "assistant" || message.object === "response",
  ).length;
  return {
    ...(inputTokens === undefined ? {} : { inputTokens }),
    ...(cachedTokens === undefined ? {} : { cachedTokens }),
    ...(outputTokens === undefined ? {} : { outputTokens }),
    ...(instanceCost === undefined && usageCost === undefined
      ? {}
      : { costUsd: instanceCost ?? usageCost }),
    ...(agentSteps ? { agentSteps } : {}),
    ...(promptTokens.length ? { peakContextTokens: Math.max(...promptTokens) } : {}),
  };
}

function messageUsage(message: MiniSweMessage): Readonly<Record<string, unknown>> {
  return message.extra?.response?.usage ?? message.usage ?? {};
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function sumNested(
  records: readonly Readonly<Record<string, unknown>>[],
  directKeys: readonly string[],
  nestedKeys: readonly (readonly [string, string])[],
): number | undefined {
  const values = records.flatMap((record) => {
    for (const key of directKeys) {
      const value = finiteNumber(record[key]);
      if (value !== undefined) return [value];
    }
    for (const [containerKey, valueKey] of nestedKeys) {
      const container = record[containerKey];
      if (!container || typeof container !== "object" || Array.isArray(container)) continue;
      const value = finiteNumber((container as Readonly<Record<string, unknown>>)[valueKey]);
      if (value !== undefined) return [value];
    }
    return [];
  });
  return values.length ? values.reduce((total, value) => total + value, 0) : undefined;
}

function mergeMetrics(
  primary: ReturnType<typeof aggregateContexts>,
  fallback: ReturnType<typeof aggregateContexts>,
): ReturnType<typeof aggregateContexts> {
  return {
    inputTokens: primary.inputTokens ?? fallback.inputTokens,
    cachedTokens: primary.cachedTokens ?? fallback.cachedTokens,
    outputTokens: primary.outputTokens ?? fallback.outputTokens,
    costUsd: primary.costUsd ?? fallback.costUsd,
    agentSteps: primary.agentSteps ?? fallback.agentSteps,
    peakContextTokens: primary.peakContextTokens ?? fallback.peakContextTokens,
  };
}

function uniqueSymbols(
  events: readonly {
    readonly arguments?: { readonly path?: string; readonly symbol?: string };
  }[],
): readonly string[] {
  return [
    ...new Set(
      events.flatMap((event) => {
        const relativePath = event.arguments?.path?.trim();
        const symbol = event.arguments?.symbol?.trim();
        return relativePath && symbol ? [`${relativePath}#${symbol}`] : [];
      }),
    ),
  ].sort();
}

async function optionalFileHash(filePath: string): Promise<string | undefined> {
  try {
    return sha256(await readFile(filePath));
  } catch {
    return undefined;
  }
}

async function firstFileIdentity(
  filePaths: readonly string[],
): Promise<{ readonly path: string; readonly sha256: string } | undefined> {
  for (const filePath of filePaths) {
    const hash = await optionalFileHash(filePath);
    if (hash) return { path: filePath, sha256: hash };
  }
  return undefined;
}

function duration(startedAt?: string, finishedAt?: string): number | undefined {
  if (!startedAt || !finishedAt) return undefined;
  const value = Date.parse(finishedAt) - Date.parse(startedAt);
  return Number.isFinite(value) && value >= 0 ? value : undefined;
}

function isNumber(value: number | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isMissingFile(error: unknown): boolean {
  return (
    error instanceof Error &&
    "code" in error &&
    (error as Error & { readonly code?: string }).code === "ENOENT"
  );
}
