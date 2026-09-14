import type {
  DeepSweArm,
  DeepSweArmSummary,
  DeepSwePairedComparison,
  DeepSweTrialResult,
} from "./contracts.js";

export function summarizeDeepSweArm(
  arm: DeepSweArm,
  trials: readonly DeepSweTrialResult[],
): DeepSweArmSummary {
  const armTrials = trials.filter((trial) => trial.arm === arm);
  const taskGroups = groupEligibleByTask(armTrials);
  const passFractions = [...taskGroups.values()].map(passFraction);
  const runScores = rolloutScores(taskGroups);
  return {
    arm,
    tasks: taskGroups.size,
    eligibleTrials: armTrials.filter((trial) => trial.eligible).length,
    excludedTrials: armTrials.filter((trial) => !trial.eligible).length,
    protocolCompleteTrials: armTrials.filter((trial) => trial.context.protocolComplete).length,
    passAt1: mean(passFractions),
    passAt4: mean(
      [...taskGroups.values()].map((taskTrials) =>
        taskTrials.slice(0, 4).some((trial) => trial.success) ? 1 : 0,
      ),
    ),
    ...(confidence95(runScores) ? { runToRun95: confidence95(runScores) } : {}),
    ...medianMetricFields(armTrials.filter((trial) => trial.eligible)),
  };
}

export function compareDeepSweArms(
  treatment: "kontext",
  control: "baseline" | "rag",
  trials: readonly DeepSweTrialResult[],
  bootstrapSamples = 10_000,
  bootstrapSeed = 0,
): DeepSwePairedComparison {
  const treatmentTasks = groupEligibleByTask(trials.filter((trial) => trial.arm === treatment));
  const controlTasks = groupEligibleByTask(trials.filter((trial) => trial.arm === control));
  const taskIds = [...treatmentTasks.keys()].filter((taskId) => controlTasks.has(taskId)).sort();
  const passAt1Deltas = taskIds.map(
    (taskId) =>
      passFraction(required(treatmentTasks, taskId)) - passFraction(required(controlTasks, taskId)),
  );
  const passAt4Deltas = taskIds.map(
    (taskId) =>
      solvedWithinFour(required(treatmentTasks, taskId)) -
      solvedWithinFour(required(controlTasks, taskId)),
  );
  const passAt1Interval = pairedBootstrap95(passAt1Deltas, bootstrapSamples, bootstrapSeed);
  const passAt4Interval = pairedBootstrap95(passAt4Deltas, bootstrapSamples, bootstrapSeed);
  return {
    treatment,
    control,
    comparableTasks: passAt1Deltas.length,
    passAt1Delta: mean(passAt1Deltas),
    passAt4Delta: mean(passAt4Deltas),
    ...(passAt1Interval ? { passAt1ClusterBootstrap95: passAt1Interval } : {}),
    ...(passAt4Interval ? { passAt4ClusterBootstrap95: passAt4Interval } : {}),
  };
}

export function pairedBootstrap95(
  taskDeltas: readonly number[],
  samples = 10_000,
  seed = 0,
): readonly [number, number] | undefined {
  if (taskDeltas.length < 2 || samples < 2) return undefined;
  const random = xorshift32(seed || 0x9e3779b9);
  const estimates: number[] = [];
  for (let sample = 0; sample < samples; sample += 1) {
    let total = 0;
    for (let index = 0; index < taskDeltas.length; index += 1) {
      total += arrayValue(taskDeltas, Math.floor(random() * taskDeltas.length));
    }
    estimates.push(total / taskDeltas.length);
  }
  estimates.sort((left, right) => left - right);
  return [percentile(estimates, 0.025), percentile(estimates, 0.975)];
}

function groupEligibleByTask(
  trials: readonly DeepSweTrialResult[],
): Map<string, DeepSweTrialResult[]> {
  const grouped = new Map<string, DeepSweTrialResult[]>();
  for (const trial of trials) {
    if (!trial.eligible) continue;
    const current = grouped.get(trial.taskId) ?? [];
    current.push(trial);
    grouped.set(trial.taskId, current);
  }
  for (const taskTrials of grouped.values()) {
    taskTrials.sort((left, right) => left.rolloutIndex - right.rolloutIndex);
  }
  return grouped;
}

function rolloutScores(tasks: ReadonlyMap<string, readonly DeepSweTrialResult[]>): number[] {
  const maxRollouts = Math.max(0, ...[...tasks.values()].map((trials) => trials.length));
  const scores: number[] = [];
  for (let index = 0; index < maxRollouts; index += 1) {
    const outcomes = [...tasks.values()].flatMap((trials) => {
      const trial = trials[index];
      return trial ? [trial.success ? 1 : 0] : [];
    });
    if (outcomes.length) scores.push(mean(outcomes));
  }
  return scores;
}

function confidence95(values: readonly number[]): readonly [number, number] | undefined {
  if (values.length < 2) return undefined;
  const center = mean(values);
  const variance =
    values.reduce((total, value) => total + (value - center) ** 2, 0) / (values.length - 1);
  const margin = (1.96 * Math.sqrt(variance)) / Math.sqrt(values.length);
  return [Math.max(0, center - margin), Math.min(1, center + margin)];
}

function medianMetricFields(trials: readonly DeepSweTrialResult[]): {
  readonly medianOutputTokens?: number;
  readonly medianDurationMilliseconds?: number;
  readonly medianCostUsd?: number;
  readonly medianAgentSteps?: number;
} {
  const outputTokens = median(trials.flatMap((trial) => value(trial.outputTokens)));
  const duration = median(trials.flatMap((trial) => value(trial.durationMilliseconds)));
  const cost = median(trials.flatMap((trial) => value(trial.costUsd)));
  const steps = median(trials.flatMap((trial) => value(trial.agentSteps)));
  return {
    ...(outputTokens === undefined ? {} : { medianOutputTokens: outputTokens }),
    ...(duration === undefined ? {} : { medianDurationMilliseconds: duration }),
    ...(cost === undefined ? {} : { medianCostUsd: cost }),
    ...(steps === undefined ? {} : { medianAgentSteps: steps }),
  };
}

function value(candidate: number | undefined): readonly number[] {
  return candidate === undefined ? [] : [candidate];
}

function passFraction(trials: readonly DeepSweTrialResult[]): number {
  return trials.length ? trials.filter((trial) => trial.success).length / trials.length : 0;
}

function solvedWithinFour(trials: readonly DeepSweTrialResult[]): number {
  return trials.slice(0, 4).some((trial) => trial.success) ? 1 : 0;
}

function mean(values: readonly number[]): number {
  return values.length ? values.reduce((total, value) => total + value, 0) / values.length : 0;
}

function median(values: readonly number[]): number | undefined {
  if (!values.length) return undefined;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? arrayValue(sorted, middle)
    : (arrayValue(sorted, middle - 1) + arrayValue(sorted, middle)) / 2;
}

function percentile(sorted: readonly number[], probability: number): number {
  const index = (sorted.length - 1) * probability;
  const lower = Math.floor(index);
  const fraction = index - lower;
  const lowerValue = arrayValue(sorted, lower);
  const upperValue = arrayValue(sorted, Math.min(lower + 1, sorted.length - 1));
  return lowerValue + (upperValue - lowerValue) * fraction;
}

function xorshift32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x1_0000_0000;
  };
}

function required<T>(values: ReadonlyMap<string, T>, key: string): T {
  const value = values.get(key);
  if (!value) throw new Error(`Missing task result for ${key}`);
  return value;
}

function arrayValue(values: readonly number[], index: number): number {
  const result = values[index];
  if (result === undefined) throw new Error(`Missing numeric value at index ${index}`);
  return result;
}
