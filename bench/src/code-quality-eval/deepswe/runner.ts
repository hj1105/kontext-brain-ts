import path from "node:path";
import { runWorkspaceCommand } from "../workspace.js";
import type { DeepSwePreparationManifest, DeepSweReport, DeepSweTrialResult } from "./contracts.js";
import { readPierArmResults } from "./pier-results.js";
import { buildDeepSweReport } from "./report.js";

export interface DeepSweRunnerDependencies {
  readonly execute?: typeof runWorkspaceCommand;
  readonly readResults?: typeof readPierArmResults;
  readonly onProgress?: (message: string) => void;
}

export async function runPreparedDeepSweEvaluation(input: {
  readonly repositoryRoot: string;
  readonly manifest: DeepSwePreparationManifest;
  readonly envFile?: string;
  readonly dryRun?: boolean;
  readonly dependencies?: DeepSweRunnerDependencies;
}): Promise<DeepSweReport | undefined> {
  if (input.dryRun) return undefined;
  const execute = input.dependencies?.execute ?? runWorkspaceCommand;
  const readResults = input.dependencies?.readResults ?? readPierArmResults;
  const progress = input.dependencies?.onProgress ?? (() => undefined);
  const adapterDirectory = path.join(
    input.repositoryRoot,
    "bench",
    "src",
    "code-quality-eval",
    "deepswe",
  );
  const pythonPath = [adapterDirectory, process.env.PYTHONPATH]
    .filter(Boolean)
    .join(path.delimiter);
  const trials: DeepSweTrialResult[] = [];
  for (const arm of rotateArms(input.manifest.arms, input.manifest.sampleSeed)) {
    progress(`[deepswe ${arm.arm}] starting ${input.manifest.tasks.length} tasks`);
    const [command, ...args] = arm.command;
    if (!command) throw new Error(`DeepSWE ${arm.arm} arm has no Pier command`);
    const result = await execute(input.repositoryRoot, command, args, {
      ...process.env,
      PYTHONPATH: pythonPath,
    });
    if (result.exitCode !== 0) {
      throw new Error(
        `Pier ${arm.arm} job failed before result ingestion: ${result.stderr || result.stdout}`,
      );
    }
    const armTrials = await readResults(arm);
    trials.push(...armTrials);
    progress(
      `[deepswe ${arm.arm}] finished: ${armTrials.filter((trial) => trial.success).length}/${armTrials.filter((trial) => trial.eligible).length} eligible rollouts passed`,
    );
  }
  return buildDeepSweReport({ manifest: input.manifest, trials });
}

function rotateArms<T>(arms: readonly T[], seed: number): readonly T[] {
  if (!arms.length) return [];
  const offset = ((seed % arms.length) + arms.length) % arms.length;
  return [...arms.slice(offset), ...arms.slice(0, offset)];
}
