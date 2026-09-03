import path from "node:path";
import { codexSubscriptionEnvironment } from "../codex-runner.js";
import { runWorkspaceCommand } from "../workspace.js";
import type { DeepSwePreparationManifest, DeepSweReport, DeepSweTrialResult } from "./contracts.js";
import { readPierArmResults } from "./pier-results.js";
import { buildDeepSweReport } from "./report.js";
import { ensureCodexWorkerImage } from "./worker-image.js";

export interface DeepSweRunnerDependencies {
  readonly execute?: typeof runWorkspaceCommand;
  readonly readResults?: typeof readPierArmResults;
  readonly ensureWorkerImage?: typeof ensureCodexWorkerImage;
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
  const ensureWorkerImage = input.dependencies?.ensureWorkerImage ?? ensureCodexWorkerImage;
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
  const runtimeEnvironment =
    input.manifest.runtime === "codex-subscription"
      ? codexSubscriptionEnvironment(process.env)
      : { ...process.env };
  const trials: DeepSweTrialResult[] = [];
  const workerImages = [];
  for (const workerImage of input.manifest.workerImages ?? []) {
    progress(`[deepswe image] ensuring ${workerImage.tag}`);
    workerImages.push(
      await ensureWorkerImage({
        spec: {
          baseImage: workerImage.baseImage,
          codexVersion: workerImage.codexVersion,
          pierVersion: workerImage.pierVersion,
          recipeVersion: workerImage.recipeVersion,
        },
        manifestsDirectory: path.join(path.dirname(input.manifest.arms[0]?.jobConfigPath ?? "")),
        repositoryRoot: input.repositoryRoot,
        execute,
      }),
    );
  }
  const executedManifest: DeepSwePreparationManifest = {
    ...input.manifest,
    workerImages,
  };
  for (const arm of rotateArms(input.manifest.arms, input.manifest.sampleSeed)) {
    progress(
      `[deepswe ${arm.arm}] starting ${arm.taskIds?.length ?? input.manifest.tasks.length} tasks`,
    );
    const [command, ...args] = arm.command;
    if (!command) throw new Error(`DeepSWE ${arm.arm} arm has no Pier command`);
    const result = await execute(input.repositoryRoot, command, args, {
      ...runtimeEnvironment,
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
  return buildDeepSweReport({ manifest: executedManifest, trials });
}

function rotateArms<T>(arms: readonly T[], seed: number): readonly T[] {
  if (!arms.length) return [];
  const offset = ((seed % arms.length) + arms.length) % arms.length;
  return [...arms.slice(offset), ...arms.slice(0, offset)];
}
