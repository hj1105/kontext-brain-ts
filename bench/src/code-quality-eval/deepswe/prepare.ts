import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { runWorkspaceCommand } from "../workspace.js";
import type {
  DeepSwePreparationManifest,
  DeepSwePrepareOptions,
  DeepSwePreparedArm,
  DeepSweTaskSnapshot,
} from "./contracts.js";
import { buildContextBundle, loadDeepSweCorpus, sha256, stableJson } from "./corpus.js";
import { planCodexWorkerImage } from "./worker-image.js";

export async function prepareDeepSweEvaluation(
  options: DeepSwePrepareOptions,
): Promise<DeepSwePreparationManifest> {
  validateOptions(options);
  const runtimeProvider =
    options.runtime === "codex-subscription" ? "codex" : modelRuntimeProvider(options.model);
  const agentVersion = requiredAgentVersion(options);
  const datasetTasksPath = path.resolve(options.datasetTasksPath);
  const deepSweRoot = path.dirname(datasetTasksPath);
  const discovered = await discoverTaskIds(datasetTasksPath);
  const selected = selectTaskIds(
    discovered,
    options.taskIds,
    options.taskLimit,
    options.sampleSeed,
  );
  const actualDeepSweRevision = await gitRevision(deepSweRoot);
  if (options.deepSweRevision && options.deepSweRevision !== actualDeepSweRevision) {
    throw new Error(
      `DeepSWE revision mismatch: ${actualDeepSweRevision} != ${options.deepSweRevision}`,
    );
  }
  const tasks: DeepSweTaskSnapshot[] = [];
  const corpora = new Map<string, Awaited<ReturnType<typeof loadDeepSweCorpus>>>();
  for (const taskId of selected) {
    const taskPath = path.join(datasetTasksPath, taskId);
    const [instruction, taskToml, corpus] = await Promise.all([
      readFile(path.join(taskPath, "instruction.md"), "utf8"),
      readFile(path.join(taskPath, "task.toml"), "utf8"),
      loadDeepSweCorpus({
        corpusRoot: options.corpusRoot,
        taskId,
        taskPath,
        datasetTasksPath,
      }),
    ]);
    const configuredTaskId = tomlString(taskToml, "task_id");
    if (configuredTaskId !== taskId) {
      throw new Error(`DeepSWE task id mismatch in ${taskId}/task.toml: ${configuredTaskId}`);
    }
    const baseCommit = tomlString(taskToml, "base_commit_hash");
    if (corpus.baseCodeRevision !== baseCommit) {
      throw new Error(
        `Corpus base code revision mismatch for ${taskId}: ${corpus.baseCodeRevision} != ${baseCommit}`,
      );
    }
    if (corpus.runtimeProvider !== runtimeProvider) {
      throw new Error(
        `Corpus runtime provider mismatch for ${taskId}: ${corpus.runtimeProvider} != ${runtimeProvider}`,
      );
    }
    const instructionSha256 = sha256(stripPierCanary(instruction));
    if ([...tasks].some((task) => task.instructionSha256 === instructionSha256)) {
      throw new Error(`Duplicate DeepSWE instruction content: ${taskId}`);
    }
    corpora.set(taskId, corpus);
    tasks.push({
      taskId,
      taskPath,
      instructionSha256,
      taskTomlSha256: sha256(taskToml),
      baseCommit,
      language: tomlString(taskToml, "language"),
      dockerImage: tomlString(taskToml, "docker_image"),
    });
  }

  const runDirectory = path.resolve(options.runDirectory);
  const jobsDirectory = path.resolve(options.jobsDirectory);
  const manifestsDirectory = path.join(runDirectory, "manifests");
  await mkdir(manifestsDirectory, { recursive: true, mode: 0o700 });
  await mkdir(jobsDirectory, { recursive: true, mode: 0o700 });
  const contextToolPath = path.join(
    options.repositoryRoot,
    "bench",
    "src",
    "code-quality-eval",
    "deepswe",
    "context_tool.py",
  );
  const adapterDirectory = path.dirname(contextToolPath);
  const corpusSha256ByTask = Object.fromEntries(
    tasks.map((task) => [
      task.taskId,
      buildContextBundle("baseline", required(corpora, task.taskId)).corpusSha256,
    ]),
  );
  const preparedAt = new Date().toISOString();
  const arms: DeepSwePreparedArm[] = [];
  const executionGroups = taskExecutionGroups(tasks, options);
  const workerImages = executionGroups.flatMap((group) =>
    group.workerImage ? [group.workerImage] : [],
  );
  for (const arm of options.arms) {
    const byInstructionSha256 = Object.fromEntries(
      tasks.map((task) => [
        task.instructionSha256,
        buildContextBundle(arm, required(corpora, task.taskId)),
      ]),
    );
    const contextIndex = {
      schemaVersion: 1,
      arm,
      deepSweRevision: actualDeepSweRevision,
      byInstructionSha256,
    } as const;
    const contextIndexPath = path.join(manifestsDirectory, `context-${arm}.json`);
    await writePrivateJson(contextIndexPath, contextIndex);
    for (const group of executionGroups) {
      const identityHash = sha256(
        stableJson({
          arm,
          deepSweRevision: actualDeepSweRevision,
          pierRevision: options.pierRevision,
          adapterRevision: options.adapterRevision,
          runtime: options.runtime,
          agentVersion,
          model: options.model,
          reasoningEffort: options.reasoningEffort,
          attempts: options.attempts,
          sampleSeed: options.sampleSeed,
          tasks: group.tasks.map((task) => task.taskId),
          workerImageIdentitySha256: group.workerImage?.identitySha256,
          corpusSha256ByTask,
        }),
      ).slice(0, 12);
      const jobName = `deepswe-${arm}-${identityHash}`;
      const jobConfigPath = path.join(manifestsDirectory, `pier-${arm}-${identityHash}.json`);
      const jobConfig = {
        job_name: jobName,
        jobs_dir: jobsDirectory,
        n_attempts: options.attempts,
        n_concurrent_trials: options.concurrency,
        quiet: true,
        retry: { max_retries: 0 },
        environment: {
          type: options.environment,
          force_build: false,
          delete: true,
          ...(group.workerImage ? { env: { PREBUILT_IMAGE_NAME: group.workerImage.tag } } : {}),
        },
        agents: [
          options.runtime === "codex-subscription"
            ? {
                import_path: "kontext_codex_agent:KontextCodexAgent",
                model_name: options.model,
                env: { CODEX_FORCE_AUTH_JSON: "true" },
                kwargs: {
                  context_index_path: contextIndexPath,
                  context_tool_path: contextToolPath,
                  reasoning_effort: options.reasoningEffort,
                  version: agentVersion,
                  command_model_name: options.model,
                  ...(group.workerImage ? { prebuilt_worker_image: true } : {}),
                },
              }
            : {
                import_path: "kontext_mini_swe_agent:KontextMiniSweAgent",
                model_name: options.model,
                kwargs: {
                  context_index_path: contextIndexPath,
                  context_tool_path: contextToolPath,
                  reasoning_effort: options.reasoningEffort,
                  version: agentVersion,
                },
              },
        ],
        tasks: group.tasks.map((task) => ({ path: task.taskPath })),
        datasets: [],
      };
      await writePrivateJson(jobConfigPath, jobConfig);
      arms.push({
        arm,
        runtime: options.runtime,
        billingMode: options.runtime === "codex-subscription" ? "subscription" : "api",
        taskIds: group.tasks.map((task) => task.taskId),
        ...(group.workerImage
          ? { workerImageIdentitySha256: group.workerImage.identitySha256 }
          : {}),
        jobName,
        jobConfigPath,
        contextIndexPath,
        expectedJobResultPath: path.join(jobsDirectory, jobName, "result.json"),
        command: [
          options.pierBinary,
          "run",
          "--config",
          jobConfigPath,
          "--yes",
          ...(options.runtime === "mini-swe-api" && options.envFile
            ? ["--env-file", path.resolve(options.envFile)]
            : []),
        ],
      });
    }
  }
  const manifest: DeepSwePreparationManifest = {
    schemaVersion: 1,
    benchmark: "deepswe-kontext-ab",
    preparedAt,
    deepSweRevision: actualDeepSweRevision,
    pierRevision: options.pierRevision,
    adapterRevision: options.adapterRevision,
    runtime: options.runtime,
    agentVersion,
    model: options.model,
    reasoningEffort: options.reasoningEffort,
    attempts: options.attempts,
    sampleSeed: options.sampleSeed,
    tasks,
    arms,
    workerImages,
    corpusSha256ByTask,
  };
  await writePrivateJson(path.join(runDirectory, "preparation.json"), manifest);
  await writeFile(path.join(manifestsDirectory, "pythonpath.txt"), `${adapterDirectory}\n`, {
    encoding: "utf8",
    mode: 0o600,
  });
  return manifest;
}

function taskExecutionGroups(
  tasks: readonly DeepSweTaskSnapshot[],
  options: DeepSwePrepareOptions,
): readonly {
  readonly tasks: readonly DeepSweTaskSnapshot[];
  readonly workerImage?: ReturnType<typeof planCodexWorkerImage>;
}[] {
  if (options.runtime !== "codex-subscription" || options.environment !== "docker") {
    return [{ tasks }];
  }
  const byBaseImage = new Map<string, DeepSweTaskSnapshot[]>();
  for (const task of tasks) {
    const group = byBaseImage.get(task.dockerImage) ?? [];
    group.push(task);
    byBaseImage.set(task.dockerImage, group);
  }
  return Array.from(byBaseImage.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([baseImage, groupedTasks]) => ({
      tasks: groupedTasks,
      workerImage: planCodexWorkerImage({
        baseImage,
        codexVersion: requiredAgentVersion(options),
        pierVersion: options.pierRevision,
      }),
    }));
}

export function stripPierCanary(source: string): string {
  const lines = source.split("\n");
  let index = 0;
  while (
    index < lines.length &&
    /^(?:<!--.*canary.*-->|#.*canary.*)$/i.test(lines[index]?.trim() ?? "")
  ) {
    index += 1;
  }
  while (index < lines.length && !(lines[index]?.trim() ?? "")) index += 1;
  return lines.slice(index).join("\n");
}

export function selectTaskIds(
  discovered: readonly string[],
  requested: readonly string[] | undefined,
  limit: number | undefined,
  seed: number,
): readonly string[] {
  const available = new Set(discovered);
  const candidates = requested?.length ? [...new Set(requested)] : [...discovered];
  for (const taskId of candidates) {
    if (!available.has(taskId)) throw new Error(`Unknown DeepSWE task: ${taskId}`);
  }
  const ordered = candidates.sort((left, right) => {
    const byHash = sha256(`${seed}\0${left}`).localeCompare(sha256(`${seed}\0${right}`));
    return byHash || left.localeCompare(right);
  });
  return limit === undefined ? ordered : ordered.slice(0, limit);
}

async function discoverTaskIds(tasksPath: string): Promise<readonly string[]> {
  const entries = await readdir(tasksPath, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isDirectory() && /^[a-z0-9][a-z0-9-]+$/.test(entry.name))
    .map((entry) => entry.name)
    .sort();
}

function tomlString(source: string, key: string): string {
  const match = source.match(new RegExp(`^${key}\\s*=\\s*"([^"]+)"\\s*$`, "m"));
  if (!match?.[1]) throw new Error(`DeepSWE task.toml is missing ${key}`);
  return match[1];
}

async function gitRevision(repositoryPath: string): Promise<string> {
  const result = await runWorkspaceCommand(repositoryPath, "git", ["rev-parse", "HEAD"]);
  if (result.exitCode !== 0) {
    throw new Error(`Cannot resolve DeepSWE revision: ${result.stderr || result.stdout}`);
  }
  return result.stdout.trim();
}

async function writePrivateJson(filePath: string, value: unknown): Promise<void> {
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, {
    encoding: "utf8",
    mode: 0o600,
  });
}

function required<T>(values: ReadonlyMap<string, T>, key: string): T {
  const value = values.get(key);
  if (!value) throw new Error(`Missing prepared value for ${key}`);
  return value;
}

function validateOptions(options: DeepSwePrepareOptions): void {
  if (!Number.isInteger(options.attempts) || options.attempts <= 0) {
    throw new Error("DeepSWE attempts must be a positive integer");
  }
  if (!Number.isInteger(options.concurrency) || options.concurrency <= 0) {
    throw new Error("DeepSWE concurrency must be a positive integer");
  }
  if (
    options.taskLimit !== undefined &&
    (!Number.isInteger(options.taskLimit) || options.taskLimit <= 0)
  ) {
    throw new Error("DeepSWE task limit must be a positive integer");
  }
  if (options.arms.length === 0) throw new Error("Select at least one DeepSWE arm");
  if (options.runtime === "codex-subscription") {
    if (!options.model.trim() || options.model.includes("/")) {
      throw new Error("Codex subscription model must use a bare Codex model name");
    }
    if (!options.codexVersion?.trim()) {
      throw new Error("Codex version is required for replayability");
    }
    if (options.envFile) {
      throw new Error("Codex subscription runtime does not accept --env-file");
    }
    return;
  }
  modelRuntimeProvider(options.model);
  if (!options.miniSweAgentVersion?.trim()) {
    throw new Error("mini-swe-agent version is required for replayability");
  }
}

function requiredAgentVersion(options: DeepSwePrepareOptions): string {
  const version =
    options.runtime === "codex-subscription" ? options.codexVersion : options.miniSweAgentVersion;
  if (!version?.trim()) throw new Error("Agent version is required for replayability");
  return version.trim();
}

function modelRuntimeProvider(model: string): string {
  const separator = model.indexOf("/");
  const provider = separator < 0 ? "" : model.slice(0, separator);
  const name = separator < 0 ? "" : model.slice(separator + 1);
  if (!provider.trim() || !name.trim()) {
    throw new Error("DeepSWE model must use provider/model format");
  }
  return provider;
}
