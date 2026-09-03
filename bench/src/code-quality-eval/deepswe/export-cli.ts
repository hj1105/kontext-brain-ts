#!/usr/bin/env node
import { randomUUID } from "node:crypto";
import { chmod, mkdir, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileTaskContextRepository, resolvePluginDataDirectory } from "@kontext-brain/local";
import { runWorkspaceCommand } from "../workspace.js";
import { buildContextBundle } from "./corpus.js";
import { exportDeepSweCorpus } from "./export-corpus.js";

interface ExportCliOptions {
  readonly taskId: string;
  readonly organizationId: string;
  readonly runtimeProvider: string;
  readonly outputPath: string;
  readonly dataDirectory: string;
  readonly generatorRevision?: string;
  readonly allowEmpty: boolean;
}

async function main(): Promise<void> {
  if (process.argv.includes("--help") || process.argv.includes("-h")) {
    process.stdout.write(helpText());
    return;
  }
  const repositoryRoot = fileURLToPath(new URL("../../../../", import.meta.url));
  const options = parseOptions(process.argv.slice(2));
  const generatorRevision =
    options.generatorRevision ?? (await resolveCleanRevision(repositoryRoot));
  const repository = new FileTaskContextRepository(options.dataDirectory);
  const [current, prepared] = await Promise.all([
    repository.getCurrent(options.taskId),
    repository.get(options.taskId),
  ]);
  if (!prepared) throw new Error(`Task "${options.taskId}" has no prepared context`);
  const corpus = exportDeepSweCorpus({
    taskId: options.taskId,
    organizationId: options.organizationId,
    runtimeProvider: options.runtimeProvider,
    generatorRevision,
    prepared,
    current,
    allowEmpty: options.allowEmpty,
  });
  await atomicPrivateWrite(options.outputPath, `${JSON.stringify(corpus, null, 2)}\n`);
  process.stdout.write(
    `${JSON.stringify(
      {
        taskId: corpus.taskId,
        outputPath: options.outputPath,
        contextDigest: corpus.contextDigest,
        sourceFreshnessDigest: corpus.sourceFreshnessDigest,
        corpusSha256: buildContextBundle("baseline", corpus).corpusSha256,
        evidence: corpus.evidence.length,
        normativeRecords: corpus.normativeRecords.length,
      },
      null,
      2,
    )}\n`,
  );
}

function parseOptions(args: readonly string[]): ExportCliOptions {
  let taskId: string | undefined;
  let organizationId: string | undefined;
  let runtimeProvider: string | undefined;
  let outputPath: string | undefined;
  let dataDirectory = resolvePluginDataDirectory();
  let generatorRevision: string | undefined;
  let allowEmpty = false;
  for (let index = 0; index < args.length; index += 1) {
    const option = args[index];
    if (option === "--allow-empty") {
      allowEmpty = true;
      continue;
    }
    const value = args[index + 1]?.trim();
    if (!value) throw new Error(`Missing value for ${option}`);
    switch (option) {
      case "--task-id":
        taskId = value;
        break;
      case "--organization-id":
        organizationId = value;
        break;
      case "--runtime-provider":
        runtimeProvider = value;
        break;
      case "--output":
        outputPath = path.resolve(value);
        break;
      case "--data-dir":
        dataDirectory = path.resolve(value);
        break;
      case "--generator-revision":
        generatorRevision = value;
        break;
      default:
        throw new Error(`Unknown option: ${option}`);
    }
    index += 1;
  }
  if (!taskId) throw new Error("--task-id is required");
  if (!organizationId) throw new Error("--organization-id is required");
  if (!runtimeProvider) throw new Error("--runtime-provider is required");
  if (!outputPath) throw new Error("--output is required");
  return {
    taskId,
    organizationId,
    runtimeProvider,
    outputPath,
    dataDirectory,
    ...(generatorRevision ? { generatorRevision } : {}),
    allowEmpty,
  };
}

async function resolveCleanRevision(repositoryRoot: string): Promise<string> {
  const [revision, trackedStatus, exporterStatus] = await Promise.all([
    runWorkspaceCommand(repositoryRoot, "git", ["rev-parse", "HEAD"]),
    runWorkspaceCommand(repositoryRoot, "git", ["status", "--porcelain", "--untracked-files=no"]),
    runWorkspaceCommand(repositoryRoot, "git", [
      "status",
      "--porcelain",
      "--untracked-files=all",
      "--",
      "bench/src/code-quality-eval/deepswe",
      "bench/package.json",
      "packages/context/src",
      "packages/local/src",
      "pnpm-lock.yaml",
    ]),
  ]);
  if (revision.exitCode !== 0 || trackedStatus.exitCode !== 0 || exporterStatus.exitCode !== 0) {
    throw new Error("Cannot resolve Kontext generator revision");
  }
  if (trackedStatus.stdout.trim() || exporterStatus.stdout.trim()) {
    throw new Error("Refusing to export a scored DeepSWE corpus from a dirty Kontext checkout");
  }
  return revision.stdout.trim();
}

async function atomicPrivateWrite(filePath: string, contents: string): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true, mode: 0o700 });
  const temporaryPath = `${filePath}.${process.pid}.${randomUUID()}.tmp`;
  await writeFile(temporaryPath, contents, { encoding: "utf8", mode: 0o600 });
  await chmod(temporaryPath, 0o600);
  await rename(temporaryPath, filePath);
}

function helpText(): string {
  return `Usage: pnpm --filter @kontext-brain/bench code-quality:deepswe:export -- \\
  --task-id <DeepSWE task id> \\
  --organization-id <Kontext Organization> \\
  --runtime-provider <provider allowed by Evidence egress> \\
  --output <corpus-root/task-id.json> [options]

Options:
  --data-dir <path>            Kontext sidecar data (defaults to plugin data directory)
  --generator-revision <ref>   Explicit exporter revision; otherwise requires a clean Git checkout
  --allow-empty                Permit an infrastructure-only empty corpus
`;
}

main().catch((error) => {
  process.stderr.write(
    `kontext DeepSWE corpus export failed: ${error instanceof Error ? error.message : String(error)}\n`,
  );
  process.exit(1);
});
