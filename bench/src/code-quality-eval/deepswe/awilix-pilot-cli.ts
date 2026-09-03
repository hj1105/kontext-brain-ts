#!/usr/bin/env node
import path from "node:path";
import { fileURLToPath } from "node:url";
import { runWorkspaceCommand } from "../workspace.js";
import { awilixAsyncInitializationPilot } from "./awilix-pilot-spec.js";
import { buildContextBundle } from "./corpus.js";
import { buildDeepSwePilotCorpus } from "./pilot-corpus.js";

interface Options {
  readonly checkoutPath: string;
  readonly outputPath: string;
  readonly dataDirectory: string;
  readonly runtimeProvider: string;
  readonly generatorRevision?: string;
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
  const result = await buildDeepSwePilotCorpus({
    spec: awilixAsyncInitializationPilot,
    checkoutPath: options.checkoutPath,
    outputPath: options.outputPath,
    dataDirectory: options.dataDirectory,
    runtimeProvider: options.runtimeProvider,
    generatorRevision,
  });
  process.stdout.write(
    `${JSON.stringify(
      {
        taskId: result.corpus.taskId,
        baseCodeRevision: result.corpus.baseCodeRevision,
        outputPath: options.outputPath,
        dataDirectory: options.dataDirectory,
        runtimeProvider: result.corpus.runtimeProvider,
        corpusSha256: buildContextBundle("baseline", result.corpus).corpusSha256,
        codeResources: result.codeResources,
        codeSymbols: result.codeSymbols,
        behaviorBearingSymbols: result.behaviorBearingSymbols,
        evidenceResources: result.evidenceResources,
        evidence: result.corpus.evidence.length,
        normativeRecords: result.corpus.normativeRecords.length,
        governingRecordIds: result.governingRecordIds,
      },
      null,
      2,
    )}\n`,
  );
}

function parseOptions(args: readonly string[]): Options {
  let checkoutPath: string | undefined;
  let outputPath: string | undefined;
  let dataDirectory: string | undefined;
  let runtimeProvider = "openai";
  let generatorRevision: string | undefined;
  for (let index = 0; index < args.length; index += 1) {
    const option = args[index];
    const value = args[index + 1]?.trim();
    if (!value) throw new Error(`Missing value for ${option}`);
    switch (option) {
      case "--checkout":
        checkoutPath = path.resolve(value);
        break;
      case "--output":
        outputPath = path.resolve(value);
        break;
      case "--data-dir":
        dataDirectory = path.resolve(value);
        break;
      case "--runtime-provider":
        runtimeProvider = value;
        break;
      case "--generator-revision":
        generatorRevision = value;
        break;
      default:
        throw new Error(`Unknown option: ${option}`);
    }
    index += 1;
  }
  if (!checkoutPath) throw new Error("--checkout is required");
  if (!outputPath) throw new Error("--output is required");
  return {
    checkoutPath,
    outputPath,
    dataDirectory: dataDirectory ?? `${outputPath}.sidecar`,
    runtimeProvider,
    ...(generatorRevision ? { generatorRevision } : {}),
  };
}

async function resolveCleanRevision(repositoryRoot: string): Promise<string> {
  const [revision, trackedStatus, pilotStatus] = await Promise.all([
    runWorkspaceCommand(repositoryRoot, "git", ["rev-parse", "HEAD"]),
    runWorkspaceCommand(repositoryRoot, "git", ["status", "--porcelain", "--untracked-files=no"]),
    runWorkspaceCommand(repositoryRoot, "git", [
      "status",
      "--porcelain",
      "--untracked-files=all",
      "--",
      "bench/src/code-quality-eval/deepswe",
      "bench/package.json",
    ]),
  ]);
  if (revision.exitCode !== 0 || trackedStatus.exitCode !== 0 || pilotStatus.exitCode !== 0) {
    throw new Error("Cannot resolve DeepSWE pilot generator revision");
  }
  if (trackedStatus.stdout.trim() || pilotStatus.stdout.trim()) {
    throw new Error("Refusing to generate a scored pilot corpus from a dirty checkout");
  }
  return revision.stdout.trim();
}

function helpText(): string {
  return `Usage: pnpm --filter @kontext-brain/bench code-quality:deepswe:pilot:awilix -- \\
  --checkout <Awilix checkout at 82ac179c> \\
  --output <corpus-root/awilix-async-container-initialization.json> [options]

Options:
  --data-dir <path>            Private sidecar state (defaults beside output)
  --runtime-provider <name>    Egress provider frozen into the corpus (default: openai)
  --generator-revision <ref>   Explicit externally pinned build revision
`;
}

main().catch((error) => {
  process.stderr.write(
    `Awilix DeepSWE pilot corpus failed: ${error instanceof Error ? error.message : String(error)}\n`,
  );
  process.exit(1);
});
