import { existsSync, readFileSync, realpathSync } from "node:fs";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  type CleanLatencyReport,
  type CleanLatencySystem,
  runCleanLatency,
} from "./clean-latency.js";
import type { DatasetId } from "./contracts.js";
import { writeJsonAtomic } from "./jsonl.js";

interface CleanLatencySuiteRow {
  readonly datasetId: Extract<DatasetId, "graphrag-bench-medical" | "graphrag-bench-novel">;
  readonly system: CleanLatencySystem;
  readonly indexSourceDirectory: string;
  readonly outputDirectoryName: string;
}

interface CleanLatencySuiteConfig {
  readonly schemaVersion: "1.0.0";
  readonly outputRoot: string;
  readonly rows: readonly CleanLatencySuiteRow[];
}

async function main(): Promise<void> {
  const repositoryRoot = resolve(fileURLToPath(import.meta.url), "../../../..");
  const localEnvironment = resolve(repositoryRoot, ".env.local");
  if (existsSync(localEnvironment)) process.loadEnvFile(localEnvironment);
  const { configPath, skipHostGuard } = parseArgs(process.argv.slice(2));
  const config = JSON.parse(readFileSync(configPath, "utf8")) as CleanLatencySuiteConfig;
  validateCleanLatencySuiteConfig(config);
  const outputRoot = resolve(dirname(configPath), config.outputRoot);
  const reports: CleanLatencyReport[] = [];
  for (const row of config.rows) {
    const workDirectory = join(outputRoot, row.outputDirectoryName);
    const reportPath = join(workDirectory, "clean-latency-report.json");
    const report = existsSync(reportPath)
      ? (JSON.parse(readFileSync(reportPath, "utf8")) as CleanLatencyReport)
      : await runCleanLatency({
          repositoryRoot,
          workDirectory,
          datasetId: row.datasetId,
          system: row.system,
          indexSourceDirectory: resolve(dirname(configPath), row.indexSourceDirectory),
          skipHostGuard,
        });
    if (report.datasetId !== row.datasetId || report.system !== row.system)
      throw new Error(`Existing clean report identity mismatch at ${reportPath}`);
    const expectedIndexSource = resolve(dirname(configPath), row.indexSourceDirectory);
    if (report.indexSource.path !== realpathSync(expectedIndexSource))
      throw new Error(`Existing clean report index source mismatch at ${reportPath}`);
    reports.push(report);
    writeJsonAtomic(join(outputRoot, "clean-latency-suite-progress.json"), {
      schemaVersion: "1.0.0",
      completedRows: reports.map((item) => ({
        datasetId: item.datasetId,
        system: item.system,
        status: item.assessment.status,
        completedAt: item.completedAt,
      })),
    });
    if (report.assessment.status !== "valid")
      throw new Error(
        `Clean latency row is invalid and the suite stopped: ${row.datasetId}/${row.system}: ${report.assessment.reasons.join("; ")}`,
      );
  }
  assertSharedSamples(reports);
  writeJsonAtomic(join(outputRoot, "clean-latency-suite-report.json"), {
    schemaVersion: "1.0.0",
    completedAt: new Date().toISOString(),
    rows: reports,
  });
  process.stdout.write(`${JSON.stringify(reports, null, 2)}\n`);
}

export function validateCleanLatencySuiteConfig(config: CleanLatencySuiteConfig): void {
  if (config.schemaVersion !== "1.0.0") throw new Error("Unsupported clean suite schema");
  if (!config.outputRoot.trim()) throw new Error("outputRoot is required");
  const systems: readonly CleanLatencySystem[] = [
    "kontext-v15",
    "kontext-v13",
    "lightrag-1.5.6",
    "microsoft-graphrag-3.1.1",
  ];
  const datasets = ["graphrag-bench-medical", "graphrag-bench-novel"] as const;
  const expected = new Set(
    datasets.flatMap((dataset) => systems.map((system) => `${dataset}\0${system}`)),
  );
  const actual = new Set(config.rows.map((row) => `${row.datasetId}\0${row.system}`));
  if (config.rows.length !== expected.size || actual.size !== expected.size)
    throw new Error("Clean suite must contain each Medical/Novel x four-system row exactly once");
  for (const key of expected) {
    if (!actual.has(key)) throw new Error(`Clean suite row missing: ${key.replace("\0", "/")}`);
  }
  const outputNames = new Set<string>();
  for (const row of config.rows) {
    if (!row.indexSourceDirectory.trim()) throw new Error("Every row requires an index source");
    if (
      !row.outputDirectoryName.trim() ||
      isAbsolute(row.outputDirectoryName) ||
      basename(row.outputDirectoryName) !== row.outputDirectoryName
    )
      throw new Error(`Unsafe outputDirectoryName ${row.outputDirectoryName}`);
    if (outputNames.has(row.outputDirectoryName))
      throw new Error(`Duplicate outputDirectoryName ${row.outputDirectoryName}`);
    outputNames.add(row.outputDirectoryName);
  }
}

function assertSharedSamples(reports: readonly CleanLatencyReport[]): void {
  for (const datasetId of ["graphrag-bench-medical", "graphrag-bench-novel"] as const) {
    const digests = new Set(
      reports
        .filter((report) => report.datasetId === datasetId)
        .map((report) => report.sampleDigest),
    );
    if (digests.size !== 1)
      throw new Error(`${datasetId} clean rows do not share one deterministic sample digest`);
  }
}

function parseArgs(args: readonly string[]): {
  readonly configPath: string;
  readonly skipHostGuard: boolean;
} {
  let configPath: string | null = null;
  let skipHostGuard = false;
  for (let index = 0; index < args.length; index += 1) {
    const name = args[index];
    if (name === "--skip-host-guard") {
      skipHostGuard = true;
      continue;
    }
    if (name !== "--config") throw new Error(`Unexpected argument ${name ?? "<missing>"}`);
    const value = args[index + 1];
    if (!value) throw new Error("--config requires a path");
    configPath = resolve(value);
    index += 1;
  }
  if (!configPath) throw new Error("--config is required");
  return { configPath, skipHostGuard };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    process.stderr.write(`${(error as Error).stack ?? (error as Error).message}\n`);
    process.exitCode = 1;
  });
}
