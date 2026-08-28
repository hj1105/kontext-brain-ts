import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  type CleanLatencyRunOptions,
  type CleanLatencySystem,
  runCleanLatency,
} from "./clean-latency.js";

async function main(): Promise<void> {
  const repositoryRoot = resolve(fileURLToPath(import.meta.url), "../../../..");
  const localEnvironment = resolve(repositoryRoot, ".env.local");
  if (existsSync(localEnvironment)) process.loadEnvFile(localEnvironment);
  const options = parseOptions(process.argv.slice(2), repositoryRoot);
  const report = await runCleanLatency(options);
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (report.assessment.status !== "valid") process.exitCode = 2;
}

function parseOptions(args: readonly string[], repositoryRoot: string): CleanLatencyRunOptions {
  const values = new Map<string, string>();
  let skipHostGuard = false;
  for (let index = 0; index < args.length; index += 1) {
    const name = args[index];
    if (!name) continue;
    if (name === "--skip-host-guard") {
      skipHostGuard = true;
      continue;
    }
    if (!name.startsWith("--")) throw new Error(`Unexpected argument ${name}`);
    const value = args[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for ${name}`);
    values.set(name, value);
    index += 1;
  }
  const dataset = required(values, "--dataset");
  if (dataset !== "graphrag-bench-medical" && dataset !== "graphrag-bench-novel")
    throw new Error(`Unsupported clean latency dataset ${dataset}`);
  const system = required(values, "--system");
  if (!isCleanLatencySystem(system)) throw new Error(`Unsupported clean latency system ${system}`);
  return {
    repositoryRoot,
    workDirectory: resolve(required(values, "--work-dir")),
    datasetId: dataset,
    system,
    indexSourceDirectory: resolve(required(values, "--index-source")),
    skipHostGuard,
  };
}

function required(values: ReadonlyMap<string, string>, name: string): string {
  const value = values.get(name);
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function isCleanLatencySystem(value: string): value is CleanLatencySystem {
  return ["kontext-v15", "kontext-v13", "lightrag-1.5.6", "microsoft-graphrag-3.1.1"].includes(
    value,
  );
}

main().catch((error) => {
  process.stderr.write(`${(error as Error).stack ?? (error as Error).message}\n`);
  process.exitCode = 1;
});
