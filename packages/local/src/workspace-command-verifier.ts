import { spawn } from "node:child_process";
import { readFile, realpath, stat } from "node:fs/promises";
import path from "node:path";
import {
  type VerifierAdapter,
  VerifierInfrastructureError,
  type VerifierRegistry,
} from "@kontext-brain/orchestrator";
import type { VerifierKind, VerifierRef } from "@kontext-brain/spec";
import { z } from "zod";

const verifierKinds = ["test", "typecheck", "build", "lint", "query", "manual_review"] as const;
const definitionSchema = z
  .object({
    kind: z.enum(verifierKinds),
    ref: z.string().min(1),
    command: z.string().min(1),
    args: z.array(z.string().max(8192)).max(128).default([]),
    timeoutMilliseconds: z.number().int().min(1_000).max(1_800_000).optional(),
  })
  .strict();
const configSchema = z
  .object({
    schemaVersion: z.literal(1),
    verifiers: z.array(definitionSchema),
  })
  .strict();
const packageSchema = z
  .object({
    packageManager: z.string().optional(),
    scripts: z.record(z.string()).optional(),
  })
  .passthrough();

type VerifierDefinition = z.infer<typeof definitionSchema>;

export interface WorkspaceCommandVerifierOptions {
  readonly configPath?: string;
  readonly defaultTimeoutMilliseconds?: number;
  readonly outputLimitBytes?: number;
}

const standardScripts = new Map<string, string>([
  ["typecheck\u0000workspace:typecheck", "typecheck"],
  ["test\u0000workspace:test", "test"],
  ["build\u0000workspace:build", "build"],
  ["lint\u0000workspace:lint", "lint"],
]);

export class WorkspaceCommandVerifierAdapter implements VerifierAdapter {
  private readonly configPath: string;
  private readonly defaultTimeoutMilliseconds: number;
  private readonly outputLimitBytes: number;

  constructor(options: WorkspaceCommandVerifierOptions = {}) {
    this.configPath = options.configPath ?? path.join(".kontext", "verifiers.json");
    this.defaultTimeoutMilliseconds = options.defaultTimeoutMilliseconds ?? 600_000;
    this.outputLimitBytes = options.outputLimitBytes ?? 1_048_576;
  }

  async execute(request: Parameters<VerifierAdapter["execute"]>[0]) {
    const workspacePath = await requireWorkspace(request.workspacePath);
    const definition =
      (await configuredDefinition(workspacePath, this.configPath, request.requirement.verifier)) ??
      (await standardDefinition(workspacePath, request.requirement.verifier));
    if (!definition) {
      throw new VerifierInfrastructureError(
        `No trusted workspace verifier definition exists for ${request.requirement.verifier.kind}:${request.requirement.verifier.ref}`,
      );
    }
    return executeDefinition(
      workspacePath,
      definition,
      definition.timeoutMilliseconds ?? this.defaultTimeoutMilliseconds,
      this.outputLimitBytes,
    );
  }
}

export function registerWorkspaceCommandVerifiers(
  registry: VerifierRegistry,
  adapter: VerifierAdapter = new WorkspaceCommandVerifierAdapter(),
): void {
  for (const kind of verifierKinds) registry.registerFallback(kind, adapter);
}

async function configuredDefinition(
  workspacePath: string,
  configPath: string,
  verifier: VerifierRef,
): Promise<VerifierDefinition | undefined> {
  const absoluteConfigPath = path.resolve(workspacePath, configPath);
  if (!isWithinOrEqual(workspacePath, absoluteConfigPath)) {
    throw new VerifierInfrastructureError("Verifier configuration path escapes the workspace");
  }
  let raw: string;
  try {
    raw = await readFile(absoluteConfigPath, "utf8");
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return undefined;
    throw new VerifierInfrastructureError(
      `Cannot read verifier configuration: ${errorMessage(error)}`,
    );
  }
  let config: z.infer<typeof configSchema>;
  try {
    config = configSchema.parse(JSON.parse(raw));
  } catch (error) {
    throw new VerifierInfrastructureError(`Invalid verifier configuration: ${errorMessage(error)}`);
  }
  const definitions = new Map<string, VerifierDefinition>();
  for (const definition of config.verifiers) {
    const key = verifierKey(definition);
    if (definitions.has(key)) {
      throw new VerifierInfrastructureError(
        `Duplicate verifier definition ${definition.kind}:${definition.ref}`,
      );
    }
    definitions.set(key, definition);
  }
  return definitions.get(verifierKey(verifier));
}

async function standardDefinition(
  workspacePath: string,
  verifier: VerifierRef,
): Promise<VerifierDefinition | undefined> {
  const script = standardScripts.get(verifierKey(verifier));
  if (!script) return undefined;
  let packageJson: z.infer<typeof packageSchema>;
  try {
    packageJson = packageSchema.parse(
      JSON.parse(await readFile(path.join(workspacePath, "package.json"), "utf8")),
    );
  } catch (error) {
    throw new VerifierInfrastructureError(
      `Cannot resolve workspace script ${script}: ${errorMessage(error)}`,
    );
  }
  if (!packageJson.scripts?.[script]) {
    throw new VerifierInfrastructureError(`Workspace package.json has no "${script}" script`);
  }
  const command = packageManagerCommand(packageJson.packageManager);
  return {
    kind: verifier.kind,
    ref: verifier.ref,
    command,
    args: ["run", script],
  };
}

async function requireWorkspace(workspacePath: string): Promise<string> {
  try {
    const canonical = await realpath(path.resolve(workspacePath));
    if (!(await stat(canonical)).isDirectory()) {
      throw new Error("path is not a directory");
    }
    return canonical;
  } catch (error) {
    throw new VerifierInfrastructureError(
      `Verifier workspace is unavailable: ${errorMessage(error)}`,
    );
  }
}

function packageManagerCommand(packageManager: string | undefined): string {
  const name = packageManager?.split("@", 1)[0];
  return name === "pnpm" || name === "yarn" || name === "bun" || name === "npm" ? name : "npm";
}

async function executeDefinition(
  workspacePath: string,
  definition: VerifierDefinition,
  timeoutMilliseconds: number,
  outputLimitBytes: number,
): Promise<{ readonly result: "passed" | "failed"; readonly output: unknown }> {
  return new Promise((resolve, reject) => {
    const child = spawn(definition.command, definition.args, {
      cwd: workspacePath,
      env: process.env,
      shell: false,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    let outputBytes = 0;
    let settled = false;
    let killTimer: NodeJS.Timeout | undefined;

    const finish = (action: () => void): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      if (killTimer) clearTimeout(killTimer);
      action();
    };
    const terminate = (message: string): void => {
      child.kill("SIGTERM");
      killTimer = setTimeout(() => child.kill("SIGKILL"), 5_000);
      killTimer.unref();
      finish(() => reject(new VerifierInfrastructureError(message)));
    };
    const capture = (target: Buffer[], chunk: Buffer): void => {
      if (settled) return;
      outputBytes += chunk.byteLength;
      if (outputBytes > outputLimitBytes) {
        terminate(`Verifier output exceeded ${outputLimitBytes} bytes`);
        return;
      }
      target.push(chunk);
    };
    child.stdout.on("data", (chunk: Buffer) => capture(stdout, chunk));
    child.stderr.on("data", (chunk: Buffer) => capture(stderr, chunk));
    child.once("error", (error) => {
      finish(() =>
        reject(new VerifierInfrastructureError(`Cannot start verifier: ${error.message}`)),
      );
    });
    child.once("close", (code, signal) => {
      finish(() =>
        resolve({
          result: code === 0 ? "passed" : "failed",
          output: {
            command: definition.command,
            args: definition.args,
            exitCode: code,
            signal,
            stdout: Buffer.concat(stdout).toString("utf8"),
            stderr: Buffer.concat(stderr).toString("utf8"),
          },
        }),
      );
    });
    const timeout = setTimeout(
      () => terminate(`Verifier exceeded ${timeoutMilliseconds}ms`),
      timeoutMilliseconds,
    );
    timeout.unref();
  });
}

function verifierKey(verifier: Pick<VerifierRef, "kind" | "ref">): string {
  return `${verifier.kind}\u0000${verifier.ref}`;
}

function isWithinOrEqual(workspacePath: string, candidatePath: string): boolean {
  const relative = path.relative(workspacePath, candidatePath);
  return relative === "" || (relative !== ".." && !relative.startsWith(`..${path.sep}`));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isNodeError(value: unknown): value is NodeJS.ErrnoException {
  return value instanceof Error && "code" in value;
}

export const workspaceVerifierKinds: readonly VerifierKind[] = verifierKinds;
