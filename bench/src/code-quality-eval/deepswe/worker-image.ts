import { chmod, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { codexSubscriptionEnvironment } from "../codex-runner.js";
import { type WorkspaceCommandResult, runWorkspaceCommand } from "../workspace.js";
import { sha256, stableJson } from "./corpus.js";

export const CODEX_WORKER_IMAGE_RECIPE_VERSION = "pier-codex-v1";

const identityLabel = "io.kontext-brain.deepswe.codex-worker.identity";
const baseImageLabel = "io.kontext-brain.deepswe.codex-worker.base-image";
const codexVersionLabel = "io.kontext-brain.deepswe.codex-worker.codex-version";
const pierVersionLabel = "io.kontext-brain.deepswe.codex-worker.pier-version";
const recipeVersionLabel = "io.kontext-brain.deepswe.codex-worker.recipe-version";

export interface CodexWorkerImageSpec {
  readonly baseImage: string;
  readonly codexVersion: string;
  readonly pierVersion: string;
  readonly recipeVersion?: string;
}

export interface CodexWorkerImagePlan {
  readonly baseImage: string;
  readonly codexVersion: string;
  readonly pierVersion: string;
  readonly recipeVersion: string;
  readonly identitySha256: string;
  readonly tag: string;
  readonly labels: Readonly<Record<string, string>>;
}

export interface EnsuredCodexWorkerImage extends CodexWorkerImagePlan {
  readonly imageId: string;
  readonly reused: boolean;
  readonly dockerfilePath?: string;
}

export type WorkerImageCommand = (
  cwd: string,
  command: string,
  args: readonly string[],
  environment?: NodeJS.ProcessEnv,
) => Promise<WorkspaceCommandResult>;

export interface EnsureCodexWorkerImageInput {
  readonly spec: CodexWorkerImageSpec;
  readonly manifestsDirectory: string;
  readonly repositoryRoot: string;
  readonly dockerBinary?: string;
  readonly execute?: WorkerImageCommand;
}

const inFlightImages = new Map<string, Promise<EnsuredCodexWorkerImage>>();

export function planCodexWorkerImage(spec: CodexWorkerImageSpec): CodexWorkerImagePlan {
  const baseImage = requiredDockerValue(spec.baseImage, "base Docker image");
  const codexVersion = requiredDockerValue(spec.codexVersion, "Codex version");
  const pierVersion = requiredDockerValue(spec.pierVersion, "Pier version");
  const recipeVersion = requiredDockerValue(
    spec.recipeVersion ?? CODEX_WORKER_IMAGE_RECIPE_VERSION,
    "worker image recipe version",
  );
  const identitySha256 = sha256(
    stableJson({ baseImage, codexVersion, pierVersion, recipeVersion }),
  );
  const tag = `kontext-brain/deepswe-codex:${identitySha256.slice(0, 24)}`;
  return {
    baseImage,
    codexVersion,
    pierVersion,
    recipeVersion,
    identitySha256,
    tag,
    labels: {
      [identityLabel]: identitySha256,
      [baseImageLabel]: baseImage,
      [codexVersionLabel]: codexVersion,
      [pierVersionLabel]: pierVersion,
      [recipeVersionLabel]: recipeVersion,
    },
  };
}

export function ensureCodexWorkerImage(
  input: EnsureCodexWorkerImageInput,
): Promise<EnsuredCodexWorkerImage> {
  const plan = planCodexWorkerImage(input.spec);
  const inFlight = inFlightImages.get(plan.tag);
  if (inFlight) return inFlight;
  const pending = ensurePlannedCodexWorkerImage(input, plan);
  inFlightImages.set(plan.tag, pending);
  return pending.finally(() => {
    if (inFlightImages.get(plan.tag) === pending) inFlightImages.delete(plan.tag);
  });
}

async function ensurePlannedCodexWorkerImage(
  input: EnsureCodexWorkerImageInput,
  plan: CodexWorkerImagePlan,
): Promise<EnsuredCodexWorkerImage> {
  const execute = input.execute ?? runWorkspaceCommand;
  const docker = input.dockerBinary ?? "docker";
  const environment = codexSubscriptionEnvironment(process.env);
  const existing = await inspectImage({
    execute,
    docker,
    repositoryRoot: input.repositoryRoot,
    tag: plan.tag,
    environment,
  });
  if (existing && labelsMatch(existing.labels, plan.labels)) {
    return { ...plan, imageId: existing.imageId, reused: true };
  }

  const buildDirectory = path.join(
    path.resolve(input.manifestsDirectory),
    "worker-images",
    plan.identitySha256,
  );
  await mkdir(buildDirectory, { recursive: true, mode: 0o700 });
  await chmod(buildDirectory, 0o700);
  const dockerfilePath = path.join(buildDirectory, "Dockerfile");
  await writeFile(dockerfilePath, renderDockerfile(plan), { encoding: "utf8", mode: 0o600 });
  await chmod(dockerfilePath, 0o600);

  const build = await execute(
    input.repositoryRoot,
    docker,
    ["build", "--pull=false", "--tag", plan.tag, buildDirectory],
    environment,
  );
  if (build.exitCode !== 0) {
    throw new Error(`Cannot build pinned Codex worker image ${plan.tag}: ${diagnostic(build)}`);
  }
  const built = await inspectImage({
    execute,
    docker,
    repositoryRoot: input.repositoryRoot,
    tag: plan.tag,
    environment,
  });
  if (!built) {
    throw new Error(`Built Codex worker image cannot be inspected: ${plan.tag}`);
  }
  if (!labelsMatch(built.labels, plan.labels)) {
    throw new Error(`Built Codex worker image has an invalid identity: ${plan.tag}`);
  }
  return {
    ...plan,
    imageId: built.imageId,
    reused: false,
    dockerfilePath,
  };
}

function renderDockerfile(plan: CodexWorkerImagePlan): string {
  const rootInstall =
    "export DEBIAN_FRONTEND=noninteractive; " +
    "if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then " +
    "apk add --no-cache curl bash nodejs npm ripgrep; " +
    "elif command -v apt-get &>/dev/null; then " +
    "apt-get update && apt-get install -y curl ripgrep; " +
    "elif command -v yum &>/dev/null; then yum install -y curl ripgrep; " +
    "else echo 'Warning: No known package manager found, assuming curl is available' >&2; fi";
  const codexInstall = `set -euo pipefail; if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then npm install -g @openai/codex@${plan.codexVersion}; else curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | env -u NODE_VERSION bash && export NVM_DIR="$HOME/.nvm" && \\. "$NVM_DIR/nvm.sh" && command -v nvm &>/dev/null && nvm install 22 && nvm alias default 22 && npm -v && npm install -g @openai/codex@${plan.codexVersion}; fi && codex --version`;
  const symlink =
    "for bin in node codex; do " +
    'BIN_PATH="$(which "$bin" 2>/dev/null || true)"; ' +
    'if [ -n "$BIN_PATH" ] && [ "$BIN_PATH" != "/usr/local/bin/$bin" ]; then ' +
    'ln -sf "$BIN_PATH" "/usr/local/bin/$bin"; fi; done';
  const labels = Object.entries(plan.labels)
    .map(([name, value]) => `${name}=${JSON.stringify(value)}`)
    .join(" ");
  return [
    `FROM ${plan.baseImage}`,
    "USER root",
    `RUN [\"/bin/bash\", \"-c\", ${JSON.stringify(rootInstall)}]`,
    `RUN [\"/bin/bash\", \"-c\", ${JSON.stringify(codexInstall)}]`,
    `RUN [\"/bin/bash\", \"-c\", ${JSON.stringify(symlink)}]`,
    `LABEL ${labels}`,
    "",
  ].join("\n");
}

async function inspectImage(input: {
  readonly execute: WorkerImageCommand;
  readonly docker: string;
  readonly repositoryRoot: string;
  readonly tag: string;
  readonly environment: NodeJS.ProcessEnv;
}): Promise<{
  readonly imageId: string;
  readonly labels: Readonly<Record<string, string>>;
} | null> {
  const result = await input.execute(
    input.repositoryRoot,
    input.docker,
    ["image", "inspect", "--format", "{{json .}}", input.tag],
    input.environment,
  );
  if (result.exitCode !== 0) return null;
  try {
    const parsed = JSON.parse(result.stdout) as {
      readonly Id?: unknown;
      readonly Config?: { readonly Labels?: unknown };
    };
    if (typeof parsed.Id !== "string" || !parsed.Id.startsWith("sha256:")) return null;
    const rawLabels = parsed.Config?.Labels;
    const labels = isStringRecord(rawLabels) ? rawLabels : {};
    return { imageId: parsed.Id, labels };
  } catch {
    return null;
  }
}

function labelsMatch(
  actual: Readonly<Record<string, string>>,
  expected: Readonly<Record<string, string>>,
): boolean {
  return Object.entries(expected).every(([name, value]) => actual[name] === value);
}

function isStringRecord(value: unknown): value is Readonly<Record<string, string>> {
  return (
    typeof value === "object" &&
    value !== null &&
    Object.values(value).every((entry) => typeof entry === "string")
  );
}

function requiredDockerValue(value: string, name: string): string {
  const trimmed = value.trim();
  if (!trimmed || !/^[A-Za-z0-9][A-Za-z0-9._/@:+-]*$/.test(trimmed)) {
    throw new Error(`Invalid ${name}: ${JSON.stringify(value)}`);
  }
  return trimmed;
}

function diagnostic(result: WorkspaceCommandResult): string {
  return (result.stderr || result.stdout || `exit ${result.exitCode}`).trim();
}
