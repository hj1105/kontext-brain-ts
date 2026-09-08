import { execFile } from "node:child_process";
import { realpath } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import {
  FileLocalNormativeOverlayStore,
  assembleCurrentTaskContextState,
} from "@kontext-brain/local";
import { prepareCodingWorkspaceSeed } from "./local-coding-workspace-seed.js";
import { LocalKnowledgeOperations } from "./local-knowledge-operations.js";
import { loadLocalKnowledgePrincipal } from "./local-knowledge-principal.js";
import { subscriptionRuntimeEnvironment } from "./subscription-runtime-environment.js";

const execFileAsync = promisify(execFile);

type TaskWorkspaceRequest = { workspacePath: string; expectedCodeRevision?: string };

export function inspectTaskWorkspace(
  dataDirectory: string,
  request: TaskWorkspaceRequest,
  environment: NodeJS.ProcessEnv = process.env,
) {
  return resolveTaskWorkspace(dataDirectory, request, environment, false);
}

export function prepareTaskWorkspace(
  dataDirectory: string,
  request: TaskWorkspaceRequest,
  environment: NodeJS.ProcessEnv = process.env,
) {
  return resolveTaskWorkspace(dataDirectory, request, environment, true);
}

async function resolveTaskWorkspace(
  dataDirectory: string,
  request: TaskWorkspaceRequest,
  environment: NodeJS.ProcessEnv,
  allowSeed: boolean,
) {
  if (!path.isAbsolute(request.workspacePath)) throw new Error("Task workspace must be absolute");
  const workspacePath = await realpath(request.workspacePath);
  const git = async (args: string[]) =>
    (
      await execFileAsync("git", ["--no-optional-locks", "-c", "core.fsmonitor=false", ...args], {
        cwd: workspacePath,
        timeout: 15_000,
        maxBuffer: 1024 * 1024,
        windowsHide: true,
        env: {
          ...subscriptionRuntimeEnvironment(dataDirectory, environment),
          GIT_TERMINAL_PROMPT: "0",
        },
      })
    ).stdout.replace(/\r?\n$/, "");
  const seed = async () => {
    if (!allowSeed) throw new Error("Task workspace inspection requires a clean Git commit");
    const captured = await prepareCodingWorkspaceSeed(dataDirectory, workspacePath, environment);
    if (
      request.expectedCodeRevision !== undefined &&
      captured.codeRevision !== request.expectedCodeRevision
    )
      throw new Error("Reviewed code revision changed before Task creation");
    return {
      ...captured,
      workspaceSeed: {
        repositoryPath: captured.repositoryPath,
        codeRevision: captured.codeRevision,
      },
    };
  };
  const repositoryRoot = await git(["rev-parse", "--show-toplevel"]).catch(() => null);
  if (repositoryRoot === null) return seed();
  if ((await realpath(repositoryRoot)) !== workspacePath)
    throw new Error("Task creation requires the Git workspace root");
  const codeRevision = await git(["rev-parse", "--verify", "HEAD^{commit}"]).catch(() => null);
  if (codeRevision === null || (await git(["status", "--porcelain=v1", "--untracked-files=all"])))
    return seed();
  if (
    !/^[a-f0-9]{40,64}$/.test(codeRevision) ||
    (request.expectedCodeRevision !== undefined && codeRevision !== request.expectedCodeRevision)
  )
    throw new Error("Reviewed code revision changed before Task creation");
  return { workspacePath, codeRevision, repositoryPath: workspacePath, workspaceSeed: undefined };
}

export async function collectPersonalTaskContext(
  dataDirectory: string,
  input: {
    taskId: string;
    workspaceId: string;
    codeRevision: string;
    sourceResourceIds: readonly string[];
  },
) {
  const principal = await loadLocalKnowledgePrincipal(dataDirectory);
  const knowledge = new LocalKnowledgeOperations(dataDirectory);
  const references = [];
  for (const resourceId of [...new Set(input.sourceResourceIds)].sort()) {
    references.push(...(await knowledge.refreshSource({ resourceId })).evidence);
  }
  const localManifest = await new FileLocalNormativeOverlayStore(dataDirectory).load({
    organizationId: principal.organizationId,
    subjectId: principal.subjectId,
    workspaceId: input.workspaceId,
  });
  const state = assembleCurrentTaskContextState({
    taskId: input.taskId,
    organizationId: principal.organizationId,
    codeRevision: input.codeRevision,
    baseScopes: [
      { kind: "personal", subjectId: principal.subjectId },
      { kind: "workspace", workspaceId: input.workspaceId },
    ],
    localManifest,
    evidence: await knowledge.collectTaskEvidence(references),
    logicPlans: [],
  });
  return {
    principal,
    localManifest,
    state,
    sourceEvidenceIds: references.map((ref) => ref.evidenceId).sort(),
  };
}
