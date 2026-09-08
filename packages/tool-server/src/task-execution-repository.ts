import { realpath } from "node:fs/promises";
import path from "node:path";
import { FileTaskContextRepository } from "@kontext-brain/local";
import { verifyCodingWorkspaceSeed } from "./local-coding-workspace-seed.js";
import { requireLocalTaskOwner } from "./local-task-owner.js";

/** Public jobs retain the source workspace; only the owning host resolves private seed storage. */
export async function resolveTaskExecutionRepository(
  dataDirectory: string,
  taskId: string,
  requestedRepository: string,
): Promise<string> {
  const repository = new FileTaskContextRepository(dataDirectory);
  if (!(await repository.getInitialRegistration(taskId))) return path.resolve(requestedRepository);
  const { registration } = await requireLocalTaskOwner(dataDirectory, taskId);
  const { workspacePath, workspaceSeed } = registration.owner;
  const canonicalWorkspace = await realpath(workspacePath);
  if ((await realpath(requestedRepository)) !== canonicalWorkspace)
    throw new Error("Schedule repository does not match the registered Task workspace");
  if (!workspaceSeed) return canonicalWorkspace;
  if (workspaceSeed.codeRevision !== registration.state.codeRevision)
    throw new Error("Coding seed does not match the reviewed initial Task revision");
  return verifyCodingWorkspaceSeed(dataDirectory, {
    workspacePath: canonicalWorkspace,
    ...workspaceSeed,
  });
}
