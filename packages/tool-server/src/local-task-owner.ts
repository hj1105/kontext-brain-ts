import { FileTaskContextRepository } from "@kontext-brain/local";
import { loadLocalKnowledgePrincipal } from "./local-knowledge-principal.js";

export async function requireLocalTaskOwner(dataDirectory: string, taskId: string) {
  const repository = new FileTaskContextRepository(dataDirectory);
  const registration = await repository.getInitialRegistration(taskId);
  const principal = await loadLocalKnowledgePrincipal(dataDirectory);
  if (
    !registration?.owner.contextSelection ||
    registration.owner.organizationId !== principal.organizationId ||
    registration.owner.subjectId !== principal.subjectId
  )
    throw new Error("Task operation requires a Task registered to the current local principal");
  return { repository, registration, principal };
}
