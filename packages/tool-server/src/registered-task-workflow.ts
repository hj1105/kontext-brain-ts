import {
  type PrepareTaskRequest,
  type TaskContextStateProvider,
  TaskContextWorkflow,
} from "@kontext-brain/context";
import type { FileTaskContextRepository } from "@kontext-brain/local";
import { taskContractDigest } from "@kontext-brain/spec";
import { loadLocalKnowledgePrincipal } from "./local-knowledge-principal.js";

/** Worker preparation may refresh context, but cannot rewrite a host-reviewed Task Contract. */
export class RegisteredTaskContextWorkflow extends TaskContextWorkflow {
  constructor(
    private readonly dataDirectory: string,
    private readonly repository: FileTaskContextRepository,
    currentState: TaskContextStateProvider,
  ) {
    super(currentState, repository);
  }
  override async prepareTask(request: PrepareTaskRequest) {
    const initial = await this.repository.getInitialRegistration(request.contract.taskId);
    if (initial) {
      const principal = await loadLocalKnowledgePrincipal(this.dataDirectory);
      if (
        initial.owner.organizationId !== principal.organizationId ||
        initial.owner.subjectId !== principal.subjectId
      )
        throw new Error("Task is unavailable to the current local principal");
      const approved = await this.repository.get(request.contract.taskId);
      if (
        !approved ||
        taskContractDigest(approved.contract) !== taskContractDigest(request.contract)
      )
        throw new Error(
          "Task Contract changes require explicit host approval, not worker preparation",
        );
    }
    return super.prepareTask(request);
  }
}
