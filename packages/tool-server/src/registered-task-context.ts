import type {
  ContextEvidenceItem,
  CurrentTaskContextState,
  TaskContextStateProvider,
} from "@kontext-brain/context";
import {
  FileLocalNormativeOverlayStore,
  FileTaskContextRepository,
  assembleCurrentTaskContextState,
} from "@kontext-brain/local";
import { LocalKnowledgeOperations } from "./local-knowledge-operations.js";
import { loadLocalKnowledgePrincipal } from "./local-knowledge-principal.js";

/** Refreshes only the source selection owned by the host; never changes the frozen Task snapshot. */
export class RegisteredTaskContextProvider implements TaskContextStateProvider {
  constructor(
    private readonly dataDirectory: string,
    private readonly repository: FileTaskContextRepository,
  ) {}

  async getCurrent(taskId: string): Promise<CurrentTaskContextState> {
    const registration = await this.repository.getInitialRegistration(taskId);
    const stored = await this.repository.getCurrent(taskId);
    if (!registration?.owner.contextSelection) return stored;
    const principal = await loadLocalKnowledgePrincipal(this.dataDirectory);
    if (
      registration.owner.organizationId !== principal.organizationId ||
      registration.owner.subjectId !== principal.subjectId
    )
      throw new Error("Task is unavailable to the current local principal");
    const { workspaceId, sourceResourceIds } = registration.owner.contextSelection;
    const knowledge = new LocalKnowledgeOperations(this.dataDirectory);
    const references = [];
    const unavailable: ContextEvidenceItem[] = [];
    for (const resourceId of sourceResourceIds) {
      try {
        const captured = await knowledge.refreshSource({ resourceId });
        references.push(...captured.evidence);
      } catch {
        const known = [
          ...new Set(
            [...stored.evidence, ...registration.state.evidence]
              .filter((item) => item.provenance?.resourceId === resourceId)
              .map((item) => item.evidenceId),
          ),
        ];
        if (!known.length) throw new Error("A required Task source is unavailable");
        unavailable.push(
          ...known.map((evidenceId) => ({
            evidenceId,
            text: "",
            availability: "unavailable" as const,
            allowedRuntimeProviders: [],
          })),
        );
      }
    }
    const localManifest = await new FileLocalNormativeOverlayStore(this.dataDirectory).load({
      organizationId: principal.organizationId,
      subjectId: principal.subjectId,
      workspaceId,
    });
    const evidence = [...(await knowledge.collectTaskEvidence(references)), ...unavailable];
    return {
      ...assembleCurrentTaskContextState({
        taskId,
        organizationId: principal.organizationId,
        codeRevision: stored.codeRevision,
        baseScopes: registration.state.effectiveScopes.filter(
          (scope) => scope.kind === "personal" || scope.kind === "workspace",
        ),
        localManifest,
        evidence,
        logicPlans: stored.logicPlans,
        governanceLinks: stored.governanceLinks,
      }),
      sourceEvidenceIds: evidence.map((item) => item.evidenceId).sort(),
    };
  }
}
