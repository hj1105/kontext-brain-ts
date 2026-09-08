import { mkdir } from "node:fs/promises";
import path from "node:path";
import {
  DefaultAccessPolicy,
  FileResourceContentStore,
  type Principal,
  type ResourceSnapshot,
  SqliteKnowledgeGraphRepository,
  SyncResourceUseCase,
} from "@kontext-brain/core";
import { withLocalFileMutationLock } from "@kontext-brain/local";
import type { z } from "zod";
import {
  type CollectedEvidenceMetadata,
  type CollectedEvidenceRef,
  CollectedTaskEvidence,
} from "./collected-task-evidence.js";
import { loadLocalKnowledgePrincipal } from "./local-knowledge-principal.js";
import { captureLocalMarkdownSource, resolveLocalMarkdownSource } from "./local-markdown-source.js";
import { listLocalSources } from "./local-source-inventory.js";
import {
  type LocalSourceLocator,
  type LocalSourceRecord,
  LocalSourceRegistry,
  type SourceSharingRequest,
  sourceConnector,
  type sourceInventoryRequestSchema,
  sourceLocator,
  sourceSharingRequestSchema,
} from "./local-source-registry.js";
import {
  nativeSessionResourceSnapshot,
  nativeSessionSourceReference,
} from "./native-session-resource-snapshot.js";
import {
  type NativeSessionOrigin,
  nativeSessionRegistrationSchema,
  readNativeSessionSource,
} from "./native-session-source-reader.js";

export class LocalKnowledgeOperations {
  constructor(private readonly dataDirectory: string) {}
  async listSources(request: z.input<typeof sourceInventoryRequestSchema>) {
    return this.mutate(() => listLocalSources(this.dataDirectory, this, request));
  }
  async registerMarkdownSource(request: { workspacePath: string; relativePath: string }) {
    return this.mutate(() => this.register(request));
  }
  async registerSessionSource(input: z.input<typeof nativeSessionRegistrationSchema>) {
    const request = nativeSessionRegistrationSchema.parse(input);
    return this.mutate(() =>
      this.registerNativeSession(request.origin, request.expectedContentDigest),
    );
  }

  async inspectSource(request: { resourceId: string }) {
    const principal = await loadLocalKnowledgePrincipal(this.dataDirectory);
    const { record, resource } = await this.ownedSource(principal, request.resourceId);
    return {
      organizationId: principal.organizationId,
      resourceId: record.resourceId,
      title: resource.title,
      ...(record.schemaVersion === 1
        ? { workspacePath: record.workspacePath, relativePath: record.relativePath }
        : { nativeSession: record.nativeSession, sourceKind: "native_session" as const }),
      contentHash: record.contentHash,
      revision: record.revision,
      status: resource.contentHash === record.contentHash ? resource.status : "stale",
      sharing: record.sharing,
      // This is configuration metadata; collectors also check current ACL/status/hash.
      normativeApproval: "not_granted" as const,
    };
  }

  async refreshSource(request: { resourceId: string }) {
    return this.mutate(async () => {
      const principal = await loadLocalKnowledgePrincipal(this.dataDirectory);
      const { record } = await this.ownedSource(principal, request.resourceId);
      try {
        if (record.schemaVersion === 2)
          return await this.registerNativeSession(record.nativeSession);
        const selection = await resolveLocalMarkdownSource(
          record.workspacePath,
          record.relativePath,
        );
        if (selection.source.externalId !== record.sourceExternalId)
          throw new Error("Registered workspace identity changed");
        return await this.register(record, record.sourceExternalId);
      } catch (cause) {
        // The saved Resource ID still works when the entire workspace has disappeared.
        const graph = await SqliteKnowledgeGraphRepository.open(this.dataDirectory);
        await graph.transaction(principal.organizationId, async (tx) => {
          const resource = await tx.getResource(record.resourceId);
          if (resource && resource.status !== "purged")
            await tx.saveResource({ ...resource, status: "stale" });
        });
        throw cause;
      }
    });
  }

  async setSourceSharing(input: SourceSharingRequest) {
    const request = sourceSharingRequestSchema.parse(input);
    return this.mutate(async () => {
      const principal = await loadLocalKnowledgePrincipal(this.dataDirectory);
      const { record, resource } = await this.ownedSource(principal, request.resourceId);
      if (
        record.revision !== request.expectedRevision ||
        record.contentHash !== request.expectedContentHash
      )
        throw new Error(
          "Source version conflict; inspect the current source before changing sharing",
        );
      if (
        request.allowedRuntimeProviders.length > 0 &&
        (resource.status !== "active" || resource.contentHash !== record.contentHash)
      )
        throw new Error("Refresh the source before granting provider sharing");
      const sharing = {
        dataClassification: request.dataClassification,
        allowedRuntimeProviders: [...new Set(request.allowedRuntimeProviders)].sort(),
      };
      const revision = record.revision + 1;
      await new LocalSourceRegistry(this.dataDirectory).save({
        ...record,
        revision,
        sharing,
        audit: [
          ...record.audit,
          {
            revision,
            contentHash: record.contentHash,
            actor: principal.subjectId,
            at: new Date().toISOString(),
            reason: "sharing_changed",
            sharing,
          },
        ],
      });
      return this.inspectSource(request);
    });
  }

  /** Trusted Task creation uses the host principal and this policy, never a worker-authored allowlist. */
  async collectTaskEvidence(references: readonly CollectedEvidenceRef[]) {
    const principal = await loadLocalKnowledgePrincipal(this.dataDirectory);
    const graph = await SqliteKnowledgeGraphRepository.open(this.dataDirectory);
    return new CollectedTaskEvidence(
      graph,
      new FileResourceContentStore(path.join(this.dataDirectory, "knowledge-content")),
      (identity, metadata) => this.allowedProviders(identity, metadata),
    ).collect(principal, references);
  }

  private async allowedProviders(principal: Principal, metadata: CollectedEvidenceMetadata) {
    const record = await new LocalSourceRegistry(this.dataDirectory).get(
      principal,
      metadata.resource.resourceId,
    );
    if (
      !record ||
      metadata.resource.source.connectorId !== sourceConnector(record) ||
      metadata.resource.source.externalId !== record.sourceExternalId ||
      metadata.resource.contentHash !== record.contentHash
    )
      return [];
    return record.sharing?.allowedRuntimeProviders ?? [];
  }

  private async ownedSource(principal: Principal, resourceId: string) {
    const graph = await SqliteKnowledgeGraphRepository.open(this.dataDirectory);
    const resource = await graph.getResource(principal.organizationId, resourceId);
    if (
      !resource ||
      resource.status === "purged" ||
      !new DefaultAccessPolicy().canAccess(principal, resource.acl)
    )
      throw new Error("Source registration unavailable");
    const record = await new LocalSourceRegistry(this.dataDirectory).get(principal, resourceId);
    if (
      !record ||
      resource.source.connectorId !== sourceConnector(record) ||
      resource.source.externalId !== record.sourceExternalId
    )
      throw new Error("Source registration unavailable");
    if (
      JSON.stringify(resource) !==
      JSON.stringify(await graph.getResource(principal.organizationId, resourceId))
    )
      throw new Error("Source registration changed during inspection");
    return { record, resource };
  }

  private mutate<T>(operation: () => Promise<T>): Promise<T> {
    return withLocalFileMutationLock(
      path.join(this.dataDirectory, "knowledge", "source-management.lock"),
      operation,
    );
  }

  private async register(
    request: { workspacePath: string; relativePath: string },
    expectedSourceExternalId?: string,
  ) {
    const principal = await loadLocalKnowledgePrincipal(this.dataDirectory);
    const selection = await resolveLocalMarkdownSource(request.workspacePath, request.relativePath);
    if (
      expectedSourceExternalId !== undefined &&
      selection.source.externalId !== expectedSourceExternalId
    )
      throw new Error("Registered workspace identity changed");
    const graph = await SqliteKnowledgeGraphRepository.open(this.dataDirectory);
    let snapshot: Awaited<ReturnType<typeof captureLocalMarkdownSource>>;
    try {
      snapshot = await captureLocalMarkdownSource(
        principal,
        request.workspacePath,
        request.relativePath,
      );
      if (snapshot.source.externalId !== selection.source.externalId)
        throw new Error("Selected workspace changed during capture");
    } catch (cause) {
      await graph.transaction(principal.organizationId, async (tx) => {
        const previous = await tx.getResourceBySource(selection.source);
        if (previous) await tx.saveResource({ ...previous, status: "stale" });
      });
      throw cause;
    }
    return this.publishCapturedSource(principal, snapshot, {
      schemaVersion: 1,
      workspacePath: selection.root,
      relativePath: selection.segments.join("/"),
    });
  }

  private async registerNativeSession(origin: NativeSessionOrigin, expectedDigest?: string) {
    const principal = await loadLocalKnowledgePrincipal(this.dataDirectory);
    try {
      const captured = await readNativeSessionSource(this.dataDirectory, origin);
      if (expectedDigest !== undefined && expectedDigest !== captured.contentDigest)
        throw new Error("Reviewed native session source changed; preview it again");
      return await this.publishCapturedSource(
        principal,
        nativeSessionResourceSnapshot(principal, captured),
        { schemaVersion: 2, nativeSession: captured.origin },
      );
    } catch (cause) {
      const graph = await SqliteKnowledgeGraphRepository.open(this.dataDirectory);
      await graph.transaction(principal.organizationId, async (tx) => {
        const previous = await tx.getResourceBySource(nativeSessionSourceReference(origin));
        if (previous && previous.status !== "purged")
          await tx.saveResource({ ...previous, status: "stale" });
      });
      throw cause;
    }
  }

  private async publishCapturedSource(
    principal: Principal,
    snapshot: ResourceSnapshot,
    locator: LocalSourceLocator,
  ) {
    const graph = await SqliteKnowledgeGraphRepository.open(this.dataDirectory);
    const existing = await graph.transaction(principal.organizationId, (tx) =>
      tx.getResourceBySource(snapshot.source),
    );
    // Revoke the old version before graph publication, including interrupted two-store writes.
    if (existing)
      await this.recordCapture(
        principal,
        existing.resourceId,
        locator,
        snapshot.source.externalId,
        snapshot.contentHash,
      );
    const contentDirectory = path.join(this.dataDirectory, "knowledge-content");
    await mkdir(contentDirectory, { recursive: true, mode: 0o700 });
    const sync = new SyncResourceUseCase(graph, new FileResourceContentStore(contentDirectory));
    const result = await sync.execute(snapshot);
    const evidence = await graph.transaction(principal.organizationId, async (tx) => {
      const resource = await tx.getResource(result.resourceId);
      if (!resource || resource.contentHash !== snapshot.contentHash)
        throw new Error("Source changed before registration was confirmed");
      return (await tx.listEvidenceForResource(result.resourceId))
        .filter((item) => item.status === "active")
        .map((item) => ({
          resourceId: item.resourceId,
          evidenceId: item.evidenceId,
          chunkId: item.chunkId,
        }));
    });
    await this.recordCapture(
      principal,
      result.resourceId,
      locator,
      snapshot.source.externalId,
      snapshot.contentHash,
    );
    return {
      organizationId: principal.organizationId,
      resourceId: result.resourceId,
      title: snapshot.title,
      contentHash: snapshot.contentHash,
      changed: result.changed,
      evidence,
      providerSharing: "not_granted" as const,
      normativeApproval: "not_granted" as const,
    };
  }

  private async recordCapture(
    principal: Principal,
    resourceId: string,
    locator: LocalSourceLocator,
    sourceExternalId: string,
    contentHash: string,
  ) {
    const registry = new LocalSourceRegistry(this.dataDirectory);
    const previous = await registry.get(principal, resourceId);
    if (
      previous &&
      (previous.sourceExternalId !== sourceExternalId ||
        previous.schemaVersion !== locator.schemaVersion ||
        (locator.schemaVersion === 1 &&
          JSON.stringify(sourceLocator(previous)) !== JSON.stringify(locator)))
    )
      throw new Error("Source registration locator mismatch");
    if (!previous || previous.contentHash !== contentHash) {
      const revision = (previous?.revision ?? 0) + 1;
      const record: LocalSourceRecord = {
        ...locator,
        organizationId: principal.organizationId,
        subjectId: principal.subjectId,
        resourceId,
        sourceExternalId,
        contentHash,
        revision,
        sharing: null,
        audit: [
          ...(previous?.audit ?? []),
          {
            revision,
            contentHash,
            actor: principal.subjectId,
            at: new Date().toISOString(),
            reason: previous ? "source_changed" : "captured",
            sharing: null,
          },
        ],
      };
      await registry.save(record);
    }
  }
}
