import { createHash } from "node:crypto";
import { DefaultAccessPolicy, SqliteKnowledgeGraphRepository } from "@kontext-brain/core";
import type { z } from "zod";
import type { LocalKnowledgeOperations } from "./local-knowledge-operations.js";
import { loadLocalKnowledgePrincipal } from "./local-knowledge-principal.js";
import { LocalSourceRegistry, sourceInventoryRequestSchema } from "./local-source-registry.js";

/** Caller holds the source-management lock; this returns saved metadata, never a live file check. */
export async function listLocalSources(
  directory: string,
  operations: Pick<LocalKnowledgeOperations, "inspectSource">,
  input: z.input<typeof sourceInventoryRequestSchema>,
) {
  const request = sourceInventoryRequestSchema.parse(input);
  const principal = await loadLocalKnowledgePrincipal(directory);
  const records = await new LocalSourceRegistry(directory).list(principal);
  const graph = await SqliteKnowledgeGraphRepository.open(directory);
  const sources: Awaited<ReturnType<LocalKnowledgeOperations["inspectSource"]>>[] = [];
  for (const resourceId of records) {
    const registration = await new LocalSourceRegistry(directory).get(principal, resourceId);
    if (registration?.schemaVersion === 2 && !request.includeNativeSessions) continue;
    const resource = await graph.getResource(principal.organizationId, resourceId);
    if (
      !resource ||
      resource.status === "purged" ||
      !new DefaultAccessPolicy().canAccess(principal, resource.acl)
    )
      continue;
    sources.push(await operations.inspectSource({ resourceId }));
  }
  const currentPrincipal = await loadLocalKnowledgePrincipal(directory);
  if (JSON.stringify(currentPrincipal) !== JSON.stringify(principal))
    throw new Error("Source inventory owner changed");
  const digest = `sha256:${createHash("sha256")
    .update(JSON.stringify([principal, request.includeNativeSessions, sources]))
    .digest("hex")}`;
  if (
    request.cursor &&
    (request.cursor.digest !== digest || request.cursor.offset >= sources.length)
  )
    throw new Error("Source inventory changed; reload its first page");
  const offset = request.cursor?.offset ?? 0;
  const next = offset + request.limit;
  return {
    organizationId: principal.organizationId,
    sources: sources.slice(offset, next),
    inventoryDigest: digest,
    nextCursor: next < sources.length ? { digest, offset: next } : null,
    observation: "saved_metadata_only" as const,
    nativeSessionsIncluded: request.includeNativeSessions,
  };
}
