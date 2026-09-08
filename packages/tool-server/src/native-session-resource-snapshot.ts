import { createHash } from "node:crypto";
import type { Principal, ResourceSnapshot } from "@kontext-brain/core";
import type { KontextSessionSourcePreview } from "./native-session-source-contract.js";
import {
  type NativeSessionOrigin,
  nativeSessionOriginSchema,
} from "./native-session-source-reader.js";

const hash = (value: string) => `sha256:${createHash("sha256").update(value).digest("hex")}`;
export function nativeSessionSourceReference(origin: NativeSessionOrigin) {
  const { runtimeId: _runtimeId, ...persistentOrigin } = nativeSessionOriginSchema.parse(origin);
  return {
    connectorId: "kondex-session",
    type: "session",
    externalId: hash(JSON.stringify(persistentOrigin)),
  };
}
export function nativeSessionResourceSnapshot(
  principal: Principal,
  captured: KontextSessionSourcePreview,
): ResourceSnapshot {
  const chunks = captured.messages.map((message, position) => {
    const text = [
      `## ${message.role} · ${message.itemId}`,
      `Journal revision: ${message.revision}; sequence: ${message.sequence}; observedAt: ${message.observedAt}; recovered: ${message.recovered}`,
      ...message.blocks.map((block) => `Text block ${block.index}\n\n${block.text}`),
    ].join("\n\n");
    return { id: `message:${hash(message.itemId)}`, position, text, contentHash: hash(text) };
  });
  return {
    organizationId: principal.organizationId,
    source: nativeSessionSourceReference(captured.origin),
    title: `Session ${captured.origin.sessionId}`,
    body: [
      JSON.stringify({
        origin: captured.origin,
        journalCursor: captured.journalCursor,
        scope: captured.scope,
        excluded: captured.excluded,
      }),
      ...chunks.map((chunk) => chunk.text),
    ].join("\n\n"),
    contentHash: captured.contentDigest,
    acl: { subjectIds: [principal.subjectId] },
    chunks,
    entities: [],
    facts: [],
    ontologyNodeIds: [],
  };
}
