import { createHash, randomUUID } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { type Server, createServer } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import { FileResourceContentStore, SqliteKnowledgeGraphRepository } from "@kontext-brain/core";
import { afterEach, expect, it } from "vitest";
import { LocalKnowledgeOperations } from "../src/local-knowledge-operations.js";
import { kontextSessionSourcePayloadSchema } from "../src/native-session-source-contract.js";
import { readNativeSessionSource } from "../src/native-session-source-reader.js";

const roots: string[] = [];
const servers: Server[] = [];
async function close(server: Server) {
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
}
afterEach(async () => {
  await Promise.all(servers.splice(0).map(close));
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true })));
});
function source(text = "Use the original domain term", runtimeId = "runtime-one") {
  const payload = kontextSessionSourcePayloadSchema.parse({
    schemaVersion: 1,
    origin: {
      kind: "kondex_session",
      runtimeId,
      executionHostId: "local",
      workspaceId: "folder-one",
      workspaceKind: "folder",
      wslDistro: null,
      sessionId: "session-one",
      provider: "codex",
    },
    journalCursor: { epoch: "epoch-one", sequence: 1 },
    scope: "journal_user_assistant_text",
    messages: [
      {
        itemId: "item-one",
        revision: 1,
        sequence: 1,
        observedAt: 1,
        recovered: false,
        role: "user",
        blocks: [{ index: 0, text }],
      },
    ],
    excluded: { items: 0, blocks: 0, unconfirmedSubmissions: 0 },
    providerSharing: "not_granted",
    normativeApproval: "not_granted",
  });
  return {
    ...payload,
    registration: "not_registered" as const,
    contentDigest: `sha256:${createHash("sha256").update(JSON.stringify(payload)).digest("hex")}`,
  };
}
async function fixture() {
  const data = await mkdtemp(path.join(tmpdir(), "kontext-native-source-"));
  roots.push(data);
  await mkdir(path.join(data, "knowledge"));
  let current = source();
  let transform: (result: { requestId: string; source: ReturnType<typeof source> }) => unknown = (
    result,
  ) => result;
  let beforeResponse = async () => {};
  let requests = 0;
  const token = "b".repeat(64);
  const server = createServer((request, response) => {
    void (async () => {
      requests++;
      if (request.headers.authorization !== `Bearer ${token}`) {
        response.writeHead(403).end();
        return;
      }
      const chunks: Buffer[] = [];
      for await (const chunk of request) chunks.push(chunk);
      const input = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      await beforeResponse();
      response
        .writeHead(200)
        .end(JSON.stringify(transform({ requestId: input.requestId, source: current })));
    })().catch(() => response.destroy());
  });
  servers.push(server);
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Missing fixture address");
  const file = path.join(data, "knowledge", "native-source-reader.json");
  const descriptor = {
    schemaVersion: 1,
    runtimeId: current.origin.runtimeId,
    instanceId: randomUUID(),
    endpoint: `http://127.0.0.1:${address.port}/v1/session`,
    token,
  };
  await writeFile(file, JSON.stringify(descriptor));
  return {
    data,
    file,
    descriptor,
    server,
    operations: new LocalKnowledgeOperations(data),
    get current() {
      return current;
    },
    get requests() {
      return requests;
    },
    setSource(value: ReturnType<typeof source>) {
      current = value;
    },
    transform(value: typeof transform) {
      transform = value;
    },
    beforeResponse(value: typeof beforeResponse) {
      beforeResponse = value;
    },
    register: () =>
      new LocalKnowledgeOperations(data).registerSessionSource({
        origin: current.origin,
        expectedContentDigest: current.contentDigest,
      }),
  };
}

it("captures exact native text as owned Evidence without normative facts or implicit sharing", async () => {
  const h = await fixture();
  const result = await h.register();
  expect(result).toMatchObject({
    changed: true,
    providerSharing: "not_granted",
    normativeApproval: "not_granted",
  });
  expect(result.evidence).toHaveLength(1);
  expect(JSON.stringify(result)).not.toContain("original domain term");
  const graph = await SqliteKnowledgeGraphRepository.open(h.data);
  const resource = await graph.getResource(result.organizationId, result.resourceId);
  if (!resource) throw new Error("Missing resource");
  expect(resource.source.connectorId).toBe("kondex-session");
  const stored = await new FileResourceContentStore(path.join(h.data, "knowledge-content")).get(
    resource.contentObjectKey,
  );
  expect(stored?.body).toContain("Use the original domain term");
  expect(stored?.body).toContain('"sessionId":"session-one"');
  expect(await graph.listFacts(result.organizationId)).toEqual([]);
  expect((await h.operations.collectTaskEvidence(result.evidence))[0]).toMatchObject({
    availability: "current",
    allowedRuntimeProviders: [],
  });
  expect((await h.operations.listSources({})).sources).toEqual([]);
  expect((await h.operations.listSources({ includeNativeSessions: true })).sources).toMatchObject([
    { sourceKind: "native_session", nativeSession: h.current.origin, sharing: null },
  ]);
  expect(await h.register()).toEqual({ ...result, changed: false });
});

it("recaptures live revisions, revokes grants on change, and preserves IDs across runtime replacement", async () => {
  const h = await fixture();
  const registered = await h.register();
  await h.operations.setSourceSharing({
    resourceId: registered.resourceId,
    expectedRevision: 1,
    expectedContentHash: registered.contentHash,
    dataClassification: "internal",
    allowedRuntimeProviders: ["codex"],
  });
  expect(
    (await h.operations.collectTaskEvidence(registered.evidence))[0]?.allowedRuntimeProviders,
  ).toEqual(["codex"]);
  h.setSource(source("Updated decision", "runtime-two"));
  await writeFile(
    h.file,
    JSON.stringify({ ...h.descriptor, runtimeId: "runtime-two", instanceId: randomUUID() }),
  );
  const refreshed = await new LocalKnowledgeOperations(h.data).refreshSource(registered);
  expect(refreshed.resourceId).toBe(registered.resourceId);
  expect(refreshed.contentHash).not.toBe(registered.contentHash);
  expect((await h.operations.inspectSource(registered)).sharing).toBeNull();
  expect((await h.operations.collectTaskEvidence(refreshed.evidence))[0]).toMatchObject({
    text: expect.stringContaining("Updated decision"),
    allowedRuntimeProviders: [],
  });
  h.transform(() => {
    throw new Error("Disconnected fixture");
  });
  await expect(h.operations.refreshSource(registered)).rejects.toThrow();
  expect((await h.operations.inspectSource(registered)).status).toBe("stale");
  expect((await h.operations.collectTaskEvidence(refreshed.evidence))[0]).toMatchObject({
    text: "",
    allowedRuntimeProviders: [],
  });
  h.transform((value) => value);
  await h.operations.refreshSource(registered);
  expect((await h.operations.inspectSource(registered)).status).toBe("active");
  await close(h.server);
  await expect(h.operations.refreshSource(registered)).rejects.toThrow();
  expect((await h.operations.inspectSource(registered)).status).toBe("stale");
});

it("requires the exact reviewed digest and refuses body, origin, request-ID or descriptor substitution", async () => {
  const h = await fixture();
  await expect(
    h.operations.registerSessionSource({
      origin: h.current.origin,
      expectedContentDigest: `sha256:${"0".repeat(64)}`,
    }),
  ).rejects.toThrow("preview");
  expect((await h.operations.listSources({ includeNativeSessions: true })).sources).toEqual([]);
  for (const field of ["workspaceId", "provider", "executionHostId", "runtimeId"] as const) {
    h.transform((result) => ({
      ...result,
      source: {
        ...result.source,
        origin: { ...result.source.origin, [field]: field === "provider" ? "claude" : "wrong" },
      },
    }));
    await expect(readNativeSessionSource(h.data, h.current.origin)).rejects.toThrow();
  }
  h.transform((result) => ({ ...result, requestId: randomUUID() }));
  await expect(readNativeSessionSource(h.data, h.current.origin)).rejects.toThrow();
  h.transform((result) => ({
    ...result,
    source: { ...result.source, contentDigest: `sha256:${"0".repeat(64)}` },
  }));
  await expect(readNativeSessionSource(h.data, h.current.origin)).rejects.toThrow("digest");
  h.transform((result) => result);
  h.beforeResponse(async () => {
    await writeFile(h.file, JSON.stringify({ ...h.descriptor, instanceId: randomUUID() }));
  });
  await expect(readNativeSessionSource(h.data, h.current.origin)).rejects.toThrow(
    "changed during capture",
  );
});

it("rejects remote endpoints, invalid credentials, oversized configuration and oversized responses", async () => {
  const h = await fixture();
  await writeFile(
    h.file,
    JSON.stringify({ ...h.descriptor, endpoint: "https://example.com/v1/session" }),
  );
  await expect(readNativeSessionSource(h.data, h.current.origin)).rejects.toThrow();
  expect(h.requests).toBe(0);
  await writeFile(h.file, " ".repeat(16 * 1024 + 1));
  await expect(readNativeSessionSource(h.data, h.current.origin)).rejects.toThrow();
  await writeFile(h.file, JSON.stringify({ ...h.descriptor, token: "c".repeat(64) }));
  await expect(readNativeSessionSource(h.data, h.current.origin)).rejects.toThrow();
  await writeFile(h.file, JSON.stringify(h.descriptor));
  h.transform(() => ({ body: "a".repeat(601 * 1024) }));
  await expect(readNativeSessionSource(h.data, h.current.origin)).rejects.toThrow();
  expect(JSON.parse(await readFile(h.file, "utf8"))).toEqual(h.descriptor);
});
