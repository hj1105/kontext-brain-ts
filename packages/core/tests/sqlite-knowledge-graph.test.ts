import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, rm, stat } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, expect, it } from "vitest";
import {
  FileResourceContentStore,
  type KnowledgeGraphUnitOfWork,
  type ResourceRecord,
  type ResourceSnapshot,
  SqliteKnowledgeGraphRepository,
  SyncResourceUseCase,
} from "../src/index.js";

const directories: string[] = [];
const { DatabaseSync }: typeof import("node:sqlite") = createRequire(import.meta.url)(
  "node:sqlite",
);
afterEach(async () => {
  await Promise.all(directories.splice(0).map((dir) => rm(dir, { recursive: true })));
});
const resource: ResourceRecord = {
  organizationId: "org:one",
  resourceId: "resource:one",
  source: { connectorId: "markdown", externalId: "one.md", type: "markdown" },
  title: "0",
  contentHash: "hash:one",
  contentObjectKey: "object:one",
  acl: { subjectIds: ["user:one"] },
  ontologyNodeIds: ["domain:one"],
  status: "active",
  updatedAt: "2026-09-06T00:00:00.000Z",
};
async function fixture() {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-sqlite-graph-"));
  directories.push(directory);
  const graph = await SqliteKnowledgeGraphRepository.open(directory);
  return { directory, graph, filename: path.join(directory, "knowledge", "graph.sqlite") };
}
it("reopens committed records and ontology links without external infrastructure", async () => {
  const h = await fixture();
  await h.graph.transaction(resource.organizationId, (tx) => tx.saveResource(resource));
  const reopened = await SqliteKnowledgeGraphRepository.open(h.directory);
  expect(await reopened.getResource(resource.organizationId, resource.resourceId)).toEqual(
    resource,
  );
  expect(await reopened.getResourceBySource(resource.organizationId, resource.source)).toEqual(
    resource,
  );
  expect(await reopened.listResourcesByOntologyNode(resource.organizationId, "domain:one")).toEqual(
    [resource],
  );
});
it("rolls back a failed transaction and cannot mutate through an expired handle", async () => {
  const h = await fixture();
  let retained: KnowledgeGraphUnitOfWork | undefined;
  await expect(
    h.graph.transaction(resource.organizationId, async (tx) => {
      retained = tx;
      await tx.saveResource(resource);
      throw new Error("Fixture transaction failed");
    }),
  ).rejects.toThrow("Fixture transaction failed");
  expect(await h.graph.getResource(resource.organizationId, resource.resourceId)).toBeNull();
  if (!retained) throw new Error("Fixture handle missing");
  await expect(retained.saveResource(resource)).rejects.toThrow("closed");
});
it("serializes read-modify-write across independent instances without blocking the event loop", async () => {
  const h = await fixture();
  await h.graph.transaction(resource.organizationId, (tx) => tx.saveResource(resource));
  const instances = await Promise.all(
    Array.from({ length: 16 }, () => SqliteKnowledgeGraphRepository.open(h.directory)),
  );
  let callbacks = 0;
  await Promise.all(
    instances.map((graph) =>
      graph.transaction(resource.organizationId, async (tx) => {
        callbacks++;
        const current = await tx.getResource(resource.resourceId);
        if (!current) throw new Error("Fixture Resource missing");
        await new Promise((resolve) => setTimeout(resolve, 2));
        await tx.saveResource({ ...current, title: String(Number(current.title) + 1) });
      }),
    ),
  );
  expect(callbacks).toBe(16);
  expect((await h.graph.getResource(resource.organizationId, resource.resourceId))?.title).toBe(
    "16",
  );
});
it("keeps Organization namespaces separate and rejects cross-Organization writes", async () => {
  const h = await fixture();
  await h.graph.transaction(resource.organizationId, (tx) => tx.saveResource(resource));
  expect(await h.graph.getResource("org:other", resource.resourceId)).toBeNull();
  await expect(h.graph.transaction("org:other", (tx) => tx.saveResource(resource))).rejects.toThrow(
    "another Organization",
  );
  await h.graph.transaction("org:other", (tx) =>
    tx.saveResource({ ...resource, organizationId: "org:other", title: "Other" }),
  );
  expect((await h.graph.getResource(resource.organizationId, resource.resourceId))?.title).toBe(
    "0",
  );
});
it("retains collected chunks, entities, facts, evidence and event history across synchronization and reopen", async () => {
  const h = await fixture();
  const store = new FileResourceContentStore(path.join(h.directory, "content"));
  const sync = new SyncResourceUseCase(h.graph, store);
  const snapshot: ResourceSnapshot = {
    organizationId: resource.organizationId,
    source: resource.source,
    title: "Source",
    contentHash: "source:one",
    body: "Customer ordered one item",
    acl: resource.acl,
    ontologyNodeIds: ["domain:one"],
    chunks: [
      {
        id: "paragraph:one",
        contentHash: "chunk:one",
        text: "Customer ordered one item",
        position: 0,
      },
    ],
    entities: [
      {
        entityId: "customer:one",
        scope: "resource",
        name: "Customer",
        mentionChunkIds: ["paragraph:one"],
      },
    ],
    facts: [
      {
        factKey: "fact:one",
        subject: { entityId: "customer:one", scope: "resource" },
        predicate: "ordered",
        object: { kind: "literal", value: 1 },
        evidenceChunkIds: ["paragraph:one"],
      },
    ],
  };
  const { resourceId } = await sync.execute(snapshot);
  const reopened = await SqliteKnowledgeGraphRepository.open(h.directory);
  expect(await reopened.listChunks(resource.organizationId, resourceId)).toHaveLength(1);
  expect(await reopened.listEntitiesForResource(resource.organizationId, resourceId)).toHaveLength(
    1,
  );
  expect(await reopened.listEntityMentions(resource.organizationId, resourceId)).toHaveLength(1);
  expect(await reopened.listEvidenceForFact(resource.organizationId, "fact:one")).toHaveLength(1);
  expect((await reopened.getFact(resource.organizationId, "fact:one"))?.status).toBe("active");
  await sync.remove(resource.organizationId, resource.source);
  expect((await reopened.getFact(resource.organizationId, "fact:one"))?.status).toBe("inactive");
  expect(
    (await reopened.listFactEvents(resource.organizationId, "fact:one")).map((event) => event.type),
  ).toEqual(["created", "invalidated"]);
});
it("does not replay a callback after caller failure", async () => {
  const h = await fixture();
  let calls = 0;
  await expect(
    h.graph.transaction(resource.organizationId, async () => {
      calls++;
      throw new Error("Not retryable");
    }),
  ).rejects.toThrow("Not retryable");
  expect(calls).toBe(1);
});
it("rejects future database versions without modifying the stored version", async () => {
  const h = await fixture();
  const db = new DatabaseSync(h.filename);
  db.exec("PRAGMA user_version = 99");
  db.close();
  await expect(SqliteKnowledgeGraphRepository.open(h.directory)).rejects.toThrow("schema version");
  await expect(h.graph.getResource(resource.organizationId, resource.resourceId)).rejects.toThrow(
    "schema version",
  );
  const check = new DatabaseSync(h.filename);
  try {
    expect(check.prepare("PRAGMA user_version").get()?.user_version).toBe(99);
  } finally {
    check.close();
  }
});
it("shows only committed state to readers while another asynchronous transaction is open", async () => {
  const h = await fixture();
  await h.graph.transaction(resource.organizationId, (tx) => tx.saveResource(resource));
  let release = () => {};
  let ready = () => {};
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const opened = new Promise<void>((resolve) => {
    ready = resolve;
  });
  const pending = h.graph.transaction(resource.organizationId, async (tx) => {
    await tx.saveResource({ ...resource, title: "Pending" });
    ready();
    await gate;
  });
  await opened;
  try {
    expect((await h.graph.getResource(resource.organizationId, resource.resourceId))?.title).toBe(
      "0",
    );
  } finally {
    release();
    await pending;
  }
  expect((await h.graph.getResource(resource.organizationId, resource.resourceId))?.title).toBe(
    "Pending",
  );
});
it("rejects malformed persisted ACLs instead of widening access", async () => {
  const h = await fixture();
  await h.graph.transaction(resource.organizationId, (tx) => tx.saveResource(resource));
  const db = new DatabaseSync(h.filename);
  db.prepare("UPDATE knowledge_records SET payload = ?").run(
    JSON.stringify({ ...resource, acl: { organizationWide: "true" } }),
  );
  db.close();
  await expect(h.graph.getResource(resource.organizationId, resource.resourceId)).rejects.toThrow();
});
it.skipIf(process.platform === "win32")(
  "creates private database and directory permissions",
  async () => {
    const h = await fixture();
    expect((await stat(h.filename)).mode & 0o777).toBe(0o600);
    expect((await stat(path.dirname(h.filename))).mode & 0o777).toBe(0o700);
  },
);
it("recovers committed state after a separate writer process is killed mid-transaction", async () => {
  const h = await fixture();
  await h.graph.transaction(resource.organizationId, (tx) => tx.saveResource(resource));
  const child = spawn(
    process.execPath,
    [
      "--input-type=module",
      "-e",
      `
    import { DatabaseSync } from 'node:sqlite';
    const db = new DatabaseSync(process.argv[1]);
    db.exec('BEGIN IMMEDIATE');
    db.prepare('DELETE FROM knowledge_records WHERE organization_id = ?').run('org:one');
    process.send('transaction-open');
    setInterval(() => {}, 1000);
  `,
      h.filename,
    ],
    { env: { HOME: h.directory }, stdio: ["ignore", "ignore", "ignore", "ipc"] },
  );
  const exited = once(child, "exit");
  try {
    await Promise.race([
      once(child, "message"),
      exited.then(() => {
        throw new Error("Fixture writer exited before opening its transaction");
      }),
    ]);
    child.kill("SIGKILL");
    await exited;
    const reopened = await SqliteKnowledgeGraphRepository.open(h.directory);
    expect(await reopened.getResource(resource.organizationId, resource.resourceId)).toEqual(
      resource,
    );
    await reopened.transaction(resource.organizationId, (tx) =>
      tx.saveResource({ ...resource, title: "After recovery" }),
    );
    expect((await h.graph.getResource(resource.organizationId, resource.resourceId))?.title).toBe(
      "After recovery",
    );
  } finally {
    if (child.exitCode === null && child.signalCode === null) {
      child.kill("SIGKILL");
      await exited;
    }
  }
});
