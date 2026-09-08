import { mkdir, open } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { setTimeout as delay } from "node:timers/promises";
import type { ResourceSource } from "./domain.js";
import type { KnowledgeGraphRepository, KnowledgeGraphUnitOfWork } from "./ports.js";
import { SqliteKnowledgeUnit } from "./sqlite-knowledge-unit.js";

/** Local durable graph; SQLite owns atomic commit, process locks and crash rollback. */
export class SqliteKnowledgeGraphRepository implements KnowledgeGraphRepository {
  private constructor(
    private readonly filename: string,
    private readonly Database: typeof DatabaseSync,
  ) {}

  static async open(dataDirectory: string): Promise<SqliteKnowledgeGraphRepository> {
    let sqlite: typeof import("node:sqlite");
    try {
      sqlite = createRequire(import.meta.url)("node:sqlite");
    } catch (cause) {
      throw new Error(
        "The local knowledge graph requires Node.js 22.13 or newer with node:sqlite enabled",
        { cause },
      );
    }
    const directory = path.resolve(dataDirectory, "knowledge");
    await mkdir(directory, { recursive: true, mode: 0o700 });
    const filename = path.join(directory, "graph.sqlite");
    try {
      const file = await open(filename, "wx", 0o600);
      await file.close();
    } catch (error) {
      if (!(error instanceof Error && "code" in error && error.code === "EEXIST")) throw error;
    }
    const repository = new SqliteKnowledgeGraphRepository(filename, sqlite.DatabaseSync);
    await repository.initialize();
    return repository;
  }

  private connection(initializing = false): DatabaseSync {
    const db = new this.Database(this.filename, { allowExtension: false });
    try {
      db.exec("PRAGMA busy_timeout = 0; PRAGMA synchronous = FULL;");
      if (!initializing) requireCurrentSchema(db);
      return db;
    } catch (error) {
      db.close();
      throw error;
    }
  }

  private async initialize(): Promise<void> {
    const db = this.connection(true);
    try {
      await retryBusy(() => db.exec("BEGIN IMMEDIATE"));
      const version = db.prepare("PRAGMA user_version").get()?.user_version;
      if (version === 0) {
        if (
          db
            .prepare(
              "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            )
            .get()
        )
          throw new Error("Refusing to initialize an unrelated SQLite database");
        db.exec(`
          CREATE TABLE knowledge_records (
            organization_id TEXT NOT NULL, kind TEXT NOT NULL, record_key TEXT NOT NULL,
            resource_id TEXT, fact_key TEXT, payload TEXT NOT NULL CHECK(json_valid(payload)),
            PRIMARY KEY (organization_id, kind, record_key)
          ) STRICT;
          CREATE INDEX knowledge_resource ON knowledge_records (organization_id, kind, resource_id);
          CREATE INDEX knowledge_fact ON knowledge_records (organization_id, kind, fact_key);
          CREATE TABLE knowledge_events (
            sequence INTEGER PRIMARY KEY, organization_id TEXT NOT NULL,
            fact_key TEXT NOT NULL, payload TEXT NOT NULL CHECK(json_valid(payload))
          ) STRICT;
          CREATE INDEX knowledge_event_fact ON knowledge_events (organization_id, fact_key, sequence);
          PRAGMA user_version = 1;
        `);
      } else if (version !== 1) throw new Error("Unsupported local knowledge graph schema version");
      db.exec(
        "CREATE INDEX IF NOT EXISTS knowledge_source ON knowledge_records (organization_id, kind, json_extract(payload, '$.source.connectorId'), json_extract(payload, '$.source.externalId'))",
      );
      await retryBusy(() => db.exec("COMMIT"));
    } catch (error) {
      try {
        db.exec("ROLLBACK");
      } catch {
        /* No transaction may have started. */
      }
      throw error;
    } finally {
      db.close();
    }
  }

  async transaction<T>(
    organizationId: string,
    work: (unit: KnowledgeGraphUnitOfWork) => Promise<T>,
  ): Promise<T> {
    requireOrganization(organizationId);
    const db = this.connection();
    const unit = new SqliteKnowledgeUnit(db, organizationId);
    try {
      await retryBusy(() => db.exec("BEGIN IMMEDIATE"));
      requireCurrentSchema(db);
      const result = await work(unit);
      unit.close();
      // Retry lock acquisition/commit only; never replay caller mutations.
      await retryBusy(() => db.exec("COMMIT"));
      return result;
    } catch (error) {
      try {
        db.exec("ROLLBACK");
      } catch {
        /* Preserve the original failure. */
      }
      throw error;
    } finally {
      unit.close();
      db.close();
    }
  }

  private async read<T>(
    organizationId: string,
    work: (unit: SqliteKnowledgeUnit) => Promise<T>,
  ): Promise<T> {
    requireOrganization(organizationId);
    const db = this.connection();
    const unit = new SqliteKnowledgeUnit(db, organizationId);
    try {
      return await work(unit);
    } finally {
      unit.close();
      db.close();
    }
  }
  getResourceBySource(org: string, source: ResourceSource) {
    return this.read(org, (unit) => unit.getResourceBySource(source));
  }
  getResource(org: string, id: string) {
    return this.read(org, (unit) => unit.getResource(id));
  }
  listResourcesByOntologyNode(org: string, id: string) {
    return this.read(org, (unit) => unit.listResourcesByOntologyNode(id));
  }
  listChunks(org: string, id: string) {
    return this.read(org, (unit) => unit.listChunks(id));
  }
  listEntitiesForResource(org: string, id: string) {
    return this.read(org, (unit) => unit.listEntities(id));
  }
  listEntityMentions(org: string, id: string) {
    return this.read(org, (unit) => unit.listEntityMentions(id));
  }
  getFact(org: string, id: string) {
    return this.read(org, (unit) => unit.getFact(id));
  }
  listFacts(org: string) {
    return this.read(org, (unit) => unit.listFacts());
  }
  listEvidenceForFact(org: string, id: string) {
    return this.read(org, (unit) => unit.listEvidenceForFact(id));
  }
  listFactEvents(org: string, id: string) {
    return this.read(org, (unit) => unit.listFactEvents(id));
  }
}

function requireOrganization(value: string): void {
  if (!value.trim()) throw new Error("Organization ID is required");
}
function requireCurrentSchema(db: DatabaseSync): void {
  if (db.prepare("PRAGMA user_version").get()?.user_version !== 1)
    throw new Error("Unsupported local knowledge graph schema version");
}
async function retryBusy(action: () => void): Promise<void> {
  for (let attempt = 0; ; attempt++) {
    try {
      action();
      return;
    } catch (error) {
      const code = error instanceof Error && "errcode" in error ? error.errcode : undefined;
      if (typeof code !== "number" || ![5, 6].includes(code & 255) || attempt >= 400) throw error;
      // Blocking SQLite busy timeouts would deadlock concurrent async transactions in one process.
      await delay(10);
    }
  }
}
