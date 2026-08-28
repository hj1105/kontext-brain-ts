import { BidirectionalNLayerRetriever, InMemoryResourceContentStore } from "@kontext-brain/core";
import type { Pool, PoolClient } from "pg";
import { describe, expect, it, vi } from "vitest";
import {
  PostgresChunkVectorIndex,
  PostgresKnowledgeSearchGraph,
  withOrganizationTransaction,
} from "../src/index.js";
import { PostgresSearchSession } from "../src/postgres-search-session.js";

const principal = {
  organizationId: "acme",
  subjectId: "user:1",
  groupIds: ["finance"],
};

function fakeDatabase(
  rowsFor: (statement: string) => unknown[] = () => [],
  errorFor: (statement: string) => unknown | undefined = () => undefined,
) {
  const statements: string[] = [];
  const release = vi.fn();
  const client = {
    async query(statement: string) {
      statements.push(statement);
      const error = errorFor(statement);
      if (error !== undefined) throw error;
      const rows = rowsFor(statement);
      return { rows, rowCount: rows.length };
    },
    release,
  } as unknown as PoolClient;
  const connect = vi.fn(async () => client);
  const pool = { connect } as unknown as Pool;
  return { pool, connect, release, statements };
}

describe("PostgresSearchSession", () => {
  it("shares one checked-out client across graph traversal and vector seeds", async () => {
    const database = fakeDatabase();
    const index = new PostgresChunkVectorIndex(
      database.pool,
      {
        async embed() {
          expect(database.connect).not.toHaveBeenCalled();
          return [0, 1];
        },
      },
      2,
    );

    const graph = new PostgresKnowledgeSearchGraph(
      database.pool,
      new InMemoryResourceContentStore(),
      [index],
    );

    await new BidirectionalNLayerRetriever(graph).retrieve({ question: "paid order", principal });

    expect(database.connect).toHaveBeenCalledTimes(1);
    expect(database.statements[0]).toBe("BEGIN READ ONLY");
    expect(database.statements.some((statement) => statement.includes("SELECT c.chunk_id"))).toBe(
      true,
    );
    expect(database.statements.at(-1)).toBe("ROLLBACK");
    expect(database.release).toHaveBeenCalledTimes(1);
  });

  it("opens a short read-only transaction when no traversal session is supplied", async () => {
    const database = fakeDatabase();
    const index = new PostgresChunkVectorIndex(
      database.pool,
      {
        async embed() {
          return [0, 1];
        },
      },
      2,
    );

    await index.seed("paid order", principal);

    expect(database.connect).toHaveBeenCalledTimes(1);
    expect(database.statements[0]).toBe("BEGIN READ ONLY");
    expect(database.statements.at(-1)).toBe("COMMIT");
    expect(database.release).toHaveBeenCalledTimes(1);
  });

  it("uses a separate provider pool without trying to reuse an incompatible session", async () => {
    const graphDatabase = fakeDatabase();
    const vectorDatabase = fakeDatabase();
    const session = await PostgresSearchSession.open(graphDatabase.pool, principal.organizationId);
    const index = new PostgresChunkVectorIndex(
      vectorDatabase.pool,
      {
        async embed() {
          return [0, 1];
        },
      },
      2,
    );

    await index.seed("paid order", principal, session);
    await session.close();

    expect(graphDatabase.connect).not.toHaveBeenCalled();
    expect(vectorDatabase.connect).toHaveBeenCalledTimes(1);
    expect(vectorDatabase.statements.at(-1)).toBe("COMMIT");
  });

  it("rejects cross-organization reuse and still releases the session", async () => {
    const database = fakeDatabase();
    const session = await PostgresSearchSession.open(database.pool, principal.organizationId);
    await session.runRead(database.pool, principal.organizationId, async () => undefined);

    await expect(session.runRead(database.pool, "other", async () => undefined)).rejects.toThrow(
      "organization mismatch",
    );
    await session.close();

    expect(database.release).toHaveBeenCalledTimes(1);
  });

  it("does not provision an organization during a read-only transaction", async () => {
    const database = fakeDatabase();

    await withOrganizationTransaction(
      database.pool,
      principal.organizationId,
      async () => undefined,
      { readOnly: true },
    );

    expect(database.statements[0]).toBe("BEGIN READ ONLY");
    expect(
      database.statements.some((statement) => statement.includes("kontext_organizations")),
    ).toBe(false);
    expect(database.statements.at(-1)).toBe("COMMIT");
  });

  it("discards a session connection when rollback fails", async () => {
    const rollbackError = new Error("rollback failed");
    const database = fakeDatabase(
      () => [],
      (statement) => (statement === "ROLLBACK" ? rollbackError : undefined),
    );
    const session = await PostgresSearchSession.open(database.pool, principal.organizationId);
    await session.runRead(database.pool, principal.organizationId, async () => undefined);

    await expect(session.close()).rejects.toBe(rollbackError);

    expect(database.release).toHaveBeenCalledWith(rollbackError);
  });

  it("discards a transaction connection and preserves both errors when rollback fails", async () => {
    const workError = new Error("work failed");
    const rollbackError = new Error("rollback failed");
    const database = fakeDatabase(
      () => [],
      (statement) => (statement === "ROLLBACK" ? rollbackError : undefined),
    );

    let thrown: unknown;
    try {
      await withOrganizationTransaction(database.pool, principal.organizationId, async () => {
        throw workError;
      });
    } catch (error) {
      thrown = error;
    }

    expect(thrown).toBeInstanceOf(AggregateError);
    expect((thrown as AggregateError).errors).toEqual([workError, rollbackError]);
    expect(database.release).toHaveBeenCalledWith(rollbackError);
  });

  it("keeps a bounded deterministic ontology fallback when lexical seeds are empty", async () => {
    const ontologyNodes = Array.from({ length: 20 }, (_, index) => ({
      id: `node-${String(index).padStart(2, "0")}`,
    }));
    const database = fakeDatabase((statement) =>
      statement.includes("SELECT deployment.graph_data")
        ? [{ graph_data: { nodes: ontologyNodes, edges: [] } }]
        : [],
    );
    const graph = new PostgresKnowledgeSearchGraph(
      database.pool,
      new InMemoryResourceContentStore(),
      [
        {
          async seed() {
            return [{ node: { kind: "chunk" as const, id: "weak" }, score: 0 }];
          },
        },
      ],
    );

    const seeds = await graph.seed("no lexical overlap", principal);
    const ontologySeeds = seeds.filter((seed) => seed.node.kind === "ontology");

    expect(seeds).toHaveLength(13);
    expect(ontologySeeds.map((seed) => seed.node.id)).toEqual(
      ontologyNodes.slice(0, 12).map((node) => node.id),
    );
    expect(ontologySeeds.every((seed) => seed.observations?.fallback === true)).toBe(true);
  });
});
