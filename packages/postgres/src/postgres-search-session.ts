import type { SearchGraphSession } from "@kontext-brain/core";
import type { Pool, PoolClient } from "pg";
import { withOrganizationTransaction } from "./postgres-knowledge-graph.js";
import { toConnectionError } from "./postgres-value-utils.js";

/** Internal PostgreSQL session shared by the graph and its database seed providers. */
export class PostgresSearchSession implements SearchGraphSession {
  private closed = false;
  private client: PoolClient | undefined;
  private opening: Promise<PoolClient> | undefined;

  private constructor(
    private readonly pool: Pool,
    private readonly organizationId: string,
  ) {}

  static async open(pool: Pool, organizationId: string): Promise<PostgresSearchSession> {
    // Acquisition is lazy so embedding/non-database seed preparation does not
    // occupy a pool slot or hold a transaction open.
    return new PostgresSearchSession(pool, organizationId);
  }

  private async openClient(): Promise<PoolClient> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN READ ONLY");
      await client.query("SELECT set_config('kontext.organization_id', $1, true)", [
        this.organizationId,
      ]);
      return client;
    } catch (error) {
      // The connection never became a usable session; do not leak it back dirty.
      try {
        await client.query("ROLLBACK");
      } catch (rollbackError) {
        client.release(toConnectionError(rollbackError));
        throw new AggregateError(
          [error, rollbackError],
          "PostgreSQL search session setup and rollback both failed",
        );
      }
      client.release();
      throw error;
    }
  }

  assertOrganization(organizationId: string): void {
    if (organizationId !== this.organizationId) {
      throw new Error(
        `PostgreSQL search session organization mismatch: expected "${this.organizationId}", received "${organizationId}"`,
      );
    }
  }

  usesPool(pool: Pool): boolean {
    return pool === this.pool;
  }

  async runRead<T>(
    pool: Pool,
    organizationId: string,
    work: (client: PoolClient) => Promise<T>,
  ): Promise<T> {
    if (this.closed) throw new Error("PostgreSQL search session is already closed");
    if (pool !== this.pool) throw new Error("PostgreSQL search session belongs to another pool");
    this.assertOrganization(organizationId);
    if (!this.client) {
      this.opening ??= this.openClient();
      try {
        this.client = await this.opening;
      } finally {
        this.opening = undefined;
      }
    }
    return work(this.client);
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    const client = this.client;
    this.client = undefined;
    if (!client) return;
    // Read-only work has nothing to commit, and COMMIT would mask a failed
    // transaction; ending with ROLLBACK returns a clean connection either way.
    try {
      await client.query("ROLLBACK");
    } catch (error) {
      client.release(toConnectionError(error));
      throw error;
    }
    client.release();
  }
}

/**
 * Runs PostgreSQL read work on a traversal session when available, or in its own
 * short read-only transaction otherwise. Custom database seed providers should
 * use this helper so they cannot deadlock a saturated pool with a second checkout.
 */
export async function runPostgresSearchRead<T>(
  pool: Pool,
  organizationId: string,
  session: SearchGraphSession | undefined,
  work: (client: PoolClient) => Promise<T>,
): Promise<T> {
  if (session instanceof PostgresSearchSession) {
    session.assertOrganization(organizationId);
    if (session.usesPool(pool)) return session.runRead(pool, organizationId, work);
  }
  return withOrganizationTransaction(pool, organizationId, work, { readOnly: true });
}
