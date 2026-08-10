import { readFile } from "node:fs/promises";
import type { Pool } from "pg";

export async function migratePostgres(pool: Pool): Promise<void> {
  const migrationUrl = new URL("../migrations/0001_knowledge_graph.sql", import.meta.url);
  const sql = await readFile(migrationUrl, "utf8");
  await pool.query(sql);
}
