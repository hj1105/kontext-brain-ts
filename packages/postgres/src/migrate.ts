import { readFile, readdir } from "node:fs/promises";
import type { Pool } from "pg";

export async function migratePostgres(pool: Pool): Promise<void> {
  const migrationsUrl = new URL("../migrations/", import.meta.url);
  const migrationNames = (await readdir(migrationsUrl))
    .filter((name) => /^\d+_[a-z0-9_]+\.sql$/.test(name))
    .sort();
  for (const migrationName of migrationNames) {
    const sql = await readFile(new URL(migrationName, migrationsUrl), "utf8");
    await pool.query(sql);
  }
}
