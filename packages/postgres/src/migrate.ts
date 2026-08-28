import { readFile } from "node:fs/promises";
import type { Pool } from "pg";

export async function migratePostgres(pool: Pool): Promise<void> {
  for (const filename of [
    "0001_knowledge_graph.sql",
    "0002_scoring_profiles_and_observations.sql",
  ]) {
    const migrationUrl = new URL(`../migrations/${filename}`, import.meta.url);
    const sql = await readFile(migrationUrl, "utf8");
    await pool.query(sql);
  }
}
