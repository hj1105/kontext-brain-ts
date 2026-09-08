import type { Dir } from "node:fs";
import { opendir } from "node:fs/promises";

/** Bounded discovery only; each owning store must still decode and verify its records. */
export async function listHashedRecordFiles(directory: string): Promise<string[]> {
  let entries: Dir;
  try {
    entries = await opendir(directory);
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") return [];
    throw error;
  }
  const files: string[] = [];
  let inspected = 0;
  for await (const entry of entries) {
    if (++inspected > 10_000) throw new Error("Record inventory exceeds its bounded scan limit");
    if (!/^[a-f0-9]{64}\.json$/.test(entry.name)) continue;
    if (!entry.isFile()) throw new Error("Record inventory contains a non-file record");
    files.push(entry.name);
  }
  return files.sort();
}
