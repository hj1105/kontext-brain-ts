import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

export function readJsonLines<T>(path: string): T[] {
  const contents = readFileSync(path, "utf8");
  const records: T[] = [];
  for (const [index, rawLine] of contents.split(/\r?\n/).entries()) {
    const line = rawLine.trim();
    if (!line) continue;
    try {
      records.push(JSON.parse(line) as T);
    } catch (error) {
      throw new Error(`Invalid JSONL at ${path}:${index + 1}: ${(error as Error).message}`);
    }
  }
  return records;
}

export function writeJsonLines(path: string, records: readonly unknown[]): void {
  mkdirSync(dirname(path), { recursive: true });
  const serialized = records.map((record) => JSON.stringify(record)).join("\n");
  writeFileSync(path, serialized ? `${serialized}\n` : "", "utf8");
}

export function writeJsonAtomic(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  const temporaryPath = `${path}.tmp`;
  writeFileSync(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  renameSync(temporaryPath, path);
}
