import { createHash, randomUUID } from "node:crypto";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { promisify } from "node:util";
import { gunzip, gzip } from "node:zlib";
import type { ResourceContentStore, StoredResourceContent } from "./ports.js";

const gzipAsync = promisify(gzip);
const gunzipAsync = promisify(gunzip);

export class FileResourceContentStore implements ResourceContentStore {
  private readonly root: string;

  constructor(root: string) {
    this.root = path.resolve(root);
  }

  async put(content: StoredResourceContent): Promise<string> {
    const objectKey = [
      "objects-v2",
      digest(content.organizationId),
      digest(content.resourceId),
      `${digest(content.contentHash)}.json.gz`,
    ].join("/");
    const file = this.resolveKey(objectKey);
    await fs.mkdir(path.dirname(file), { recursive: true, mode: 0o700 });
    const temporary = `${file}.${process.pid}.${randomUUID()}.tmp`;
    try {
      const encoded = await gzipAsync(Buffer.from(JSON.stringify(content), "utf8"));
      await fs.writeFile(temporary, encoded, { flag: "wx", mode: 0o600 });
      await fs.rename(temporary, file);
      return objectKey;
    } finally {
      await fs.unlink(temporary).catch(() => undefined);
    }
  }

  async get(objectKey: string): Promise<StoredResourceContent | null> {
    const file = this.resolveKey(objectKey);
    try {
      const compressed = await fs.readFile(file);
      const decoded = await gunzipAsync(compressed);
      return JSON.parse(decoded.toString("utf8")) as StoredResourceContent;
    } catch (error) {
      if (isFileNotFound(error)) return null;
      throw error;
    }
  }

  async purge(objectKey: string): Promise<void> {
    const file = this.resolveKey(objectKey);
    try {
      await fs.unlink(file);
    } catch (error) {
      if (!isFileNotFound(error)) throw error;
    }
  }

  private resolveKey(objectKey: string): string {
    if (path.isAbsolute(objectKey) || !objectKey.endsWith(".json.gz")) {
      throw new Error("Invalid object key");
    }
    const resolved = path.resolve(this.root, objectKey);
    const relative = path.relative(this.root, resolved);
    if (relative.startsWith("..") || path.isAbsolute(relative)) {
      throw new Error("Invalid object key");
    }
    return resolved;
  }
}

function digest(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function isFileNotFound(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error as { code?: unknown }).code === "ENOENT"
  );
}
