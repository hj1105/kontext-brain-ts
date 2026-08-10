import { promisify } from "node:util";
import { gunzip, gzip } from "node:zlib";
import {
  DeleteObjectCommand,
  GetObjectCommand,
  PutObjectCommand,
  S3Client,
} from "@aws-sdk/client-s3";
import type { ResourceContentStore, StoredResourceContent } from "@kontext-brain/core";

const gzipAsync = promisify(gzip);
const gunzipAsync = promisify(gunzip);

export interface S3ResourceContentStoreOptions {
  readonly bucket: string;
  readonly prefix?: string;
}

export class S3ResourceContentStore implements ResourceContentStore {
  private readonly prefix: string;

  constructor(
    private readonly client: S3Client,
    private readonly options: S3ResourceContentStoreOptions,
  ) {
    this.prefix = normalizePrefix(options.prefix ?? "kontext/resources");
  }

  async put(content: StoredResourceContent): Promise<string> {
    const objectKey = [
      this.prefix,
      encodeURIComponent(content.organizationId),
      encodeURIComponent(content.resourceId),
      `${encodeURIComponent(content.contentHash)}.json.gz`,
    ]
      .filter(Boolean)
      .join("/");
    const body = await gzipAsync(Buffer.from(JSON.stringify(content), "utf8"));
    await this.client.send(
      new PutObjectCommand({
        Bucket: this.options.bucket,
        Key: objectKey,
        Body: body,
        ContentType: "application/json",
        ContentEncoding: "gzip",
        Metadata: { contentHash: content.contentHash },
      }),
    );
    return objectKey;
  }

  async get(objectKey: string): Promise<StoredResourceContent | null> {
    this.assertObjectKey(objectKey);
    try {
      const output = await this.client.send(
        new GetObjectCommand({ Bucket: this.options.bucket, Key: objectKey }),
      );
      if (!output.Body) return null;
      const compressed = Buffer.from(await output.Body.transformToByteArray());
      const decoded = await gunzipAsync(compressed);
      return JSON.parse(decoded.toString("utf8")) as StoredResourceContent;
    } catch (error) {
      if (isMissingObject(error)) return null;
      throw error;
    }
  }

  async purge(objectKey: string): Promise<void> {
    this.assertObjectKey(objectKey);
    await this.client.send(
      new DeleteObjectCommand({ Bucket: this.options.bucket, Key: objectKey }),
    );
  }

  private assertObjectKey(objectKey: string): void {
    if (!objectKey.startsWith(`${this.prefix}/`) || objectKey.includes("..")) {
      throw new Error("Invalid object key");
    }
  }
}

function normalizePrefix(prefix: string): string {
  const normalized = prefix.replace(/^\/+|\/+$/g, "");
  if (normalized.includes("..")) throw new Error("Invalid object prefix");
  return normalized;
}

function isMissingObject(error: unknown): boolean {
  if (typeof error !== "object" || error === null) return false;
  const value = error as { name?: unknown; $metadata?: { httpStatusCode?: unknown } };
  return value.name === "NoSuchKey" || value.$metadata?.httpStatusCode === 404;
}
