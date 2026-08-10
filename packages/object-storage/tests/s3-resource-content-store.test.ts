import { CreateBucketCommand, S3Client } from "@aws-sdk/client-s3";
import { describe, expect, it } from "vitest";
import { S3ResourceContentStore } from "../src/index.js";

const endpoint = process.env.KONTEXT_TEST_S3_ENDPOINT;

describe.runIf(endpoint !== undefined)("S3ResourceContentStore", () => {
  it("round-trips one compressed Resource object through an S3-compatible service", async () => {
    const client = new S3Client({
      endpoint,
      region: "us-east-1",
      forcePathStyle: true,
      credentials: { accessKeyId: "kontext", secretAccessKey: "kontext-secret" },
    });
    const bucket = `kontext-${Date.now()}`;
    await client.send(new CreateBucketCommand({ Bucket: bucket }));
    const store = new S3ResourceContentStore(client, { bucket });
    const content = {
      organizationId: "acme",
      resourceId: "slack:thread-1",
      contentHash: "v1",
      body: "Thread body",
      chunks: { "message-1": "Hello" },
    };

    const key = await store.put(content);

    expect(await store.get(key)).toEqual(content);
    await store.purge(key);
    expect(await store.get(key)).toBeNull();
    client.destroy();
  });
});
