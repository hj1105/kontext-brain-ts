import { createHash, randomUUID } from "node:crypto";
import { open } from "node:fs/promises";
import { request as httpRequest } from "node:http";
import path from "node:path";
import { z } from "zod";
import {
  kontextSessionSourcePayloadSchema,
  kontextSessionSourcePreviewSchema,
} from "./native-session-source-contract.js";

const descriptorSchema = z
  .object({
    schemaVersion: z.literal(1),
    runtimeId: z.string().min(1).max(4096),
    instanceId: z.string().uuid(),
    endpoint: z.string().regex(/^http:\/\/127\.0\.0\.1:[1-9][0-9]{0,4}\/v1\/session$/),
    token: z.string().regex(/^[a-f0-9]{64}$/),
  })
  .strict();
export const nativeSessionOriginSchema = kontextSessionSourcePayloadSchema.shape.origin;
export type NativeSessionOrigin = z.infer<typeof nativeSessionOriginSchema>;
export const nativeSessionRegistrationSchema = z
  .object({
    origin: nativeSessionOriginSchema,
    expectedContentDigest: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  })
  .strict();

async function readDescriptor(file: string): Promise<string> {
  const handle = await open(file, "r");
  try {
    if (!(await handle.stat()).isFile())
      throw new Error("Native source reader configuration is invalid");
    const buffer = Buffer.alloc(16 * 1024 + 1);
    let length = 0;
    while (length < buffer.length) {
      const read = await handle.read(buffer, length, buffer.length - length, length);
      if (read.bytesRead === 0) break;
      length += read.bytesRead;
    }
    if (length === buffer.length) throw new Error("Native source reader configuration is invalid");
    return new TextDecoder("utf-8", { fatal: true }).decode(buffer.subarray(0, length));
  } finally {
    await handle.close();
  }
}

export async function readNativeSessionSource(directory: string, input: NativeSessionOrigin) {
  const origin = nativeSessionOriginSchema.parse(input);
  const file = path.join(directory, "knowledge", "native-source-reader.json");
  const bytes = await readDescriptor(file);
  if (Buffer.byteLength(bytes, "utf8") > 16 * 1024)
    throw new Error("Native source reader configuration is invalid");
  const descriptor = descriptorSchema.parse(JSON.parse(bytes));
  const requestId = randomUUID();
  const response = await new Promise<string>((resolve, reject) => {
    const request = httpRequest(
      descriptor.endpoint,
      {
        method: "POST",
        agent: false,
        signal: AbortSignal.timeout(5000),
        headers: {
          authorization: `Bearer ${descriptor.token}`,
          "content-type": "application/json",
        },
      },
      (incoming) => {
        if (incoming.statusCode !== 200) {
          incoming.destroy();
          reject(new Error("Native session source unavailable"));
          return;
        }
        const chunks: Buffer[] = [];
        let length = 0;
        incoming.on("data", (chunk: Buffer) => {
          length += chunk.length;
          if (length > 600 * 1024) {
            incoming.destroy(new Error("Native session source response too large"));
            return;
          }
          chunks.push(chunk);
        });
        incoming.on("error", reject);
        incoming.on("end", () => {
          try {
            resolve(new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks)));
          } catch (error) {
            reject(error);
          }
        });
      },
    );
    request.setTimeout(5000, () => request.destroy(new Error("Native source reader timed out")));
    request.on("error", () => reject(new Error("Native session source unavailable")));
    request.end(JSON.stringify({ requestId, sessionId: origin.sessionId }));
  });
  if ((await readDescriptor(file)) !== bytes)
    throw new Error("Native source reader changed during capture");
  const result = z
    .object({ requestId: z.literal(requestId), source: kontextSessionSourcePreviewSchema })
    .strict()
    .parse(JSON.parse(response));
  if (
    result.source.origin.runtimeId !== descriptor.runtimeId ||
    JSON.stringify({ ...result.source.origin, runtimeId: origin.runtimeId }) !==
      JSON.stringify(origin)
  )
    throw new Error("Native source origin changed");
  const payload = kontextSessionSourcePayloadSchema.parse(result.source);
  if (Buffer.byteLength(JSON.stringify(payload), "utf8") > 512 * 1024)
    throw new Error("Native session source exceeds 512 KiB");
  const digest = `sha256:${createHash("sha256").update(JSON.stringify(payload)).digest("hex")}`;
  if (digest !== result.source.contentDigest)
    throw new Error("Native session source content digest mismatch");
  return result.source;
}
