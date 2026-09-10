import { randomUUID } from "node:crypto";
import { link, mkdir, open, readFile, unlink } from "node:fs/promises";
import path from "node:path";
import type { Principal } from "@kontext-brain/core";
import { z } from "zod";

const identitySchema = z
  .object({
    schemaVersion: z.literal(1),
    organizationId: z.string().uuid(),
    subjectId: z.string().uuid(),
  })
  .strict();
export async function loadLocalKnowledgePrincipal(dataDirectory: string): Promise<Principal> {
  const directory = path.join(dataDirectory, "knowledge");
  await mkdir(directory, { recursive: true, mode: 0o700 });
  const filename = path.join(directory, "local-principal.json");
  try {
    return principal(JSON.parse(await readFile(filename, "utf8")));
  } catch (error) {
    if (!isMissing(error)) throw error;
  }
  const temporary = path.join(directory, `.${randomUUID()}.identity`);
  try {
    const file = await open(temporary, "wx", 0o600);
    try {
      await file.writeFile(
        JSON.stringify({ schemaVersion: 1, organizationId: randomUUID(), subjectId: randomUUID() }),
      );
      await file.sync();
    } finally {
      await file.close();
    }
    try {
      await link(temporary, filename);
    } catch (error) {
      if (!(error instanceof Error && "code" in error && error.code === "EEXIST")) throw error;
    }
    return principal(JSON.parse(await readFile(filename, "utf8")));
  } finally {
    await unlink(temporary).catch(() => undefined);
  }
}
function principal(value: unknown): Principal {
  const identity = identitySchema.parse(value);
  return { organizationId: identity.organizationId, subjectId: identity.subjectId, groupIds: [] };
}
function isMissing(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}
