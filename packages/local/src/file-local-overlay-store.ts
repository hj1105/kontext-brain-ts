import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  type NormativeManifest,
  decodeNormativeManifest,
  encodeNormativeManifest,
  normativeManifestDigest,
} from "@kontext-brain/spec";
import type {
  LocalNormativeOverlayStore,
  LocalOverlayKey,
  LocalOverlayWriteOptions,
  LocalOverlayWriteResult,
} from "./domain.js";
import { assertLocalNormativeManifest } from "./local-overlay-validation.js";

interface LocalOverlayEnvelope {
  readonly schemaVersion: 1;
  readonly key: LocalOverlayKey;
  readonly manifestDigest: string;
  readonly manifest: NormativeManifest;
}

export class FileLocalNormativeOverlayStore implements LocalNormativeOverlayStore {
  constructor(private readonly pluginDataDirectory: string) {}

  async load(key: LocalOverlayKey): Promise<NormativeManifest | undefined> {
    const filePath = this.filePath(key);
    let serialized: string;
    try {
      serialized = await readFile(filePath, "utf8");
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") return undefined;
      throw error;
    }
    const envelope = decodeEnvelope(serialized);
    if (stableKey(envelope.key) !== stableKey(key)) {
      throw new Error("Local normative overlay key does not match its storage location");
    }
    const manifest = decodeNormativeManifest(JSON.stringify(envelope.manifest));
    if (normativeManifestDigest(manifest) !== envelope.manifestDigest) {
      throw new Error("Local normative overlay manifest digest mismatch");
    }
    assertLocalNormativeManifest(key, manifest);
    return manifest;
  }

  async save(
    key: LocalOverlayKey,
    manifest: NormativeManifest,
    options: LocalOverlayWriteOptions = {},
  ): Promise<LocalOverlayWriteResult> {
    assertLocalNormativeManifest(key, manifest);
    const existing = await this.load(key);
    const existingDigest = existing ? normativeManifestDigest(existing) : undefined;
    if (options.expectedDigest !== undefined && options.expectedDigest !== existingDigest) {
      throw new Error("Local normative overlay changed since it was read");
    }

    const digest = normativeManifestDigest(manifest);
    const envelope: LocalOverlayEnvelope = {
      schemaVersion: 1,
      key,
      manifestDigest: digest,
      manifest: decodeNormativeManifest(encodeNormativeManifest(manifest)),
    };
    const directory = path.dirname(this.filePath(key));
    const temporaryPath = path.join(directory, `.${randomUUID()}.tmp`);
    await mkdir(directory, { recursive: true, mode: 0o700 });
    try {
      await writeFile(temporaryPath, `${JSON.stringify(envelope, null, 2)}\n`, {
        encoding: "utf8",
        mode: 0o600,
      });
      await rename(temporaryPath, this.filePath(key));
    } catch (error) {
      await rm(temporaryPath, { force: true }).catch(() => undefined);
      throw error;
    }
    return { digest, created: !existing };
  }

  filePath(key: LocalOverlayKey): string {
    return path.join(
      this.pluginDataDirectory,
      "normative-overlays",
      `${createHash("sha256").update(stableKey(key)).digest("hex")}.json`,
    );
  }
}

function decodeEnvelope(serialized: string): LocalOverlayEnvelope {
  const parsed: unknown = JSON.parse(serialized);
  if (
    !isRecord(parsed) ||
    parsed.schemaVersion !== 1 ||
    !isOverlayKey(parsed.key) ||
    typeof parsed.manifestDigest !== "string" ||
    !isRecord(parsed.manifest)
  ) {
    throw new Error("Invalid local normative overlay envelope");
  }
  return parsed as unknown as LocalOverlayEnvelope;
}

function isOverlayKey(value: unknown): value is LocalOverlayKey {
  return (
    isRecord(value) &&
    nonEmptyString(value.organizationId) &&
    nonEmptyString(value.subjectId) &&
    nonEmptyString(value.workspaceId) &&
    (value.codebaseId === undefined || nonEmptyString(value.codebaseId))
  );
}

function stableKey(key: LocalOverlayKey): string {
  return JSON.stringify([key.organizationId, key.subjectId, key.workspaceId, key.codebaseId ?? ""]);
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNodeError(value: unknown): value is NodeJS.ErrnoException {
  return value instanceof Error && "code" in value;
}
