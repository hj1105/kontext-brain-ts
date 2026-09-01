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

export class InMemoryLocalNormativeOverlayStore implements LocalNormativeOverlayStore {
  private readonly overlays = new Map<string, NormativeManifest>();

  async load(key: LocalOverlayKey): Promise<NormativeManifest | undefined> {
    const manifest = this.overlays.get(keyString(key));
    return manifest ? decodeNormativeManifest(encodeNormativeManifest(manifest)) : undefined;
  }

  async save(
    key: LocalOverlayKey,
    manifest: NormativeManifest,
    options: LocalOverlayWriteOptions = {},
  ): Promise<LocalOverlayWriteResult> {
    assertLocalNormativeManifest(key, manifest);
    const existing = this.overlays.get(keyString(key));
    const existingDigest = existing ? normativeManifestDigest(existing) : undefined;
    if (options.expectedDigest !== undefined && options.expectedDigest !== existingDigest) {
      throw new Error("Local normative overlay changed since it was read");
    }
    const normalized = decodeNormativeManifest(encodeNormativeManifest(manifest));
    const digest = normativeManifestDigest(normalized);
    this.overlays.set(keyString(key), normalized);
    return { digest, created: !existing };
  }
}

function keyString(key: LocalOverlayKey): string {
  return JSON.stringify([key.organizationId, key.subjectId, key.workspaceId, key.codebaseId ?? ""]);
}
