import type { NormativeManifest } from "@kontext-brain/spec";

export interface LocalOverlayKey {
  readonly organizationId: string;
  readonly subjectId: string;
  readonly workspaceId: string;
  readonly codebaseId?: string;
}

export interface LocalOverlayWriteOptions {
  readonly expectedDigest?: string;
}

export interface LocalOverlayWriteResult {
  readonly digest: string;
  readonly created: boolean;
}

export interface LocalNormativeOverlayStore {
  load(key: LocalOverlayKey): Promise<NormativeManifest | undefined>;
  save(
    key: LocalOverlayKey,
    manifest: NormativeManifest,
    options?: LocalOverlayWriteOptions,
  ): Promise<LocalOverlayWriteResult>;
}
