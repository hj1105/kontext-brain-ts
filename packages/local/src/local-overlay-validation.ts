import { type NormativeManifest, encodeNormativeManifest } from "@kontext-brain/spec";
import type { LocalOverlayKey } from "./domain.js";

export function assertLocalNormativeManifest(
  key: LocalOverlayKey,
  manifest: NormativeManifest,
): void {
  encodeNormativeManifest(manifest);
  if (manifest.organizationId !== key.organizationId) {
    throw new Error("Local normative overlay Organization mismatch");
  }
  for (const revision of manifest.revisions) {
    if (revision.scope.kind !== "personal" && revision.scope.kind !== "workspace") {
      throw new Error("Local normative overlay accepts only Personal or Workspace revisions");
    }
  }
  for (const activation of manifest.activations) {
    if (activation.state !== "accepted_local" && activation.state !== "retired") {
      throw new Error("Local normative overlay cannot contain managed activations");
    }
  }
}
