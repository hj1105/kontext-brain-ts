import {
  type GovernanceScope,
  type NormativeManifest,
  decodeNormativeManifest,
  encodeNormativeManifest,
  normativeManifestDigest,
} from "@kontext-brain/spec";
import type { Pool } from "pg";
import { withOrganizationTransaction } from "./postgres-knowledge-graph.js";

export interface ProjectNormativeManifestInput {
  readonly manifest: NormativeManifest;
  readonly sourceRepository: string;
  readonly sourceCommit: string;
  readonly projectedAt?: string;
}

export interface ProjectedNormativeManifest {
  readonly manifest: NormativeManifest;
  readonly manifestDigest: string;
  readonly sourceRepository: string;
  readonly sourceCommit: string;
  readonly projectedAt: string;
}

export class PostgresNormativeProjection {
  constructor(private readonly pool: Pool) {}

  async project(input: ProjectNormativeManifestInput): Promise<string> {
    assertManagedManifest(input.manifest);
    if (!input.sourceRepository.trim() || !input.sourceCommit.trim()) {
      throw new Error("Managed normative projection requires Git repository and commit provenance");
    }
    const normalized = decodeNormativeManifest(encodeNormativeManifest(input.manifest));
    const digest = normativeManifestDigest(normalized);
    const projectedAt = input.projectedAt ?? new Date().toISOString();

    return withOrganizationTransaction(this.pool, normalized.organizationId, async (client) => {
      await client.query(
        `INSERT INTO kontext_normative_manifests (
             organization_id, manifest_digest, manifest_data,
             source_repository, source_commit, projected_at
           ) VALUES ($1,$2,$3::jsonb,$4,$5,$6)
           ON CONFLICT (organization_id, manifest_digest) DO UPDATE SET
             source_repository = EXCLUDED.source_repository,
             source_commit = EXCLUDED.source_commit,
             projected_at = EXCLUDED.projected_at`,
        [
          normalized.organizationId,
          digest,
          JSON.stringify(normalized),
          input.sourceRepository,
          input.sourceCommit,
          projectedAt,
        ],
      );

      for (const revision of normalized.revisions) {
        const inserted = await client.query(
          `INSERT INTO kontext_normative_revisions (
               organization_id, kind, record_id, revision_id, revision_data,
               first_manifest_digest, projected_at
             ) VALUES ($1,$2,$3,$4,$5::jsonb,$6,$7)
             ON CONFLICT (organization_id, kind, record_id, revision_id) DO UPDATE SET
               projected_at = kontext_normative_revisions.projected_at
             WHERE kontext_normative_revisions.revision_data = EXCLUDED.revision_data
             RETURNING revision_id`,
          [
            normalized.organizationId,
            revision.kind,
            revision.recordId,
            revision.revisionId,
            JSON.stringify(revision),
            digest,
            projectedAt,
          ],
        );
        if (inserted.rowCount !== 1) {
          throw new Error(`Immutable normative revision collision: ${revision.revisionId}`);
        }
      }

      await client.query("DELETE FROM kontext_normative_activations WHERE organization_id = $1", [
        normalized.organizationId,
      ]);
      for (const activation of normalized.activations) {
        await client.query(
          `INSERT INTO kontext_normative_activations (
               organization_id, kind, record_id, scope_key, revision_id,
               activation_data, manifest_digest, projected_at
             ) VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8)`,
          [
            normalized.organizationId,
            activation.kind,
            activation.recordId,
            scopeKey(activation.scope),
            activation.revisionId,
            JSON.stringify(activation),
            digest,
            projectedAt,
          ],
        );
      }

      await client.query(
        `INSERT INTO kontext_normative_runtime (
             organization_id, current_manifest_digest, source_commit, projected_at
           ) VALUES ($1,$2,$3,$4)
           ON CONFLICT (organization_id) DO UPDATE SET
             current_manifest_digest = EXCLUDED.current_manifest_digest,
             source_commit = EXCLUDED.source_commit,
             projected_at = EXCLUDED.projected_at`,
        [normalized.organizationId, digest, input.sourceCommit, projectedAt],
      );
      return digest;
    });
  }

  async getCurrent(organizationId: string): Promise<ProjectedNormativeManifest | null> {
    return withOrganizationTransaction(this.pool, organizationId, async (client) => {
      const result = await client.query(
        `SELECT manifest.manifest_data, manifest.manifest_digest,
                manifest.source_repository, runtime.source_commit, runtime.projected_at
         FROM kontext_normative_runtime runtime
         JOIN kontext_normative_manifests manifest
           ON manifest.organization_id = runtime.organization_id
          AND manifest.manifest_digest = runtime.current_manifest_digest
         WHERE runtime.organization_id = $1`,
        [organizationId],
      );
      const row = result.rows[0];
      if (!row) return null;
      return {
        manifest: decodeNormativeManifest(JSON.stringify(row.manifest_data)),
        manifestDigest: String(row.manifest_digest),
        sourceRepository: String(row.source_repository),
        sourceCommit: String(row.source_commit),
        projectedAt:
          row.projected_at instanceof Date
            ? row.projected_at.toISOString()
            : new Date(String(row.projected_at)).toISOString(),
      };
    });
  }
}

function assertManagedManifest(manifest: NormativeManifest): void {
  encodeNormativeManifest(manifest);
  for (const revision of manifest.revisions) {
    if (revision.scope.kind !== "codebase" && revision.scope.kind !== "organization") {
      throw new Error("Managed normative projection accepts only Codebase or Organization scope");
    }
  }
  for (const activation of manifest.activations) {
    if (activation.state !== "accepted" && activation.state !== "retired") {
      throw new Error("Managed normative projection cannot enforce Local Acceptance");
    }
    if (activation.state === "accepted" && !activation.mergeCommit) {
      throw new Error("Managed normative activation requires merge commit provenance");
    }
  }
}

function scopeKey(scope: GovernanceScope): string {
  switch (scope.kind) {
    case "personal":
      return JSON.stringify(["personal", scope.subjectId]);
    case "workspace":
      return JSON.stringify(["workspace", scope.workspaceId]);
    case "codebase":
      return JSON.stringify(["codebase", scope.codebaseId]);
    case "organization":
      return JSON.stringify(["organization", scope.organizationId]);
  }
}
