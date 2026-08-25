import {
  type AnyTraversalScorePolicy,
  BalancedTraversalScorePolicy,
  CalibratedTraversalScorePolicy,
  type Principal,
  type TraversalScorePolicyResolver,
  type TraversalScoringProfile,
  scoringProfileDigest,
  validateTraversalScoringProfile,
} from "@kontext-brain/core";
import type { Pool, QueryResultRow } from "pg";
import { withOrganizationTransaction } from "./postgres-knowledge-graph.js";
import { toIsoString } from "./postgres-value-utils.js";

export type ScoringProfileDeploymentStatus = "staged" | "active" | "retired" | "failed";

export interface ScoringProfileDeployment {
  readonly organizationId: string;
  readonly profile: TraversalScoringProfile;
  readonly profileDigest: string;
  readonly status: ScoringProfileDeploymentStatus;
  readonly failure?: string;
  readonly createdAt: string;
  readonly evaluationSummary?: Readonly<Record<string, unknown>>;
}

export interface ScoringProfileActivationOptions {
  /** Break-glass only; normal activation requires a persisted evaluation summary. */
  readonly allowUnevaluated?: boolean;
}

/**
 * Stores immutable scoring profiles and resolves one profile per retrieval.
 * The short cache avoids a profile lookup on every question; activation clears
 * this process's entry immediately and other processes converge after the TTL.
 */
export class PostgresScoringProfileRepository implements TraversalScorePolicyResolver {
  private readonly activeCache = new Map<
    string,
    {
      readonly expiresAt: number;
      readonly policy: AnyTraversalScorePolicy | null;
      readonly canaryPercent: number;
    }
  >();
  private readonly shadowCache = new Map<
    string,
    { readonly expiresAt: number; readonly policy: AnyTraversalScorePolicy | null }
  >();

  constructor(
    private readonly pool: Pool,
    private readonly cacheTtlMs = 5_000,
    private readonly fallbackPolicy: () => AnyTraversalScorePolicy = () =>
      new BalancedTraversalScorePolicy(),
    private readonly now: () => number = () => Date.now(),
  ) {}

  async resolve(principal: Principal): Promise<AnyTraversalScorePolicy> {
    const cached = this.activeCache.get(principal.organizationId);
    const selection =
      cached && cached.expiresAt > this.now()
        ? cached
        : await this.loadActiveSelection(principal.organizationId);
    if (selection.policy && inDeterministicCanary(principal, selection.canaryPercent)) {
      return selection.policy;
    }
    return this.fallbackPolicy();
  }

  async setCanaryPercent(organizationId: string, percent: number): Promise<void> {
    if (!Number.isInteger(percent) || percent < 0 || percent > 100) {
      throw new Error("Scoring canary percent must be an integer between 0 and 100");
    }
    await withOrganizationTransaction(this.pool, organizationId, async (client) => {
      await client.query(
        `INSERT INTO kontext_organization_runtime (organization_id, scoring_canary_percent)
         VALUES ($1,$2)
         ON CONFLICT (organization_id) DO UPDATE SET
           scoring_canary_percent = EXCLUDED.scoring_canary_percent`,
        [organizationId, percent],
      );
    });
    this.activeCache.delete(organizationId);
  }

  async resolveShadow(principal: Principal): Promise<AnyTraversalScorePolicy | null> {
    const cached = this.shadowCache.get(principal.organizationId);
    if (cached && cached.expiresAt > this.now()) return cached.policy;
    const shadow = await this.getShadow(principal.organizationId);
    const policy = shadow ? new CalibratedTraversalScorePolicy(shadow.profile) : null;
    this.shadowCache.set(principal.organizationId, {
      expiresAt: this.now() + this.cacheTtlMs,
      policy,
    });
    return policy;
  }

  async getActive(organizationId: string): Promise<ScoringProfileDeployment | null> {
    return withOrganizationTransaction(
      this.pool,
      organizationId,
      async (client) => {
        const result = await client.query(
          `SELECT profile.*
           FROM kontext_organization_runtime runtime
           JOIN kontext_scoring_profiles profile
             ON profile.organization_id = runtime.organization_id
            AND profile.profile_digest = runtime.active_scoring_profile_digest
           WHERE runtime.organization_id = $1`,
          [organizationId],
        );
        return result.rows[0] ? mapDeployment(result.rows[0]) : null;
      },
      { readOnly: true },
    );
  }

  private async loadActiveSelection(organizationId: string): Promise<{
    readonly expiresAt: number;
    readonly policy: AnyTraversalScorePolicy | null;
    readonly canaryPercent: number;
  }> {
    const selection = await withOrganizationTransaction(
      this.pool,
      organizationId,
      async (client) => {
        const result = await client.query(
          `SELECT profile.*, runtime.scoring_canary_percent
           FROM kontext_organization_runtime runtime
           LEFT JOIN kontext_scoring_profiles profile
             ON profile.organization_id = runtime.organization_id
            AND profile.profile_digest = runtime.active_scoring_profile_digest
           WHERE runtime.organization_id = $1`,
          [organizationId],
        );
        const row = result.rows[0];
        return {
          policy:
            row?.profile_digest == null
              ? null
              : new CalibratedTraversalScorePolicy(mapDeployment(row).profile),
          canaryPercent:
            row?.scoring_canary_percent == null ? 100 : Number(row.scoring_canary_percent),
        };
      },
      { readOnly: true },
    );
    const cached = {
      ...selection,
      expiresAt: this.now() + this.cacheTtlMs,
    };
    this.activeCache.set(organizationId, cached);
    return cached;
  }

  async getShadow(organizationId: string): Promise<ScoringProfileDeployment | null> {
    return withOrganizationTransaction(
      this.pool,
      organizationId,
      async (client) => {
        const result = await client.query(
          `SELECT profile.*
           FROM kontext_organization_runtime runtime
           JOIN kontext_scoring_profiles profile
             ON profile.organization_id = runtime.organization_id
            AND profile.profile_digest = runtime.shadow_scoring_profile_digest
           WHERE runtime.organization_id = $1`,
          [organizationId],
        );
        return result.rows[0] ? mapDeployment(result.rows[0]) : null;
      },
      { readOnly: true },
    );
  }

  async stage(
    organizationId: string,
    profile: TraversalScoringProfile,
    evaluationSummary?: Readonly<Record<string, unknown>>,
  ): Promise<ScoringProfileDeployment> {
    validateTraversalScoringProfile(profile);
    const profileDigest = scoringProfileDigest(profile);
    const createdAt = new Date().toISOString();
    const deployment = await withOrganizationTransaction(
      this.pool,
      organizationId,
      async (client) => {
        const result = await client.query(
          `INSERT INTO kontext_scoring_profiles (
             organization_id, profile_id, version, feature_schema_version, profile_digest, profile_data,
             evaluation_summary, status, created_at
           ) VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7::jsonb,'staged',$8)
           ON CONFLICT (organization_id, profile_digest) DO UPDATE SET
             profile_data = EXCLUDED.profile_data,
             evaluation_summary = EXCLUDED.evaluation_summary,
             status = CASE
               WHEN kontext_scoring_profiles.status = 'active' THEN 'active'
               ELSE 'staged'
             END,
             failure = NULL
           RETURNING *`,
          [
            organizationId,
            profile.id,
            profile.version,
            profile.featureSchemaVersion,
            profileDigest,
            JSON.stringify(profile),
            evaluationSummary === undefined ? null : JSON.stringify(evaluationSummary),
            createdAt,
          ],
        );
        const row = result.rows[0];
        if (!row) throw new Error(`Scoring profile "${profile.id}" was not staged`);
        return mapDeployment(row);
      },
    );
    return deployment;
  }

  async activate(
    organizationId: string,
    profileDigest: string,
    options: ScoringProfileActivationOptions = {},
  ): Promise<ScoringProfileDeployment> {
    const deployment = await withOrganizationTransaction(
      this.pool,
      organizationId,
      async (client) => {
        const locked = await client.query(
          `SELECT status, evaluation_summary
           FROM kontext_scoring_profiles
           WHERE organization_id = $1 AND profile_digest = $2
           FOR UPDATE`,
          [organizationId, profileDigest],
        );
        if (locked.rows[0]?.status !== "staged") {
          throw new Error(`Scoring profile "${profileDigest}" is not staged`);
        }
        if (locked.rows[0]?.evaluation_summary == null && !options.allowUnevaluated) {
          throw new Error(`Scoring profile "${profileDigest}" has no evaluation summary`);
        }
        await client.query(
          `UPDATE kontext_scoring_profiles
           SET status = 'retired'
           WHERE organization_id = $1 AND status = 'active' AND profile_digest <> $2`,
          [organizationId, profileDigest],
        );
        const activated = await client.query(
          `UPDATE kontext_scoring_profiles
           SET status = 'active', failure = NULL
           WHERE organization_id = $1 AND profile_digest = $2
           RETURNING *`,
          [organizationId, profileDigest],
        );
        await client.query(
          `INSERT INTO kontext_organization_runtime (
             organization_id, active_scoring_profile_digest
           ) VALUES ($1,$2)
           ON CONFLICT (organization_id) DO UPDATE SET
             active_scoring_profile_digest = EXCLUDED.active_scoring_profile_digest`,
          [organizationId, profileDigest],
        );
        await client.query(
          `UPDATE kontext_organization_runtime
           SET shadow_scoring_profile_digest = NULL
           WHERE organization_id = $1 AND shadow_scoring_profile_digest = $2`,
          [organizationId, profileDigest],
        );
        const row = activated.rows[0];
        if (!row) throw new Error(`Scoring profile "${profileDigest}" was not activated`);
        return mapDeployment(row);
      },
    );
    this.activeCache.delete(organizationId);
    this.shadowCache.delete(organizationId);
    return deployment;
  }

  async rollback(organizationId: string, profileDigest: string): Promise<ScoringProfileDeployment> {
    const deployment = await withOrganizationTransaction(
      this.pool,
      organizationId,
      async (client) => {
        const target = await client.query(
          `SELECT status FROM kontext_scoring_profiles
           WHERE organization_id = $1 AND profile_digest = $2
           FOR UPDATE`,
          [organizationId, profileDigest],
        );
        if (!target.rows[0] || target.rows[0].status === "failed") {
          throw new Error(`Scoring profile "${profileDigest}" is not available for rollback`);
        }
        await client.query(
          `UPDATE kontext_scoring_profiles
           SET status = 'retired'
           WHERE organization_id = $1 AND status = 'active' AND profile_digest <> $2`,
          [organizationId, profileDigest],
        );
        const restored = await client.query(
          `UPDATE kontext_scoring_profiles
           SET status = 'active', failure = NULL
           WHERE organization_id = $1 AND profile_digest = $2
           RETURNING *`,
          [organizationId, profileDigest],
        );
        await client.query(
          `INSERT INTO kontext_organization_runtime (
             organization_id, active_scoring_profile_digest, scoring_canary_percent
           ) VALUES ($1,$2,100)
           ON CONFLICT (organization_id) DO UPDATE SET
             active_scoring_profile_digest = EXCLUDED.active_scoring_profile_digest,
             scoring_canary_percent = 100`,
          [organizationId, profileDigest],
        );
        await client.query(
          `UPDATE kontext_organization_runtime
           SET shadow_scoring_profile_digest = NULL
           WHERE organization_id = $1 AND shadow_scoring_profile_digest = $2`,
          [organizationId, profileDigest],
        );
        const row = restored.rows[0];
        if (!row) throw new Error(`Scoring profile "${profileDigest}" was not restored`);
        return mapDeployment(row);
      },
    );
    this.activeCache.delete(organizationId);
    this.shadowCache.delete(organizationId);
    return deployment;
  }

  async setShadow(organizationId: string, profileDigest: string | null): Promise<void> {
    await withOrganizationTransaction(this.pool, organizationId, async (client) => {
      if (profileDigest !== null) {
        const profile = await client.query(
          `SELECT 1 FROM kontext_scoring_profiles
           WHERE organization_id = $1 AND profile_digest = $2 AND status <> 'failed'`,
          [organizationId, profileDigest],
        );
        if (profile.rowCount !== 1) {
          throw new Error(`Scoring profile "${profileDigest}" is not available for shadowing`);
        }
        const active = await client.query(
          `SELECT 1 FROM kontext_organization_runtime
           WHERE organization_id = $1 AND active_scoring_profile_digest = $2`,
          [organizationId, profileDigest],
        );
        if (active.rowCount === 1) {
          throw new Error("Shadow scoring profile must differ from the active profile");
        }
      }
      await client.query(
        `INSERT INTO kontext_organization_runtime (
           organization_id, shadow_scoring_profile_digest
         ) VALUES ($1,$2)
         ON CONFLICT (organization_id) DO UPDATE SET
           shadow_scoring_profile_digest = EXCLUDED.shadow_scoring_profile_digest`,
        [organizationId, profileDigest],
      );
    });
    this.shadowCache.delete(organizationId);
  }

  async markFailed(organizationId: string, profileDigest: string, failure: string): Promise<void> {
    await withOrganizationTransaction(this.pool, organizationId, async (client) => {
      await client.query(
        `UPDATE kontext_scoring_profiles
         SET status = 'failed', failure = $3
         WHERE organization_id = $1 AND profile_digest = $2 AND status <> 'active'`,
        [organizationId, profileDigest, failure],
      );
    });
  }
}

function mapDeployment(row: QueryResultRow): ScoringProfileDeployment {
  const profile = row.profile_data as TraversalScoringProfile;
  validateTraversalScoringProfile(profile);
  const expectedDigest = scoringProfileDigest(profile);
  const storedDigest = String(row.profile_digest);
  if (String(row.feature_schema_version) !== profile.featureSchemaVersion) {
    throw new Error("Scoring profile feature schema column does not match profile data");
  }
  if (storedDigest !== expectedDigest) {
    throw new Error(
      `Scoring profile digest mismatch: stored ${storedDigest}, computed ${expectedDigest}`,
    );
  }
  return {
    organizationId: String(row.organization_id),
    profile,
    profileDigest: storedDigest,
    status: row.status as ScoringProfileDeploymentStatus,
    failure: row.failure === null ? undefined : String(row.failure),
    createdAt: toIsoString(row.created_at),
    evaluationSummary:
      row.evaluation_summary === null || row.evaluation_summary === undefined
        ? undefined
        : (row.evaluation_summary as Readonly<Record<string, unknown>>),
  };
}

function inDeterministicCanary(principal: Principal, percent: number): boolean {
  if (percent <= 0) return false;
  if (percent >= 100) return true;
  const value = `${principal.organizationId}:${principal.subjectId}`;
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash % 100 < percent;
}
